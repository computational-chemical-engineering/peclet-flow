/// @file
/// @brief flow — VoF Part III rung W0: the per-bubble VoF BLOCK, a third container over the same
/// L1 kernels (`suite/docs/VOF_PLAN.md` §10, the TBFsolver `vofBlock` pattern).
///
/// ## What a block is, and why
///
/// One bubble = one `WyAdvector` on a small dense global index box, plus a master rank that owns
/// it. The global colour the closures see is the **union** `C = max_blocks C_block`, never the
/// source. Because each marker is transported on its own field, two bubbles that touch **cannot
/// coalesce numerically** — coalescence becomes an explicit model decision (rung W4) instead of a
/// numerical accident of a single global colour field. That is the whole point of the container;
/// everything else here is bookkeeping.
///
/// Lineage: Cifani et al., *Computers & Fluids* 2018 (TBFsolver, `src/VOF/vofBlocks.f90`) —
/// moving bounding box = bubble extent + a **3-cell offset**, a master rank assigned independently
/// of the flow decomposition, gather `u` to the master, run the whole VoF pipeline on the dense
/// block, scatter back with UNPACK_MAX. Also Coyajee & Boersma JCP 2009 and Balcázar et al. IJHFF
/// 2015 (multiple-marker VoF/CLSVOF). Nothing is transcribed from TBFsolver; the kernels are the
/// suite's own L1 set (`peclet::core::vof`) and the advection is Weymouth–Yue, not TBFsolver's
/// implicit-in-sweep scheme.
///
/// ## The three index boxes (get these right and the rest follows)
///
///   bubble box   tight bounding box of the cells this block's colour is > 0
///   INNER box    bubble box grown by `margin` (= 3) on every side — the advector's inner region
///   EXTENDED box inner box grown by `ghost` (= 3) — the advector's extended block
///
/// so the extended block reaches 6 cells beyond the bubble, exactly TBFsolver's offset. All three
/// are in GLOBAL cell indices and may hang outside `[0, gs)` on a periodic axis (the gather wraps).
///
/// **Why margin 3 is what makes the block conservative.** One `advect()` runs one sweep per axis,
/// so colour moves at most ONE cell along any axis per step. With the nonzero colour ≥ 3 cells from
/// the inner-box boundary at the start of a step, nothing can reach the boundary *and* be fluxed
/// across it, so no liquid ever leaves the block and `Σ C` over the inner box is conserved by the
/// same telescoping argument the global field enjoys. Re-centring (below) restores the margin every
/// time it is spent.
///
/// ## Ghost policy of a block (the reason G1 can be BITWISE)
///
/// A block's extended-box cells that lie inside the domain but outside its inner box are **pure
/// gas** — that is what the margin guarantees — so the ghost fill writes 0 there, which is exactly
/// the value the global-field advector holds at the same global cells. Cells outside the domain on
/// a non-periodic axis take the globally-clamped (zero-gradient) value, the same rule
/// `colour_field.hpp::clampFill` applies to the structured field. A block whose box spans a whole
/// periodic axis gets the periodic wrap on that axis instead. Since the face velocities are
/// gathered from the owners of the same global cells, every double the block's WY update consumes
/// is bit-for-bit the double the global-field update consumes — hence gate G1 is a *bitwise* gate
/// and not a tolerance.
///
/// ## What lives where
///
/// This header is MPI-free and container-agnostic index math + orchestration. The gather/scatter
/// (which needs the flow decomposition and MPI) is `vof/block_exchange.hpp`; it is injected through
/// `VofBlockExchangeBase` so this file compiles in the single-rank module unchanged.
#ifndef PECLET_FLOW_VOF_BLOCK_CONTAINER_HPP
#define PECLET_FLOW_VOF_BLOCK_CONTAINER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "peclet/core/decomp/block_decomposer.hpp"
#include "vof/advect_wy.hpp"
#include "vof/colour_field.hpp"
#include "vof/curvature_field.hpp"
#include "vof/surface_tension.hpp"

namespace peclet::flow::vof {

// ---------------------------------------------------------------------------------------------
// index boxes
// ---------------------------------------------------------------------------------------------

/// Half-open global cell box `[lo, hi)`. `lo` may be negative and `hi` may exceed the grid on a
/// periodic axis; the exchange wraps.
struct VofBox {
  int lo[3]{0, 0, 0};
  int hi[3]{0, 0, 0};

  int n(int d) const { return hi[d] - lo[d]; }
  long cells() const { return static_cast<long>(n(0)) * n(1) * n(2); }
  bool empty() const { return n(0) <= 0 || n(1) <= 0 || n(2) <= 0; }
  I3 size() const { return I3{n(0), n(1), n(2)}; }
  I3 origin() const { return I3{lo[0], lo[1], lo[2]}; }

  bool contains(const VofBox& o) const {
    for (int d = 0; d < 3; ++d)
      if (o.lo[d] < lo[d] || o.hi[d] > hi[d])
        return false;
    return true;
  }
  bool operator==(const VofBox& o) const {
    for (int d = 0; d < 3; ++d)
      if (lo[d] != o.lo[d] || hi[d] != o.hi[d])
        return false;
    return true;
  }
  bool operator!=(const VofBox& o) const { return !(*this == o); }

  static VofBox grown(const VofBox& b, int m) {
    VofBox r = b;
    for (int d = 0; d < 3; ++d) {
      r.lo[d] -= m;
      r.hi[d] += m;
    }
    return r;
  }
  /// Intersection in global coordinates (may be empty).
  static VofBox intersect(const VofBox& a, const VofBox& b) {
    VofBox r;
    for (int d = 0; d < 3; ++d) {
      r.lo[d] = a.lo[d] > b.lo[d] ? a.lo[d] : b.lo[d];
      r.hi[d] = a.hi[d] < b.hi[d] ? a.hi[d] : b.hi[d];
    }
    return r;
  }
};

/// One contiguous run of GLOBAL indices covered by a block's index range on one axis, with the
/// block-local index the run starts at. A range on a periodic axis that crosses the seam produces
/// two runs; a range on a non-periodic axis produces one (the part outside the domain is not
/// gathered — it is the block's own clamp fill).
struct VofRun {
  int g0 = 0;    ///< first GLOBAL index of the run
  int len = 0;   ///< number of cells
  int loc0 = 0;  ///< block-local index (0-based within `[a, b)`) the run starts at
};

/// Decompose `[a, b)` on one axis into global runs. Returns the count.
///
/// The runs partition the range in BLOCK-LOCAL index, not in global index: a range longer than the
/// periodic axis (which is exactly what the EXTENDED box of a block spanning the whole axis is)
/// revisits the same global cells, and that is correct — the extra local cells are the block's
/// periodic ghosts and must carry the wrapped owner's value.
inline int vofAxisRuns(int a, int b, int gs, bool periodic, VofRun out[5]) {
  int n = 0;
  if (b <= a)
    return 0;
  if (!periodic) {
    const int s = a < 0 ? 0 : a;
    const int e = b > gs ? gs : b;
    if (e > s)
      out[n++] = VofRun{s, e - s, s - a};
    return n;
  }
  if (b - a > 3 * gs)  // guarded by vofClampBox(); a block box is at most the whole axis
    throw std::runtime_error("peclet::flow::vof: block range exceeds three periodic images");
  int k = a;
  while (k < b && n < 5) {
    const int w = ((k % gs) + gs) % gs;
    int len = gs - w;
    if (len > b - k)
      len = b - k;
    out[n++] = VofRun{w, len, k - a};
    k += len;
  }
  return n;
}

/// Clamp a block box so it is representable: at most the whole axis on a periodic axis, inside the
/// domain on a non-periodic one.
inline VofBox vofClampBox(VofBox b, I3 gs, const std::array<bool, 3>& per) {
  const int g[3] = {gs.x, gs.y, gs.z};
  for (int d = 0; d < 3; ++d) {
    if (per[d]) {
      if (b.n(d) > g[d]) {
        b.lo[d] = 0;
        b.hi[d] = g[d];
      }
    } else {
      if (b.lo[d] < 0)
        b.lo[d] = 0;
      if (b.hi[d] > g[d])
        b.hi[d] = g[d];
      if (b.hi[d] <= b.lo[d]) {
        b.lo[d] = 0;
        b.hi[d] = g[d] < 1 ? 1 : g[d];
      }
    }
  }
  return b;
}

// ---------------------------------------------------------------------------------------------
// master assignment (rung W1 item a)
// ---------------------------------------------------------------------------------------------

/// How a block's MASTER rank is chosen. The assignment is deliberately independent of where the
/// bubble's cells live (that is the whole point of the container: bubble work is load-balanced
/// separately from the flow decomposition), so it is a pure function of the REPLICATED block
/// table and needs no communication in any of the three modes.
enum class VofMasterAssign {
  RoundRobin = 0,  ///< `id % size` — rung W0's assignment; blind to block SIZE
  Lpt = 1,         ///< longest-processing-time greedy on the block cell counts (rung W1 default)
  WeightedOrb = 2  ///< weighted ORB over a 1-D "block space" (`core::decomp::BlockDecomposer<1>`)
};

/// Longest-processing-time greedy (Graham 1969): heaviest block first onto the currently
/// least-loaded rank, ties to the LOWEST rank id. `4/3 − 1/(3p)` of optimal in the worst case,
/// and — the property the bitwise gates need — a deterministic pure function of the replicated
/// weights, identical on every rank without an exchange.
inline void vofAssignLpt(const std::vector<long>& w, int size, std::vector<int>& master) {
  const std::size_t n = w.size();
  master.assign(n, 0);
  if (n == 0 || size <= 1)
    return;
  std::vector<std::size_t> ord(n);
  for (std::size_t i = 0; i < n; ++i)
    ord[i] = i;
  // stable, and the comparator is a strict weak ordering on the weight alone, so equal weights
  // keep block-id order -> with equal blocks LPT reproduces the round robin exactly.
  std::stable_sort(ord.begin(), ord.end(),
                   [&](std::size_t a, std::size_t b) { return w[a] > w[b]; });
  std::vector<long> load(static_cast<std::size_t>(size), 0);
  for (std::size_t k = 0; k < n; ++k) {
    int best = 0;
    for (int r = 1; r < size; ++r)
      if (load[static_cast<std::size_t>(r)] < load[static_cast<std::size_t>(best)])
        best = r;
    master[ord[k]] = best;
    load[static_cast<std::size_t>(best)] += w[ord[k]];
  }
}

/// Weighted ORB over a 1-D block space: the blocks are laid out as a 1-D "grid" of `N` cells in
/// block-id order carrying their weights, and `core::decomp::BlockDecomposer<1>` recursively
/// bisects it into `size` CONTIGUOUS runs of balanced weight. This is the work order's
/// `BlockDecomposer::init(…, weights)` route, using the same partitioner the flow decomposition
/// uses. Its constraint (and its weakness here) is that a rank's blocks must be contiguous in id,
/// which LPT is free of.
inline void vofAssignOrb(const std::vector<long>& w, int size, std::vector<int>& master) {
  const std::size_t n = w.size();
  master.assign(n, 0);
  if (n == 0 || size <= 1)
    return;
  std::vector<peclet::core::Real> wr(n);
  for (std::size_t i = 0; i < n; ++i)
    wr[i] = static_cast<peclet::core::Real>(w[i]);
  peclet::core::decomp::BlockDecomposer<1> dec(
      static_cast<std::size_t>(size),
      peclet::core::IVec<1>{static_cast<peclet::core::Index>(n)}, wr);
  for (int r = 0; r < size; ++r) {
    const auto b = dec.block(static_cast<std::size_t>(r));
    for (auto i = b.origin[0]; i < b.origin[0] + b.size[0]; ++i)
      master[static_cast<std::size_t>(i)] = r;
  }
}

// ---------------------------------------------------------------------------------------------
// one block
// ---------------------------------------------------------------------------------------------

/// Per-block Lagrangian census, computed on the master (VOF_PLAN §10 W1 outputs, available at W0).
struct VofBlockStats {
  long id = 0;
  int master = 0;
  int lo[3]{0, 0, 0};
  int hi[3]{0, 0, 0};
  double volume = 0.0;  ///< Σ C over the inner box, in cell volumes
  double centroid[3]{0, 0, 0};
  double velocity[3]{0, 0, 0};  ///< d(centroid)/dt of the last `advect`
  /// Central second moments Σ C (x−x̄)(y−ȳ)/Σ C, order xx, yy, zz, xy, xz, yz (the deformation).
  double moment[6]{0, 0, 0, 0, 0, 0};
  bool recentred = false;  ///< the box moved at the end of the last step
  long cells = 0;          ///< inner-box cell count (the master's work)
  /// Cumulative colour dropped by re-centrings, i.e. the Weymouth-Yue round-off WISPS left in the
  /// bubble's wake that fell below `VofBlockSet::bubbleEps`. Never physical liquid (see the
  /// `bubbleEps` note); reported because a container that silently loses mass is not acceptable.
  double discarded = 0.0;
  /// Interface area of the marker, in cell-size units squared: the sum of the PLIC polygon areas
  /// over the inner box (`plicPolygon` + `polygonAreaCentroid` on the MYC normal, i.e. exactly the
  /// planes the curvature cascade reconstructs). Rung W1 item (d) — the gallery's per-bubble
  /// Lagrangian output. A sphere of radius R reads 4 pi R^2 to the PLIC discretization error
  /// (~1 % at R/h = 8), NOT exactly; it is a measure of the reconstructed surface, not of a fit.
  double area = 0.0;
};

/// One bubble's block. Every rank holds the (id, box, master) triple — the replicated table — and
/// ONLY the master allocates the advector and the colour.
class VofBlock {
 public:
  long id = 0;
  int master = 0;
  VofBox box;  ///< the INNER box, global
  /// The BUBBLE extent this block was seeded from (empty for a sphere seed, which paints its own
  /// colour). Used once, at seeding, to clip a neighbour's liquid out of the margin ring.
  VofBox seed_{};

  bool mine() const { return mine_; }
  WyAdvector& advector() { return adv_; }
  const WyAdvector& advector() const { return adv_; }
  VofBox extended(int ghost) const { return VofBox::grown(box, ghost); }
  const VofBlockStats& stats() const { return st_; }
  /// Rung W2: the block's own curvature cascade and the CSF face force it forms. Allocated only
  /// when `VofBlockSet::csfEnabled` is on and only on the master. `csfForce(c)` holds the force at
  /// the LOW face of each cell of the block's own extended array (flow's face convention), i.e.
  /// `sigma kappa_f (C(i) - C(i - s_c)) / h`, and is what the scatter sums into the global field.
  VofCurvature& curvature() { return curv_; }
  const VofCurvature& curvature() const { return curv_; }
  SField csfForce(int c) const { return f_[c]; }
  /// The non-colour state a master carries between steps: the previous centroid (so a migrated
  /// block's reported velocity has no gap) and whether it is valid. Four doubles; everything else
  /// in a block is either replicated (the table) or recomputed every step.
  void serializeAux(double out[4]) const {
    for (int d = 0; d < 3; ++d)
      out[d] = prevCentroid_[d];
    out[3] = hasPrev_ ? 1.0 : 0.0;
  }
  void deserializeAux(const double in[4]) {
    for (int d = 0; d < 3; ++d)
      prevCentroid_[d] = in[d];
    hasPrev_ = (in[3] != 0.0);
  }

  friend class VofBlockSet;

 private:
  bool mine_ = false;
  bool allocated_ = false;
  WyAdvector adv_;
  VofCurvature curv_;
  SField f_[3];
  VofBlockStats st_;
  double prevCentroid_[3]{0, 0, 0};
  bool hasPrev_ = false;
};

// ---------------------------------------------------------------------------------------------
// the exchange interface (implemented in vof/block_exchange.hpp)
// ---------------------------------------------------------------------------------------------

class VofBlockSet;

/// Gather / scatter / table replication. Kept abstract so `block_container.hpp` stays MPI-free and
/// the single-rank Python module builds it unchanged.
struct VofBlockExchangeBase {
  virtual ~VofBlockExchangeBase() = default;
  /// Fill every MASTER block's `faceU/V/W` over its EXTENDED box, from the face velocity the ranks
  /// own. Cells with no owner (outside a non-periodic domain) are left untouched — the block's own
  /// clamp fill supplies them.
  virtual void gatherFaceVel(std::vector<VofBlock>& blocks, int ghost) = 0;
  /// UNION the masters' inner colour into the caller's local colour patch: zero the inner region,
  /// then `C = max(C, C_block)` cell by cell (TBFsolver's UNPACK_MAX).
  virtual void scatterColourMax(std::vector<VofBlock>& blocks, SField cLocal) = 0;
  /// Replicate the (possibly re-centred) boxes so every rank's table agrees again.
  virtual void syncTable(std::vector<VofBlock>& blocks) = 0;
  /// Fill every MASTER block's INNER colour from the caller's colour patch (the mirror of
  /// `scatterColourMax`). Used to seed a block from an arbitrary global colour field — the general
  /// seeding path, which a sphere seed is only a convenience over.
  virtual void gatherColour(std::vector<VofBlock>& blocks, SField cLocal) = 0;
  /// SUM every master block's CSF face force into the caller's three face-force patches
  /// (TBFsolver's UNPACK_SUM, `VOF.f90::computeSurfaceTension`): overlapping markers ADD their
  /// forces, which is what makes two touching bubbles push on the fluid twice and not once.
  virtual void scatterForceSum(std::vector<VofBlock>& blocks, SField fx, SField fy, SField fz) = 0;
  /// Move a block's state from `oldMaster[i]` to `blocks[i].master` after a re-assignment. The
  /// gaining rank has ALREADY allocated the (zeroed) advector; only the colour travels — everything
  /// else in a block is either replicated (the table) or recomputed per step.
  virtual void migrateColour(std::vector<VofBlock>& blocks, const std::vector<int>& oldMaster) = 0;
  /// Bytes moved by the last gather / scatter on THIS rank (sent + received).
  virtual long gatherBytes() const { return 0; }
  virtual long scatterBytes() const { return 0; }
};

// ---------------------------------------------------------------------------------------------
// the block set
// ---------------------------------------------------------------------------------------------

/// The container: a replicated table of blocks, the per-step orchestration (gather → advect →
/// re-centre → scatter/union) and the per-bubble statistics.
class VofBlockSet {
 public:
  /// @param gs   global grid size in cells
  /// @param per  periodicity per axis (an axis with a domain BC on either face is NOT periodic)
  /// @param rank,size  this rank's id and the communicator size (1 single-rank)
  /// @param h    cell size (flow works in cell units, h = 1; the standalone scenes use 1/N so
  ///             they can reuse the structured V1 scene builders verbatim)
  void init(I3 gs, std::array<bool, 3> per, int rank, int size, double h = 1.0) {
    h_ = h;
    gs_ = gs;
    per_ = per;
    rank_ = rank;
    size_ = size < 1 ? 1 : size;
    blocks_.clear();
    step_ = 0;
  }

  void setExchange(std::shared_ptr<VofBlockExchangeBase> x) { exch_ = std::move(x); }
  VofBlockExchangeBase* exchange() const { return exch_.get(); }

  int margin() const { return margin_; }
  int ghost() const { return ghost_; }
  /// Cells of slack added beyond the margin when a block is re-centred, so a translating bubble
  /// re-allocates every `pad + 1` steps instead of every step. Pure bookkeeping: the transported
  /// colour is copied by GLOBAL index and is therefore bit-exact whatever the padding.
  int recentrePad = 2;
  double cflLimit = 0.25;
  /// Rung W1 item (a): how masters are chosen, and how often the choice is revisited.
  /// `reassignEvery = 0` never re-assigns (W0's behaviour). A re-assignment MIGRATES the block's
  /// colour to the new master — nothing else in a block is state — so it is exact by construction
  /// and the bitwise gate holds ACROSS re-assignment events.
  VofMasterAssign assignMode = VofMasterAssign::RoundRobin;
  long reassignEvery = 0;
  /// Rung W1 item (c): recycle the advectors a re-centring retires, keyed by the EXACT extent.
  /// A translating bubble keeps its box SIZE and only moves its origin, so the hit rate is ~100 %
  /// and the ten Views of the new box are not allocated at all. A recycled advector has its colour
  /// and its three face-velocity fields zeroed on acquisition, which is exactly the state a freshly
  /// `init`ed one is in (Kokkos value-initializes), so `usePool` is BITWISE inert — the ctest gates
  /// that rather than asserting it.
  bool usePool = true;
  int poolCapacity = 8;  ///< advectors kept per extent; beyond it the retired one is freed
  /// Rung W2: form the CSF face force on each block (its own curvature cascade, the V4 balanced
  /// -force rule on the block's faces) and SUM it into the caller's global face-force fields.
  bool csfEnabled = false;
  double sigma = 0.0;  ///< surface-tension coefficient of the block CSF (cell units)
  /// The curvature cascade's TUNABLES for every block, copied into each block's own
  /// `VofCurvature` when it is allocated.  This is not decoration: `interfaceEps` (the wisp guard
  /// on the interfacial predicate) is what makes a block's curvature a function of the INTERFACE
  /// rather than of the Weymouth-Yue round-off residue, and without it a 1e-16 colour difference
  /// between two decompositions flips a cascade branch and moves the CSF face force by O(1e-3)
  /// -- measured, on the rung-W2 MPI gate, before this was propagated.  The solver sets it from
  /// its own `set_vof_interface_eps` / `set_surface_tension`, so the block and structured paths
  /// run the SAME estimator.
  VofCurvature curvProto;
  /// The colour threshold that defines the BUBBLE EXTENT (and hence the box). Weymouth-Yue leaves
  /// round-off residue in every cell its sweeps touch — measured down to 1e-35 and of either sign
  /// (the same residue that made the V4 curvature cascade need `interfaceEps = 1e-8`) — so a
  /// literal `C != 0` extent grows along the bubble's whole WAKE and the block degenerates into
  /// the global field. `1e-12` is 12 orders below any physical colour and ~5 orders above the
  /// residue; what it drops is accumulated into `VofBlockStats::discarded`.
  double bubbleEps = 1e-12;
  /// Wisp guard on the INTERFACE-AREA predicate (`VofBlockStats::area`), the same threshold and
  /// the same reason as the curvature's `interfaceEps`. A Weymouth-Yue round-off wisp satisfies
  /// `0 < C < 1`, its MYC normal is degenerate (the stencil is all zeros, so the kernel returns
  /// (1,0,0)) and `plicAlpha(1,0,0,1e-15)` puts a plane just inside the face: the polygon is the
  /// FULL unit square and the cell contributes an area of 1. Measured: three such cells made the
  /// reported area of a marker differ by 3.0 cells^2 between np = 1 and np = 4, off a colour that
  /// agreed to 1e-14.
  double areaEps = 1e-8;
  /// Let a block SHRINK when it has grown much larger than the bubble needs. Off gives a block
  /// that only ever grows — the G3 reference (a block large enough that it never has to move).
  bool allowShrink = true;
  double cellSize() const { return h_; }

  std::size_t count() const { return blocks_.size(); }
  std::vector<VofBlock>& blocks() { return blocks_; }
  const std::vector<VofBlock>& blocks() const { return blocks_; }
  long step() const { return step_; }
  void setStep(long s) { step_ = s; }

  /// Master assignment: round robin by block id (VOF_PLAN §10 W0; the weighted-ORB assignment is
  /// W1). Deliberately independent of where the bubble's cells live, which is the property gate G4
  /// exercises — with more blocks than ranks at least one block's master owns none of its cells.
  int masterOf(long id) const { return static_cast<int>(id % size_); }

  /// Seed a spherical bubble (centre and radius in CELL units, global). The block box is the
  /// sphere's cell extent grown by `margin`; the master fills the colour with the same exact
  /// `sphereCellFraction` the structured scenes use, so a seeded block and a seeded global field
  /// agree bit for bit.
  /// @param extraPad  cells added to the box beyond the margin (0 for production; a large value
  ///                   makes a block that never has to re-centre, which is the G3 reference).
  void seedSphere(double cx, double cy, double cz, double r, int subLevels = 4, int extraPad = 0) {
    VofBox bb;
    const double c[3] = {cx, cy, cz};
    for (int d = 0; d < 3; ++d) {
      bb.lo[d] = static_cast<int>(std::floor((c[d] - r) / h_)) - 1;
      bb.hi[d] = static_cast<int>(std::ceil((c[d] + r) / h_)) + 1;
    }
    VofBlock b;
    b.id = static_cast<long>(blocks_.size());
    b.master = masterOf(b.id);
    b.box = vofClampBox(VofBox::grown(bb, margin_ + extraPad), gs_, per_);
    b.mine_ = (b.master == rank_);
    blocks_.push_back(std::move(b));
    const std::size_t idx = blocks_.size() - 1;
    // Re-install every earlier block's hook: `push_back` may have reallocated the table, and the
    // hooks capture `this` + an index (so they survive it), but the advectors themselves moved.
    for (std::size_t k = 0; k + 1 < blocks_.size(); ++k)
      if (blocks_[k].mine_ && blocks_[k].allocated_)
        installHook(k);
    VofBlock& nb = blocks_[idx];
    if (nb.mine_) {
      allocate(idx);
      const I3 e = nb.adv_.extent();
      const int g = ghost_;
      const I3 o = nb.box.origin();
      const double hh = h_;
      SField cf = nb.adv_.colour();
      Kokkos::parallel_for(
          "vof::block::seed_sphere",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const double gx = (x - g + o.x) * hh, gy = (y - g + o.y) * hh, gz = (z - g + o.z) * hh;
            cf(L3(x, y, z, e)) = sphereCellFraction(cx, cy, cz, r, gx, gy, gz, hh, subLevels);
          });
      Kokkos::fence();
      fillBlockGhosts(blocks_[idx]);
      measure(blocks_[idx], 0.0);
    }
  }

  /// One kinematic block step: gather the face velocity, advect every local block, re-centre,
  /// replicate the table, and union the colour back into `cLocal`.
  ///
  /// @param cLocal  the caller's colour patch (the same extended block the local face velocity
  ///                lives on); the INNER region is overwritten with the union.
  void advect(double dt, SField cLocal) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    exch_->gatherFaceVel(blocks_, ghost_);
    for (auto& b : blocks_) {
      if (!b.mine_)
        continue;
      clampFaceVelocity(b);
      b.adv_.advect(dt, step_);
      b.st_.recentred = false;
    }
    for (std::size_t k = 0; k < blocks_.size(); ++k)
      if (blocks_[k].mine_)
        recentre(k);
    exch_->syncTable(blocks_);
    for (auto& b : blocks_)
      b.mine_ = (b.master == rank_);
    // Rung W1: re-balance the master assignment on the CURRENT boxes, before the scatter, so the
    // step's union is produced by the new owners and there is no half-migrated state anywhere.
    // The boxes have just been replicated, so `plannedMasters()` is the same on every rank.
    lastReassigned_ = 0;
    if (reassignEvery > 0 && ((step_ + 1) % reassignEvery) == 0)
      lastReassigned_ = assignMasters();
    exch_->scatterColourMax(blocks_, cLocal);
    for (auto& b : blocks_)
      if (b.mine_)
        measure(b, dt);
    ++step_;
  }

  /// Blocks that changed master at the last `advect()` (rung W1 item a).
  long lastReassigned() const { return lastReassigned_; }

  // ---- rung W2: per-block curvature + the CSF face force ---------------------------------------

  /// Turn on the block CSF. Each master block then carries its OWN `VofCurvature` on its own
  /// extended box and forms the V4 balanced-force face force there; `computeCsf` scatters the
  /// three face fields into the caller's patches with UNPACK_SUM.
  void enableCsf(double sigmaValue) {
    csfEnabled = true;
    sigma = sigmaValue;
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (blocks_[i].mine_ && blocks_[i].allocated_)
        allocateCsf(i);
  }

  /// Per-block curvature cascade + the CSF face force, summed into the caller's three face-force
  /// patches. `fx/fy/fz` are ZEROED first (the force is a per-step quantity, never accumulated).
  void computeCsf(SField fx, SField fy, SField fz) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    curvStats_ = VofCurvature::Stats{};
    for (auto& b : blocks_) {
      if (!b.mine_)
        continue;
      const VofCurvature::Stats st = b.curv_.compute(b.adv_.colour());
      curvStats_.interfacial += st.interfacial;
      curvStats_.hf += st.hf;
      curvStats_.hfMixed += st.hfMixed;
      curvStats_.hfFit += st.hfFit;
      curvStats_.pv += st.pv;
      curvStats_.pvReduced += st.pvReduced;
      curvStats_.noEstimate += st.noEstimate;
      buildCsfForce(b);
    }
    exch_->scatterForceSum(blocks_, fx, fy, fz);
  }
  VofCurvature::Stats csfCurvatureStats() const { return curvStats_; }

  void allocateCsf(std::size_t idx) {
    VofBlock& b = blocks_[idx];
    const I3 n = b.adv_.inner();
    b.curv_.init(n.x, n.y, n.z, ghost_);
    b.curv_.weightWidth = curvProto.weightWidth;
    b.curv_.monoTol = curvProto.monoTol;
    b.curv_.ptWeightWidth = curvProto.ptWeightWidth;
    b.curv_.cosMin = curvProto.cosMin;
    b.curv_.interfaceEps = curvProto.interfaceEps;
    b.curv_.useMixedHeightFit = curvProto.useMixedHeightFit;
    const long len = static_cast<long>(b.adv_.extent().x) * b.adv_.extent().y * b.adv_.extent().z;
    for (int c = 0; c < 3; ++c)
      b.f_[c] = SField("vof::block::csf", len);
  }

  /// The V4 balanced-force CSF on the BLOCK's faces — the same `csfFaceCurvature` +
  /// `csfFaceForce` pair `Solver::addCsfRhs` applies to the global field, on the block's own
  /// colour and curvature. Formed over the inner box; the low face of an inner cell at local index
  /// 0 reads the block's ghost, which the margin guarantees is pure gas, so the force is exactly
  /// zero there and the block's force has compact support inside its own box.
  void buildCsfForce(VofBlock& b) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const double sig = sigma, hh = h_;
    SField cv = b.adv_.colour(), kp = b.curv_.kappa(), kb = b.curv_.branch();
    const long sy = e.x, sz = static_cast<long>(e.x) * e.y;
    for (int c = 0; c < 3; ++c) {
      SField ff = b.f_[c];
      const long strd = (c == 0) ? 1 : (c == 1 ? sy : sz);
      Kokkos::parallel_for(
          "vof::block::csf_force",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                        {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            const double dC = cv(i) - cv(i - strd);
            double f = 0.0;
            if (dC != 0.0) {
              double kf = 0.0;
              vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
              f = vof::csfFaceForce(sig, kf, dC, hh);
            }
            ff(i) = f;
          });
    }
    Kokkos::fence();
  }

  /// Union the current colour into `cLocal` without advecting (used right after seeding).
  void scatter(SField cLocal) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    exch_->syncTable(blocks_);
    exch_->scatterColourMax(blocks_, cLocal);
  }

  // ---- rung W1 item (a): master assignment and re-assignment ---------------------------------

  /// The master each block WOULD get under `assignMode`, from the current block cell counts. A
  /// pure function of the replicated table, so every rank computes the same vector without an
  /// exchange — which is what lets a re-assignment happen mid-run without breaking a bitwise gate.
  std::vector<int> plannedMasters() const {
    std::vector<long> w(blocks_.size());
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      w[i] = blocks_[i].box.cells();
    std::vector<int> m;
    switch (assignMode) {
      case VofMasterAssign::Lpt:
        vofAssignLpt(w, size_, m);
        break;
      case VofMasterAssign::WeightedOrb:
        vofAssignOrb(w, size_, m);
        break;
      case VofMasterAssign::RoundRobin:
      default:
        m.assign(blocks_.size(), 0);
        for (std::size_t i = 0; i < blocks_.size(); ++i)
          m[i] = static_cast<int>(blocks_[i].id % size_);
        break;
    }
    return m;
  }

  /// Apply `plannedMasters()`: allocate on the gaining ranks, migrate the colour, free on the
  /// losing ranks. Returns the number of blocks that changed master. A no-op — and no message —
  /// when the assignment is unchanged, which is the common case once the swarm has settled.
  long assignMasters() {
    if (blocks_.empty() || size_ <= 1)
      return 0;
    const std::vector<int> want = plannedMasters();
    std::vector<int> old(blocks_.size());
    long moved = 0;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      old[i] = blocks_[i].master;
      if (old[i] != want[i])
        ++moved;
    }
    if (moved == 0)
      return 0;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      blocks_[i].master = want[i];
      blocks_[i].st_.master = want[i];
    }
    // gaining ranks allocate FIRST (zeroed), so the migration has somewhere to land
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (want[i] == rank_ && old[i] != rank_) {
        blocks_[i].mine_ = true;
        allocate(i);
      }
    if (exch_)
      exch_->migrateColour(blocks_, old);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      if (old[i] == rank_ && want[i] != rank_) {
        release(i);
        blocks_[i].mine_ = false;
      } else if (want[i] == rank_ && old[i] != rank_) {
        // A GAINED block only. A block this rank keeps is left completely alone -- `measure` would
        // reset its previous centroid and blank one step of its reported velocity.
        blocks_[i].mine_ = true;
        fillBlockGhosts(blocks_[i]);
        double aux[4];
        blocks_[i].serializeAux(aux);  // the migrated previous centroid survives the re-measure
        measure(blocks_[i], 0.0);
        blocks_[i].deserializeAux(aux);
      }
    }
    return moved;
  }

  // ---- general seeding: a block from an arbitrary global colour field -------------------------

  /// Seed one block on the global index box `bb` (the BUBBLE extent; the inner box is `bb` grown by
  /// `margin`), with the colour gathered from the caller's colour patch. This is the general
  /// seeding path — a sphere seed is only a convenience over it — and it is how a marker of any
  /// shape (a quasi-2-D cylinder, a Hysing bubble, a scanned geometry) enters the container.
  /// Call `finishSeeding(cLocal)` once after the last `seedBox` to perform the gather.
  void seedBox(const VofBox& bb, int extraPad = 0) {
    VofBlock b;
    b.id = static_cast<long>(blocks_.size());
    b.master = static_cast<int>(b.id % size_);
    b.box = vofClampBox(VofBox::grown(bb, margin_ + extraPad), gs_, per_);
    b.seed_ = bb;
    b.mine_ = (b.master == rank_);
    blocks_.push_back(std::move(b));
    const std::size_t idx = blocks_.size() - 1;
    for (std::size_t k = 0; k + 1 < blocks_.size(); ++k)
      if (blocks_[k].mine_ && blocks_[k].allocated_)
        installHook(k);
    if (blocks_[idx].mine_)
      allocate(idx);
  }

  /// Gather the colour of every `seedBox`-seeded block from `cLocal`, fill its ghosts and measure.
  void finishSeeding(SField cLocal) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    exch_->gatherColour(blocks_, cLocal);
    for (auto& b : blocks_)
      if (b.mine_) {
        clipToSeed(b);
        fillBlockGhosts(b);
        measure(b, 0.0);
      }
  }

  /// A seeded block's colour is its OWN bubble's: the gather copies the whole INNER box (bubble
  /// extent + margin) out of the global field, and the margin ring belongs to no marker by the
  /// margin contract (it is what makes the block conservative), so anything the gather picked up
  /// there is a NEIGHBOURING marker's liquid and must not be adopted. Without this clip two
  /// bubbles closer than `2 * margin` would each seed with a slice of the other -- measured on the
  /// rung-W2 MPI gate as marker volumes 491 and 528 against a seed volume of 382.
  void clipToSeed(VofBlock& b) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner(), o = b.box.origin();
    const int g = ghost_;
    const int lx = b.seed_.lo[0], hx = b.seed_.hi[0], ly = b.seed_.lo[1], hy = b.seed_.hi[1],
              lz = b.seed_.lo[2], hz = b.seed_.hi[2];
    if (hx <= lx || hy <= ly || hz <= lz)
      return;  // no seed box recorded (a sphere seed): nothing to clip
    SField c = b.adv_.colour();
    Kokkos::parallel_for(
        "vof::block::clip_seed",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n.x, n.y, n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const int gx = x + o.x, gy = y + o.y, gz = z + o.z;
          if (gx < lx || gx >= hx || gy < ly || gy >= hy || gz < lz || gz >= hz)
            c(L3(x + g, y + g, z + g, e)) = 0.0;
        });
    Kokkos::fence();
  }

  /// The block's own colour ghost policy — see the file header. Installed as the advector's
  /// `exchange` hook, so it also runs between the three sweeps.
  void fillBlockGhosts(VofBlock& b) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner(), o = b.box.origin();
    const int g = ghost_;
    SField f = b.adv_.colour();
    // 1. every ghost cell -> 0 (the block's far field: pure gas, guaranteed by the margin).
    Kokkos::parallel_for(
        "vof::block::ghost_zero",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          if (x >= g && x < g + n.x && y >= g && y < g + n.y && z >= g && z < g + n.z)
            return;
          f(L3(x, y, z, e)) = 0.0;
        });
    Kokkos::fence();
    // 2. an axis the block spans entirely, and which is globally periodic, wraps within the block.
    const bool sx = per_[0] && b.box.n(0) == gs_.x;
    const bool sy = per_[1] && b.box.n(1) == gs_.y;
    const bool sz = per_[2] && b.box.n(2) == gs_.z;
    if (sx || sy || sz)
      vof::periodicFill(f, e, g, sx, sy, sz);
    // 3. the part of the extended box outside a NON-periodic domain takes the globally clamped
    //    (zero-gradient) value — the same rule the structured colour field uses.
    if (!per_[0] || !per_[1] || !per_[2])
      vof::clampFill(f, e, g, o, gs_, per_[0], per_[1], per_[2]);
  }

  /// Per-bubble census on the master (empty entries on a non-master).
  std::vector<VofBlockStats> statsAll() const {
    // `id`, `master` and the box come from the REPLICATED table, so they are reported on every
    // rank; only the measured entries (volume, centroid, velocity, moments, area) are the
    // master's.  Filling them here rather than in `measure()` is what makes that promise true on
    // a rank that masters nothing.
    std::vector<VofBlockStats> v;
    v.reserve(blocks_.size());
    for (const auto& b : blocks_) {
      VofBlockStats q = b.st_;
      q.id = b.id;
      q.master = b.master;
      for (int d = 0; d < 3; ++d) {
        q.lo[d] = b.box.lo[d];
        q.hi[d] = b.box.hi[d];
      }
      q.cells = b.box.cells();
      v.push_back(q);
    }
    return v;
  }

  /// Load-balance census of the CURRENT master assignment: `masters[r]` = blocks mastered by rank
  /// r, `cells[r]` = the inner cells those blocks carry (the actual VoF work). W0 assigns masters
  /// round-robin; W1 replaces this with the weighted ORB and these are the numbers to beat.
  void masterCensus(std::vector<long>& masters, std::vector<long>& cells) const {
    masters.assign(size_, 0);
    cells.assign(size_, 0);
    for (const auto& b : blocks_) {
      masters[b.master] += 1;
      cells[b.master] += VofBox::grown(b.box, 0).cells();
    }
  }
  /// max/mean of the per-rank block-cell load (1.0 = perfect). 0 blocks -> 1.0.
  double cellImbalance() const {
    std::vector<long> m, c;
    masterCensus(m, c);
    long tot = 0, mx = 0;
    for (long v : c) {
      tot += v;
      mx = v > mx ? v : mx;
    }
    if (tot == 0)
      return 1.0;
    return static_cast<double>(mx) * size_ / static_cast<double>(tot);
  }

  I3 globalSize() const { return gs_; }
  std::array<bool, 3> periodic() const { return per_; }
  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  /// Install the advector's ghost hook. It captures the block's INDEX, never a pointer: the block
  /// table is a `std::vector` that reallocates on every `seedSphere`, and `WyAdvector` is moved
  /// wholesale on a re-centre, so a captured `VofBlock*` is a dangling read waiting to happen.
  void installHook(std::size_t idx) {
    VofBlockSet* self = this;
    blocks_[idx].adv_.exchange = [self, idx](SField) { self->fillBlockGhosts(self->blocks_[idx]); };
  }

  void allocate(std::size_t idx) {
    VofBlock& b = blocks_[idx];
    b.adv_ = acquireAdvector(b.box.n(0), b.box.n(1), b.box.n(2));
    installHook(idx);
    b.allocated_ = true;
    b.st_.id = b.id;
    b.st_.master = b.master;
    if (csfEnabled)
      allocateCsf(idx);
  }

  /// Give a block's advector back (a master that lost the block on a re-assignment). The block's
  /// table row survives — only the state is released.
  void release(std::size_t idx) {
    VofBlock& b = blocks_[idx];
    retireAdvector(std::move(b.adv_));
    b.adv_ = WyAdvector();
    b.curv_ = VofCurvature();
    for (int c = 0; c < 3; ++c)
      b.f_[c] = SField();
    b.allocated_ = false;
    b.hasPrev_ = false;
  }

  /// Rung W1 item (c): the block pool. Keyed by the EXACT extent, so a translating bubble (whose
  /// box keeps its size and only moves its origin) never allocates after the first re-centring.
  /// A recycled advector is handed back in the state a freshly `init`ed one is in — colour and the
  /// three face-velocity fields zeroed — so the pool is bitwise inert. Everything else a
  /// `WyAdvector` holds (the frozen dilation flag, the per-sweep PLIC planes, the face Courant
  /// numbers and the fluxes) is written before it is read inside `advect()`.
  WyAdvector acquireAdvector(int nx, int ny, int nz) {
    const std::array<int, 3> key{nx, ny, nz};
    if (usePool) {
      auto it = pool_.find(key);
      if (it != pool_.end() && !it->second.empty()) {
        WyAdvector a = std::move(it->second.back());
        it->second.pop_back();
        ++poolHits_;
        Kokkos::deep_copy(a.colour(), 0.0);
        for (int d = 0; d < 3; ++d)
          Kokkos::deep_copy(a.faceVel(d), 0.0);
        Kokkos::fence();
        a.cflLimit = cflLimit;
        a.globalMax = nullptr;
        a.exchange = nullptr;  // re-installed by installHook
        return a;
      }
    }
    ++poolMisses_;
    WyAdvector a;
    a.init(nx, ny, nz, h_, ghost_);
    a.cflLimit = cflLimit;
    a.globalMax = nullptr;  // the block IS the whole domain of its own advector
    return a;
  }

  void retireAdvector(WyAdvector&& a) {
    if (!usePool || a.colour().extent(0) == 0)
      return;
    const I3 n = a.inner();
    const std::array<int, 3> key{n.x, n.y, n.z};
    auto& v = pool_[key];
    if (static_cast<int>(v.size()) >= poolCapacity)
      return;
    a.exchange = nullptr;  // the hook captures a block index; never carry it into the pool
    v.push_back(std::move(a));
  }

 public:
  long poolHits() const { return poolHits_; }
  long poolMisses() const { return poolMisses_; }
  void clearPool() { pool_.clear(); }

 private:

  /// Face velocity outside a non-periodic domain: the gather leaves it untouched, so continue it
  /// with the same globally clamped rule the colour uses. Inside the domain every extended-box cell
  /// has an owner and was filled.
  void clampFaceVelocity(VofBlock& b) {
    if (per_[0] && per_[1] && per_[2])
      return;
    const I3 e = b.adv_.extent(), o = b.box.origin();
    for (int d = 0; d < 3; ++d)
      vof::clampFill(b.adv_.faceVel(d), e, ghost_, o, gs_, per_[0], per_[1], per_[2]);
  }

  // NOTE: everything below is public ONLY because nvcc refuses an extended `__host__ __device__`
  // lambda inside a private or protected member function ("The enclosing parent function ... cannot
  // have private or protected access within its class"). These are implementation details of the
  // step; treat them as such.
 public:
  /// Tight bounding box of `C > 0` over the inner region, in GLOBAL indices. Computed in the
  /// block's LOCAL frame first, so a bubble straddling a periodic seam stays contiguous.
  bool bubbleBox(const VofBlock& b, VofBox& out) const {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const double eps = bubbleEps;
    SField c = b.adv_.colour();
    int lo[3] = {n.x, n.y, n.z}, hi[3] = {-1, -1, -1};
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    Kokkos::parallel_reduce(
        "vof::block::bbox_lo", pol,
        KOKKOS_LAMBDA(int x, int y, int z, int& lx, int& ly, int& lz) {
          if (!(Kokkos::fabs(c(L3(x, y, z, e))) > eps))
            return;
          const int px = x - g, py = y - g, pz = z - g;
          lx = px < lx ? px : lx;
          ly = py < ly ? py : ly;
          lz = pz < lz ? pz : lz;
        },
        Kokkos::Min<int>(lo[0]), Kokkos::Min<int>(lo[1]), Kokkos::Min<int>(lo[2]));
    Kokkos::parallel_reduce(
        "vof::block::bbox_hi", pol,
        KOKKOS_LAMBDA(int x, int y, int z, int& hx, int& hy, int& hz) {
          if (!(Kokkos::fabs(c(L3(x, y, z, e))) > eps))
            return;
          const int px = x - g, py = y - g, pz = z - g;
          hx = px > hx ? px : hx;
          hy = py > hy ? py : hy;
          hz = pz > hz ? pz : hz;
        },
        Kokkos::Max<int>(hi[0]), Kokkos::Max<int>(hi[1]), Kokkos::Max<int>(hi[2]));
    Kokkos::fence();
    if (hi[0] < lo[0])
      return false;  // the block is empty
    for (int d = 0; d < 3; ++d) {
      out.lo[d] = lo[d] + b.box.lo[d];
      out.hi[d] = hi[d] + 1 + b.box.lo[d];
    }
    return true;
  }

  /// Move / resize the block when the bubble has spent its margin (or when the box has grown
  /// wastefully large). The colour is copied by GLOBAL index — exact, no interpolation.
  void recentre(std::size_t idx) {
    VofBlock& b = blocks_[idx];
    VofBox bb;
    if (!bubbleBox(b, bb))
      return;
    const VofBox need = vofClampBox(VofBox::grown(bb, margin_), gs_, per_);
    bool wasteful = false;
    if (allowShrink)
      for (int d = 0; d < 3; ++d)
        if (b.box.n(d) > need.n(d) + 2 * (margin_ + recentrePad))
          wasteful = true;
    if (b.box.contains(need) && !wasteful)
      return;
    const VofBox nb = vofClampBox(VofBox::grown(bb, margin_ + recentrePad), gs_, per_);
    if (nb == b.box)
      return;
    // What the move DROPS: the colour of old-box cells that fall outside the new box. Measured
    // directly rather than as `sum(old) - sum(new)`, which at |sum| ~ 1e2 is 1e-13 of summation
    // rounding and says nothing about the wisps (measured -5.7e-14 that way on the G1 scene).
    const double dropped = outOfBoxSum(b, nb);
    // exact copy by global index (both boxes are in the same unwrapped global frame)
    WyAdvector fresh = acquireAdvector(nb.n(0), nb.n(1), nb.n(2));
    const I3 se = b.adv_.extent(), de = fresh.extent();
    const I3 dn = fresh.inner();
    const int g = ghost_;
    const int sx = b.box.lo[0], sy = b.box.lo[1], sz = b.box.lo[2];
    const int snx = b.box.n(0), sny = b.box.n(1), snz = b.box.n(2);
    const int dx = nb.lo[0], dy = nb.lo[1], dz = nb.lo[2];
    SField src = b.adv_.colour(), dst = fresh.colour();
    Kokkos::parallel_for(
        "vof::block::recentre_copy",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {dn.x, dn.y, dn.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const int gx = x + dx, gy = y + dy, gz = z + dz;
          const int lx = gx - sx, ly = gy - sy, lz = gz - sz;
          double v = 0.0;
          if (lx >= 0 && lx < snx && ly >= 0 && ly < sny && lz >= 0 && lz < snz)
            v = src(L3(lx + g, ly + g, lz + g, se));
          dst(L3(x + g, y + g, z + g, de)) = v;
        });
    Kokkos::fence();
    b.box = nb;
    retireAdvector(std::move(b.adv_));
    b.adv_ = std::move(fresh);
    b.adv_.cflLimit = cflLimit;
    b.adv_.globalMax = nullptr;
    installHook(idx);
    if (csfEnabled)
      allocateCsf(idx);
    fillBlockGhosts(blocks_[idx]);
    blocks_[idx].st_.recentred = true;
    blocks_[idx].st_.discarded += dropped;
  }

  /// Sum of the block's colour over inner cells whose GLOBAL index falls outside `nb`.
  double outOfBoxSum(const VofBlock& b, const VofBox& nb) const {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const int sx = b.box.lo[0], sy = b.box.lo[1], sz = b.box.lo[2];
    const int lx = nb.lo[0], ly = nb.lo[1], lz = nb.lo[2];
    const int hx = nb.hi[0], hy = nb.hi[1], hz = nb.hi[2];
    SField c = b.adv_.colour();
    double v = 0.0;
    Kokkos::parallel_reduce(
        "vof::block::out_of_box",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& a) {
          const int gx = x - g + sx, gy = y - g + sy, gz = z - g + sz;
          if (gx >= lx && gx < hx && gy >= ly && gy < hy && gz >= lz && gz < hz)
            return;
          a += c(L3(x, y, z, e));
        },
        v);
    Kokkos::fence();
    return v;
  }

  double innerSum(const VofBlock& b) const {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    SField c = b.adv_.colour();
    double v = 0.0;
    Kokkos::parallel_reduce(
        "vof::block::inner_sum",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& a) { a += c(L3(x, y, z, e)); }, v);
    Kokkos::fence();
    return v;
  }

  /// Rung W1 item (d): the marker's interface area, in CELL units squared (the same convention
  /// `volume` uses: cell volumes, NOT the physical length units `centroid` carries) — the sum
  /// over the inner box
  /// of the PLIC polygon area of every mixed cell, reconstructed from the MYC normal exactly as the
  /// curvature cascade does (`mycNormal` -> `plicAlpha` -> `plicPolygon` -> `polygonAreaCentroid`).
  /// A reconstructed area, not a fit: a sphere reads 4 pi R^2 to the PLIC discretization error.
  double interfaceArea(const VofBlock& b) const {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const double hh = h_, aeps = areaEps;
    SField c = b.adv_.colour();
    const long sy = e.x, sz = static_cast<long>(e.x) * e.y;
    double a = 0.0;
    Kokkos::parallel_reduce(
        "vof::block::area",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = L3(x, y, z, e);
          const double ci = c(i);
          if (!(ci > aeps) || !(ci < 1.0 - aeps))
            return;
          double st[27];
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx)
                st[vof::plicSt(dx + 1, dy + 1, dz + 1)] = c(i + dx + dy * sy + dz * sz);
          double m[3];
          vof::mycNormal(st, m);
          const double al = vof::plicAlpha(m[0], m[1], m[2], ci);
          double v[8][3], ctr[3], ar = 0.0;
          const int nv = vof::plicPolygon(m[0], m[1], m[2], al, v);
          vof::polygonAreaCentroid(v, nv, ctr, ar);
          acc += ar;
        },
        a);
    Kokkos::fence();
    (void)hh;
    return a;  // CELL units squared, matching `volume`'s cell volumes
  }

  /// Volume, centroid, centroid velocity and the central second moments over the inner box.
  void measure(VofBlock& b, double dt) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner(), o = b.box.origin();
    const int g = ghost_;
    const double hh = h_;
    SField c = b.adv_.colour();
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    double v = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
    Kokkos::parallel_reduce(
        "vof::block::moments1", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& av, double& ax, double& ay, double& az) {
          const double q = c(L3(x, y, z, e));
          av += q;
          ax += q * (x - g + o.x + 0.5) * hh;
          ay += q * (y - g + o.y + 0.5) * hh;
          az += q * (z - g + o.z + 0.5) * hh;
        },
        v, mx, my, mz);
    Kokkos::fence();
    b.st_.id = b.id;
    b.st_.master = b.master;
    for (int d = 0; d < 3; ++d) {
      b.st_.lo[d] = b.box.lo[d];
      b.st_.hi[d] = b.box.hi[d];
    }
    b.st_.cells = b.box.cells();
    b.st_.volume = v;
    if (v <= 0.0) {
      b.st_.area = 0.0;
      for (int d = 0; d < 3; ++d)
        b.st_.centroid[d] = b.st_.velocity[d] = 0.0;
      for (int d = 0; d < 6; ++d)
        b.st_.moment[d] = 0.0;
      return;
    }
    const double cx = mx / v, cy = my / v, cz = mz / v;
    double sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
    Kokkos::parallel_reduce(
        "vof::block::moments2", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& axx, double& ayy, double& azz, double& axy,
                      double& axz, double& ayz) {
          const double q = c(L3(x, y, z, e));
          const double px = (x - g + o.x + 0.5) * hh - cx, py = (y - g + o.y + 0.5) * hh - cy,
                       pz = (z - g + o.z + 0.5) * hh - cz;
          axx += q * px * px;
          ayy += q * py * py;
          azz += q * pz * pz;
          axy += q * px * py;
          axz += q * px * pz;
          ayz += q * py * pz;
        },
        sxx, syy, szz, sxy, sxz, syz);
    Kokkos::fence();
    const double nc[3] = {cx, cy, cz};
    for (int d = 0; d < 3; ++d) {
      b.st_.velocity[d] = (b.hasPrev_ && dt > 0.0) ? (nc[d] - b.prevCentroid_[d]) / dt : 0.0;
      b.prevCentroid_[d] = nc[d];
      b.st_.centroid[d] = nc[d];
    }
    b.hasPrev_ = true;
    b.st_.area = interfaceArea(b);
    const double iv = 1.0 / v;
    b.st_.moment[0] = sxx * iv;
    b.st_.moment[1] = syy * iv;
    b.st_.moment[2] = szz * iv;
    b.st_.moment[3] = sxy * iv;
    b.st_.moment[4] = sxz * iv;
    b.st_.moment[5] = syz * iv;
  }

 private:
  double h_ = 1.0;
  I3 gs_{0, 0, 0};
  std::array<bool, 3> per_{true, true, true};
  int rank_ = 0, size_ = 1;
  int margin_ =
      3;           ///< TBFsolver's 3-cell offset; see the header (it is what makes it conservative)
  int ghost_ = 3;  ///< the colour field's own halo (VOF_PLAN §3 rule 1)
  long step_ = 0;
  std::vector<VofBlock> blocks_;
  std::shared_ptr<VofBlockExchangeBase> exch_;
  std::map<std::array<int, 3>, std::vector<WyAdvector>> pool_;
  long poolHits_ = 0, poolMisses_ = 0;
  long lastReassigned_ = 0;
  VofCurvature::Stats curvStats_{};
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_CONTAINER_HPP
