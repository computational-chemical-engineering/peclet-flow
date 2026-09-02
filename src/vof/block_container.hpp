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
#include <memory>
#include <stdexcept>
#include <vector>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "vof/advect_wy.hpp"
#include "vof/colour_field.hpp"

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
};

/// One bubble's block. Every rank holds the (id, box, master) triple — the replicated table — and
/// ONLY the master allocates the advector and the colour.
class VofBlock {
 public:
  long id = 0;
  int master = 0;
  VofBox box;  ///< the INNER box, global

  bool mine() const { return mine_; }
  WyAdvector& advector() { return adv_; }
  const WyAdvector& advector() const { return adv_; }
  VofBox extended(int ghost) const { return VofBox::grown(box, ghost); }
  const VofBlockStats& stats() const { return st_; }

  friend class VofBlockSet;

 private:
  bool mine_ = false;
  bool allocated_ = false;
  WyAdvector adv_;
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
  /// The colour threshold that defines the BUBBLE EXTENT (and hence the box). Weymouth-Yue leaves
  /// round-off residue in every cell its sweeps touch — measured down to 1e-35 and of either sign
  /// (the same residue that made the V4 curvature cascade need `interfaceEps = 1e-8`) — so a
  /// literal `C != 0` extent grows along the bubble's whole WAKE and the block degenerates into
  /// the global field. `1e-12` is 12 orders below any physical colour and ~5 orders above the
  /// residue; what it drops is accumulated into `VofBlockStats::discarded`.
  double bubbleEps = 1e-12;
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
    exch_->scatterColourMax(blocks_, cLocal);
    for (auto& b : blocks_)
      if (b.mine_)
        measure(b, dt);
    ++step_;
  }

  /// Union the current colour into `cLocal` without advecting (used right after seeding).
  void scatter(SField cLocal) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    exch_->syncTable(blocks_);
    exch_->scatterColourMax(blocks_, cLocal);
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
    std::vector<VofBlockStats> v;
    v.reserve(blocks_.size());
    for (const auto& b : blocks_)
      v.push_back(b.st_);
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
    b.adv_.init(b.box.n(0), b.box.n(1), b.box.n(2), h_, ghost_);
    b.adv_.cflLimit = cflLimit;
    b.adv_.globalMax = nullptr;  // the block IS the whole domain of its own advector
    installHook(idx);
    b.allocated_ = true;
    b.st_.id = b.id;
    b.st_.master = b.master;
  }

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
    WyAdvector fresh;
    fresh.init(nb.n(0), nb.n(1), nb.n(2), h_, ghost_);
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
    b.adv_ = std::move(fresh);
    b.adv_.cflLimit = cflLimit;
    b.adv_.globalMax = nullptr;
    installHook(idx);
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
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_CONTAINER_HPP
