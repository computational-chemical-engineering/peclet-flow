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

// ---------------------------------------------------------------------------------------------
// rung W4: pairs, events, and the two explicit models
// ---------------------------------------------------------------------------------------------

/// Rung W4 item 2: the state of one marker PAIR whose inner boxes overlap.
///
/// `film` is the design's film thickness: the gap between the two markers' interfaces **along the
/// line of centres**, measured as `|c_B - c_A| - d_A - d_B` where `d_X` is the distance from
/// marker X's centroid to the last point along that line at which its OWN colour is still >= 1/2
/// (a ray march with trilinear interpolation on the marker's own block — the PLIC plane's own
/// crossing to within the sampling step, which is `h/8`). It is NEGATIVE when the two markers
/// interpenetrate, which is the state the block container is designed to carry and the state that
/// broke rung W2's force rule (WO-W3 finding 7 measured `dmin = 8.0` cells at `D = 10`).
///
/// Everything here is in CELL units and is REPLICATED on every rank: the masters compute their own
/// halves and one `MPI_Allreduce` over disjoint contributions makes the census identical
/// everywhere, whatever the decomposition (gate G6).
struct VofPairStats {
  long idA = 0, idB = 0;
  double dist = 0.0;      ///< centroid separation, minimum image
  double dA = 0.0, dB = 0.0;  ///< the two markers' interface radii along the line of centres
  double film = 0.0;      ///< `dist - dA - dB`; < 0 = interpenetration
  double approach = 0.0;  ///< normal approach velocity, `-(v_B - v_A) . n`; > 0 = closing
  double weber = 0.0;     ///< `rho_l U_n^2 D_eq / sigma`, the collision Weber number
  double dEq = 0.0;       ///< `2 D_A D_B / (D_A + D_B)`, the equivalent diameter
  double volA = 0.0, volB = 0.0;
  long contactSteps = 0;  ///< consecutive steps with `film <= contactFilm`
  double filmTime = 0.0;  ///< time accumulated with `film < hCrit` (the drainage clock)
  int state = 0;          ///< 1 = in the census, 2 = in contact
};

/// Which coalescence model decides what a pair in contact does. `Never` is the DEFAULT and it is
/// the physically right default for a bubbly flow: coalescence in a contaminated or high-Weber
/// system is the exception, and the whole point of the container is that the numerics does not
/// decide it (TBFsolver makes the same choice).
enum class VofCoalescence { Never = 0, Film = 1, Weber = 2 };

/// The model's parameters, all explicit, all in the solver's units (cells and seconds).
struct VofCoalescenceModel {
  VofCoalescence model = VofCoalescence::Never;
  /// "film": merge once the film has been thinner than `hCrit` for a drainage time
  /// `t_d = drainC * mu_l * D_eq / sigma` (the Prince & Blanch 1990 form; `drainC` is the
  /// coefficient their model leaves to the flow, so it is a parameter here and not a constant).
  double hCrit = 1.0;
  double drainC = 1.0;
  /// "weber": merge AT CONTACT when the collision Weber number is below this. Bubble-pair
  /// experiments put the bouncing/coalescence transition near We ~ 1 (Duineveld 1998), so 1.0 is
  /// the documented starting point and not a tuned number.
  double weCrit = 1.0;
  /// A pair is IN CONTACT when the film is at or below this many cells. One cell is the thinnest
  /// film this grid resolves, which is what makes it the natural threshold.
  double contactFilm = 1.0;
};

/// One census line. `type`: 0 approach (the pair entered the census), 1 contact (the film reached
/// `contactFilm`), 2 merge, 3 split. For a split, `idA` is the parent and `idB` the first child.
struct VofEvent {
  int type = 0;
  long step = 0;
  double time = 0.0;
  long idA = 0, idB = 0;
  double film = 0.0, approach = 0.0, weber = 0.0, dEq = 0.0;
  double volA = 0.0, volB = 0.0, volNew = 0.0;
  long children = 0;  ///< split only: how many blocks the parent became
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
  bool alive() const { return alive_; }
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
  /// Rung W4 item 1: the four per-face ACCUMULATORS this marker contributes to the owner, in the
  /// scatter's component order `i = 4 c + k`:
  ///
  ///   k = 0  `KS` = `kappa_f`                       — used as-is when this marker is ALONE on
  ///                 the face, so a single-marker face gets its own curvature bit for bit
  ///   k = 1  `W`  = `|dC_marker|`                   — the weight of this marker's curvature
  ///   k = 2  `KW` = `kappa_f |dC_marker|`
  ///   k = 3  `N`  = 1                               — how many markers claim the face
  ///
  /// A marker contributes at all only where its curvature is DEFINED (`csfFaceCurvature` true);
  /// an orphan face contributes nothing, so it can never drag the union curvature towards zero.
  ///
  /// What is NOT scattered is the FORCE, and that is the whole of rung W4. A marker's own
  /// `sigma kappa dC_marker / h` is the right force only where the union's colour jump IS that
  /// marker's; on the film between two interpenetrating markers one marker has a jump and the
  /// union has none, and applying that marker's force there is an unbalanced force that the
  /// projection cannot remove. Measured, on the W3 `channel_18` checkpoint: keeping the marker's
  /// own force wherever `N == 1` (which is bitwise-safest, and was tried first) fails at 1.524
  /// eddy turnovers — EARLIER than rung W2's own 1.605 — while the rule below, which never uses
  /// anything but the union's own difference, runs past it. See `Solver::assembleBlockCsf`.
  SField csfAcc(int c, int k) const { return (k == 0) ? f_[c] : a_[c][k - 1]; }
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
  /// Rung W4: a block RETIRED by a merge or a split keeps its table row (block id == table index
  /// is an invariant every lookup relies on) but has an empty box, no state, and is skipped by
  /// every loop and every transfer.
  bool alive_ = true;
  long splitSteps_ = 0;  ///< consecutive steps this block held more than one component
  /// Is this block ELIGIBLE to be split? A block a MERGE just created holds two components by
  /// construction — that is what it was made from — so breakup would undo the merge on its third
  /// step and the container would oscillate (measured: merge@1, split@3, merge@4, split@6, ... on
  /// the W2 MPI pair scene with `contactFilm = 4`). A merged block is therefore not armed until it
  /// has been a SINGLE component once, i.e. until its two markers have actually joined; from then
  /// on it splits by the ordinary rule.
  bool splitArmed_ = true;
  WyAdvector adv_;
  VofCurvature curv_;
  SField f_[3];
  SField a_[3][3];  ///< W4: {W, KW, N} per component (see `csfAcc`)
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
  /// SUM every master block's CSF accumulators into the caller's patches (TBFsolver's UNPACK_SUM,
  /// `VOF.f90::computeSurfaceTension`). `nc == 3` scatters the rung-W2 FORCE alone (`loc` is
  /// fx/fy/fz) — overlapping markers then ADD their forces, which is the pairing defect rung W4
  /// exists to repair; `nc == 12` scatters the four W4 accumulators of `VofBlock::csfAcc` in the
  /// order `i = 4 c + k`, and the owner forms ONE force per face from the union colour.
  virtual void scatterCsfSum(std::vector<VofBlock>& blocks, SField* loc, int nc) = 0;
  /// Sum `n` doubles across the ranks that share the block table, in place. Every entry is written
  /// by exactly ONE rank (the block's master) and read by all, so the sum is a broadcast that
  /// happens to be spelled as a reduction: exact, and identical on every rank whatever the
  /// decomposition — which is what lets the collision census, the merge decision and the split
  /// decision be taken redundantly and consistently everywhere (gate G6).
  virtual void allreduceSum(double* v, int n) { (void)v; (void)n; }
  /// Elementwise MAX of `n` doubles across the ranks. The transport used by a MERGE and a SPLIT:
  /// the source master writes its own colour into the buffer and every other rank writes zero, so
  /// the result is the union `max` on the destination's box, on every rank at once.
  virtual void allreduceMax(double* v, int n) { (void)v; (void)n; }
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
  /// Rung W4 item 1: form ONE force per face from the UNION colour instead of summing the markers'
  /// forces. `false` is rung W2's rule, kept ONLY as the ablation that reproduces the `channel_18`
  /// blow-up; it is not a production path. See `buildCsfForce` and `Solver::assembleBlockCsf`.
  bool csfUnion = true;
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
      if (!b.mine_ || !b.alive_)
        continue;
      clampFaceVelocity(b);
      b.adv_.advect(dt, step_);
      b.st_.recentred = false;
    }
    for (std::size_t k = 0; k < blocks_.size(); ++k)
      if (blocks_[k].mine_ && blocks_[k].alive_)
        recentre(k);
    exch_->syncTable(blocks_);
    for (auto& b : blocks_)
      b.mine_ = b.alive_ && (b.master == rank_);  // a RETIRED block belongs to nobody
    // Rung W1: re-balance the master assignment on the CURRENT boxes, before the scatter, so the
    // step's union is produced by the new owners and there is no half-migrated state anywhere.
    // The boxes have just been replicated, so `plannedMasters()` is the same on every rank.
    lastReassigned_ = 0;
    if (reassignEvery > 0 && ((step_ + 1) % reassignEvery) == 0)
      lastReassigned_ = assignMasters();
    exch_->scatterColourMax(blocks_, cLocal);
    for (auto& b : blocks_)
      if (b.mine_ && b.alive_)
        measure(b, dt);
    ++step_;
    t_ += dt;
    // Rung W4: the census, then the two models. A merge or a split leaves the UNION colour
    // unchanged bit for bit (a merge unions the same support; a split partitions the parent's
    // cells), so `cLocal` is still the field the closures must see and nothing is re-scattered.
    updateInteractions(dt);
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

  /// Per-block curvature cascade + the CSF face accumulators, summed into the caller's patches.
  /// `loc` is ZEROED first (the force is a per-step quantity, never accumulated). `nc` is 12 with
  /// `csfUnion` (the W4 accumulators, `VofBlock::csfAcc` order) and 3 without it (the W2 force).
  void computeCsf(SField* loc, int nc) {
    if (!exch_)
      throw std::runtime_error("peclet::flow::vof::VofBlockSet: no exchange installed");
    if (nc != (csfUnion ? 12 : 3))
      throw std::runtime_error("peclet::flow::vof::VofBlockSet::computeCsf: component count does "
                               "not match csfUnion");
    curvStats_ = VofCurvature::Stats{};
    for (auto& b : blocks_) {
      if (!b.mine_ || !b.alive_)
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
    exch_->scatterCsfSum(blocks_, loc, nc);
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
    b.curv_.useWorklist = curvProto.useWorklist;  // WO-V9: the compaction follows the prototype
    const long len = static_cast<long>(b.adv_.extent().x) * b.adv_.extent().y * b.adv_.extent().z;
    for (int c = 0; c < 3; ++c) {
      b.f_[c] = SField("vof::block::csf", len);
      if (csfUnion)
        for (int k = 0; k < 3; ++k)
          b.a_[c][k] = SField("vof::block::csf_acc", len);
    }
  }

  /// The V4 balanced-force CSF on the BLOCK's faces — the same `csfFaceCurvature` +
  /// `csfFaceForce` pair `Solver::addCsfRhs` applies to the global field, on the block's own
  /// colour and curvature. Formed over the inner box; the low face of an inner cell at local index
  /// 0 reads the block's ghost, which the margin guarantees is pure gas, so the force is exactly
  /// zero there and the block's force has compact support inside its own box.
  ///
  /// RUNG W4. What is scattered is no longer only the force. A face inside the OVERLAP of two
  /// markers receives two forces through UNPACK_SUM while the projection — which reads the UNION
  /// colour `max_blocks C` — sees a single colour jump there, so the balanced-force pairing that
  /// makes V4 exact (the force being the discrete gradient of `sigma kappa C` under the SAME
  /// difference operator the projection inverts) is broken exactly where two bubbles meet. Measured
  /// consequence: `channel_18` blows up inside one step at ~1.5 eddy turnovers, at a dt that is 4x
  /// below the one that fails, the moment two markers interpenetrate (WO-W3 finding 7).
  ///
  /// So each marker contributes what the owner needs to form ONE force from the UNION's own face
  /// difference: `A` (its W2 force, which IS the answer where it is alone), `W = |dC|`,
  /// `KW = kappa_f |dC|` and `N = 1`. The owner's rule is in `Solver::assembleBlockCsf`.
  void buildCsfForce(VofBlock& b) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const double sig = sigma, hh = h_;
    const bool uni = csfUnion;
    SField cv = b.adv_.colour(), kp = b.curv_.kappa(), kb = b.curv_.branch();
    const long sy = e.x, sz = static_cast<long>(e.x) * e.y;
    for (int c = 0; c < 3; ++c) {
      SField ff = b.f_[c];
      const long strd = (c == 0) ? 1 : (c == 1 ? sy : sz);
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>> pol(SExec(), {g, g, g},
                                                        {g + n.x, g + n.y, g + n.z});
      if (!uni) {
        Kokkos::parallel_for(
            "vof::block::csf_force", pol, KOKKOS_LAMBDA(int x, int y, int z) {
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
        continue;
      }
      SField fw = b.a_[c][0], fkw = b.a_[c][1], fn = b.a_[c][2];
      Kokkos::parallel_for(
          "vof::block::csf_acc", pol, KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            const double dC = cv(i) - cv(i - strd);
            double ks = 0.0, w = 0.0, kw = 0.0, nn = 0.0;
            if (dC != 0.0) {
              double kf = 0.0;
              // an ORPHAN face carries no weight: it must not pull kappa_union towards 0
              if (vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf)) {
                w = Kokkos::fabs(dC);
                ks = kf;
                kw = kf * w;
                nn = 1.0;
              }
            }
            ff(i) = ks;
            fw(i) = w;
            fkw(i) = kw;
            fn(i) = nn;
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
    std::vector<int> want = plannedMasters();
    std::vector<int> old(blocks_.size());
    long moved = 0;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      old[i] = blocks_[i].master;
      if (!blocks_[i].alive_)
        want[i] = old[i];  // a retired block has no state to move and no work to balance
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

  // ---- rung W3: checkpoint / restart of the block state ---------------------------------------
  //
  // A run longer than one job's wall clock has to write its markers out and read them back. The
  // union colour field is NOT enough to do that: `seedBox`/`finishSeeding` gather each marker out
  // of the UNION and clip to the seed extent, so two markers that touch each adopt a slice of the
  // other (WO-W12 open item 5, measured -2.7 % / +7.1 %). A block's ONLY state is its own inner
  // colour (everything else is either replicated -- the table -- or recomputed every step), so a
  // checkpoint is exactly {box, colour} per block and the restart below is EXACT whatever the
  // markers are doing.

  /// One block's INNER colour, copied to the host in x-fastest order over `blocks_[idx].box`.
  /// Empty on a rank that does not master the block (its state lives on the master).
  std::vector<double> blockColourHost(std::size_t idx) {
    VofBlock& b = blocks_.at(idx);
    if (!b.mine_ || !b.allocated_)
      return {};
    const I3 n = b.adv_.inner();
    SField c = b.adv_.colour();
    auto hc = Kokkos::create_mirror_view(c);
    Kokkos::deep_copy(hc, c);
    std::vector<double> out(static_cast<std::size_t>(n.x) * static_cast<std::size_t>(n.y) *
                            static_cast<std::size_t>(n.z));
    std::size_t q = 0;
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          out[q++] = hc(b.adv_.index(x, y, z));
    return out;
  }

  /// Seed a block whose INNER box is EXACTLY `bb` (not grown by the margin: `bb` is what
  /// `blockColourHost` was paired with) and whose colour is the given host array, x-fastest over
  /// `bb`. No seed clip runs -- the colour is given, not gathered out of a union.
  void seedBoxWithColour(const VofBox& bb, const std::vector<double>& colour) {
    VofBlock b;
    b.id = static_cast<long>(blocks_.size());
    b.master = masterOf(b.id);
    b.box = vofClampBox(bb, gs_, per_);
    b.mine_ = (b.master == rank_);
    blocks_.push_back(std::move(b));
    const std::size_t idx = blocks_.size() - 1;
    for (std::size_t k = 0; k + 1 < blocks_.size(); ++k)
      if (blocks_[k].mine_ && blocks_[k].allocated_)
        installHook(k);
    if (!blocks_[idx].mine_)
      return;
    allocate(idx);
    VofBlock& nb = blocks_[idx];
    const I3 n = nb.adv_.inner();
    const std::size_t want = static_cast<std::size_t>(n.x) * static_cast<std::size_t>(n.y) *
                             static_cast<std::size_t>(n.z);
    if (colour.size() != want)
      throw std::runtime_error(
          "peclet::flow::vof::VofBlockSet::seedBoxWithColour: colour size does not match the box");
    SField c = nb.adv_.colour();
    auto hc = Kokkos::create_mirror_view(c);
    Kokkos::deep_copy(hc, c);  // ghosts as allocated (zero); the fill below re-does them anyway
    std::size_t q = 0;
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          hc(nb.adv_.index(x, y, z)) = colour[q++];
    Kokkos::deep_copy(c, hc);
    Kokkos::fence();
    fillBlockGhosts(nb);
    measure(nb, 0.0);
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
      if (!b.alive_)
        continue;  // rung W4: a merged / split marker keeps its row but is no longer a marker
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
      if (!b.alive_)
        continue;
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
    for (int c = 0; c < 3; ++c) {
      b.f_[c] = SField();
      for (int k = 0; k < 3; ++k)
        b.a_[c][k] = SField();
    }
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

  // =============================================================================================
  // rung W4 (WO-W4) items 2-5: the pair census, coalescence and breakup as EXPLICIT models
  // =============================================================================================
  //
  // The container's promise is that two markers never coalesce numerically. The price of that
  // promise is that something else has to decide what a pair in contact does, and that something
  // is here: a census of every pair whose boxes overlap (film thickness, approach velocity,
  // collision Weber number), a coalescence model that may merge them, and a breakup model that may
  // split one marker into two. All three are OFF by default except the census, which changes no
  // number anywhere; `Never` is the coalescence default because it is the physically right one for
  // a bubbly flow and because the alternative would smuggle a numerical decision back in.

  /// Marker-pair census (item 2). Costs two small `MPI_Allreduce`s per step and changes no field.
  bool pairCensus = true;
  VofCoalescenceModel coalescence;
  /// Breakup (item 4): in-block connected-component labelling of `C > 1/2`; a block that holds
  /// more than one component for `breakupSteps` CONSECUTIVE steps is split. OFF by default — the
  /// labelling is an iterative propagation and costs O(box diameter) kernel launches per block per
  /// step, which is a real cost to pay silently (measured in the findings).
  bool breakup = false;
  long breakupSteps = 3;  ///< the design's `N_split`: ignore transient necks
  /// Satellite policy: a component smaller than `satelliteVolume` cell volumes is still made into
  /// its own (tiny) block — the design's "default keep" — and counted. Set it to a volume and
  /// `absorbSatellites = true` to fold such components into the largest child instead.
  double satelliteVolume = 0.0;
  bool absorbSatellites = false;
  int maxSplitChildren = 4;  ///< components beyond this are folded into the largest child
  /// The liquid properties the collision model needs (cell units, like `sigma`).
  double rhoLiquid = 1.0;
  double muLiquid = 1.0;

  const std::vector<VofPairStats>& pairs() const { return pairs_; }
  const std::vector<VofEvent>& events() const { return events_; }
  void clearEvents() { events_.clear(); }
  double time() const { return t_; }
  void setTime(double t) { t_ = t; }
  long mergeCount() const { return nMerge_; }
  long splitCount() const { return nSplit_; }
  long satelliteCount() const { return nSatellite_; }
  double orphanColour() const { return orphanColour_; }
  /// Colour the last merge did NOT put into the merged block: the SHARED liquid, i.e. cells both
  /// markers claimed, where the union `max` keeps one value and the sum of the two is larger. It
  /// is the exact accounting of why a merged volume is not always the sum, and it is zero for two
  /// markers whose supports are disjoint.
  double transportLost() const { return transportLost_; }
  /// Live (non-retired) blocks.
  std::size_t aliveCount() const {
    std::size_t k = 0;
    for (const auto& b : blocks_)
      if (b.alive_)
        ++k;
    return k;
  }

  /// The census + the two models, run once per `advect()` after the statistics are in.
  void updateInteractions(double dt) {
    if (!pairCensus && coalescence.model == VofCoalescence::Never && !breakup)
      return;
    if (pairCensus || coalescence.model != VofCoalescence::Never)
      censusPairs(dt);
    if (coalescence.model != VofCoalescence::Never)
      applyCoalescence();
    if (breakup)
      applyBreakup();
  }

  // ---- item 2: the census ----------------------------------------------------------------------

  /// Minimum-image displacement `b - a` on the periodic axes, in the caller's units.
  void minImage(const double a[3], const double b[3], double out[3]) const {
    const int g[3] = {gs_.x, gs_.y, gs_.z};
    for (int d = 0; d < 3; ++d) {
      double q = b[d] - a[d];
      const double L = g[d] * h_;
      if (per_[d] && L > 0.0)
        q -= L * std::round(q / L);
      out[d] = q;
    }
  }

  /// Do the two blocks' INNER boxes overlap (with the periodic wrap)? The inner box already
  /// carries the 3-cell margin, so an overlap means the two markers' colour supports are within
  /// ~6 cells — which is exactly the range over which their CSF bands can share a face.
  bool boxesOverlap(const VofBox& a, const VofBox& b) const {
    const int g[3] = {gs_.x, gs_.y, gs_.z};
    for (int d = 0; d < 3; ++d) {
      bool any = false;
      for (int k = -1; k <= 1 && !any; ++k) {
        if (k != 0 && !per_[d])
          continue;
        const int lo = b.lo[d] + k * g[d], hi = b.hi[d] + k * g[d];
        any = (lo < a.hi[d] && a.lo[d] < hi);
      }
      if (!any)
        return false;
    }
    return true;
  }

  void censusPairs(double dt) {
    const std::size_t nb = blocks_.size();
    // 1. replicate {alive, volume, centroid, velocity} -- the master writes, everyone else zero.
    std::vector<double> tab(nb * 8, 0.0);
    for (std::size_t i = 0; i < nb; ++i) {
      const VofBlock& b = blocks_[i];
      if (!b.alive_ || !b.mine_)
        continue;
      tab[8 * i + 0] = 1.0;
      tab[8 * i + 1] = b.st_.volume;
      for (int d = 0; d < 3; ++d) {
        tab[8 * i + 2 + d] = b.st_.centroid[d];
        tab[8 * i + 5 + d] = b.st_.velocity[d];
      }
    }
    if (exch_)
      exch_->allreduceSum(tab.data(), static_cast<int>(tab.size()));
    // 2. the pair list, from the REPLICATED boxes -- identical on every rank by construction.
    std::vector<std::pair<std::size_t, std::size_t>> pl;
    for (std::size_t i = 0; i < nb; ++i) {
      if (!blocks_[i].alive_ || tab[8 * i] <= 0.0 || !(tab[8 * i + 1] > 0.0))
        continue;
      for (std::size_t j = i + 1; j < nb; ++j) {
        if (!blocks_[j].alive_ || tab[8 * j] <= 0.0 || !(tab[8 * j + 1] > 0.0))
          continue;
        if (boxesOverlap(blocks_[i].box, blocks_[j].box))
          pl.emplace_back(i, j);
      }
    }
    // 3. each master ray-marches its OWN marker along the line of centres; one allreduce shares
    //    both halves. Two doubles per pair, so this is a handful of bytes.
    std::vector<double> ray(pl.size() * 2, 0.0);
    for (std::size_t k = 0; k < pl.size(); ++k) {
      const std::size_t i = pl[k].first, j = pl[k].second;
      double ci[3], cj[3], n[3];
      for (int d = 0; d < 3; ++d) {
        ci[d] = tab[8 * i + 2 + d];
        cj[d] = tab[8 * j + 2 + d];
      }
      minImage(ci, cj, n);
      const double L = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (!(L > 0.0))
        continue;
      for (int d = 0; d < 3; ++d)
        n[d] /= L;
      if (blocks_[i].mine_ && blocks_[i].allocated_)
        ray[2 * k + 0] = rayRadius(blocks_[i], ci, n, L);
      if (blocks_[j].mine_ && blocks_[j].allocated_) {
        const double m[3] = {-n[0], -n[1], -n[2]};
        ray[2 * k + 1] = rayRadius(blocks_[j], cj, m, L);
      }
    }
    if (exch_ && !ray.empty())
      exch_->allreduceSum(ray.data(), static_cast<int>(ray.size()));
    // 4. assemble, carry the per-pair state forward, and emit the events.
    std::vector<VofPairStats> out;
    out.reserve(pl.size());
    for (std::size_t k = 0; k < pl.size(); ++k) {
      const std::size_t i = pl[k].first, j = pl[k].second;
      VofPairStats p;
      p.idA = blocks_[i].id;
      p.idB = blocks_[j].id;
      p.volA = tab[8 * i + 1];
      p.volB = tab[8 * j + 1];
      double ci[3], cj[3], n[3], dv[3];
      for (int d = 0; d < 3; ++d) {
        ci[d] = tab[8 * i + 2 + d];
        cj[d] = tab[8 * j + 2 + d];
        dv[d] = tab[8 * j + 5 + d] - tab[8 * i + 5 + d];
      }
      minImage(ci, cj, n);
      const double L = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (L > 0.0)
        for (int d = 0; d < 3; ++d)
          n[d] /= L;
      p.dist = L / h_;
      p.dA = ray[2 * k + 0] / h_;
      p.dB = ray[2 * k + 1] / h_;
      p.film = p.dist - p.dA - p.dB;
      const double un = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];  // > 0 = separating
      p.approach = -un / h_;
      const double dA = 2.0 * std::cbrt(3.0 * p.volA / (4.0 * M_PI));
      const double dB = 2.0 * std::cbrt(3.0 * p.volB / (4.0 * M_PI));
      p.dEq = (dA + dB > 0.0) ? 2.0 * dA * dB / (dA + dB) : 0.0;
      p.weber = (sigma > 0.0) ? rhoLiquid * (un / h_) * (un / h_) * p.dEq / sigma : 0.0;
      // carry the history of this pair (keyed by the two ids, which never change under us)
      const VofPairStats* old = findPair(p.idA, p.idB);
      const bool contact = p.film <= coalescence.contactFilm;
      p.contactSteps = (old && contact) ? old->contactSteps + 1 : (contact ? 1 : 0);
      const bool draining = p.film < coalescence.hCrit;
      p.filmTime = (old && draining) ? old->filmTime + dt : (draining ? dt : 0.0);
      p.state = contact ? 2 : 1;
      if (!old)
        emit(VofEvent{0, step_, t_, p.idA, p.idB, p.film, p.approach, p.weber, p.dEq, p.volA,
                      p.volB, 0.0, 0});
      if (contact && (!old || old->state != 2))
        emit(VofEvent{1, step_, t_, p.idA, p.idB, p.film, p.approach, p.weber, p.dEq, p.volA,
                      p.volB, 0.0, 0});
      out.push_back(p);
    }
    pairs_.swap(out);
  }

  const VofPairStats* findPair(long a, long b) const {
    for (const auto& p : pairs_)
      if (p.idA == a && p.idB == b)
        return &p;
    return nullptr;
  }

  /// March along `n` from `c0` (both in the block's own unwrapped global frame, physical units)
  /// and return the LARGEST `s <= L` at which this marker's own colour, trilinearly interpolated,
  /// is still >= 1/2. That is the marker's interface radius in the direction of its partner. The
  /// step is `h/8`, so the film thickness is resolved to an eighth of a cell.
  double rayRadius(const VofBlock& b, const double c0[3], const double n[3], double L) const {
    const I3 e = b.adv_.extent();
    const int g = ghost_;
    const double hh = h_;
    const int ox = b.box.lo[0], oy = b.box.lo[1], oz = b.box.lo[2];
    const double x0 = c0[0], y0 = c0[1], z0 = c0[2];
    const double nx = n[0], ny = n[1], nz = n[2];
    int ns = static_cast<int>(L / (0.125 * hh)) + 1;
    if (ns < 2)
      ns = 2;
    if (ns > 4096)
      ns = 4096;
    const double ds = L / ns;
    SField c = b.adv_.colour();
    double r = 0.0;
    Kokkos::parallel_reduce(
        "vof::block::ray", Kokkos::RangePolicy<SExec>(SExec(), 0, ns + 1),
        KOKKOS_LAMBDA(int k, double& m) {
          const double s = k * ds;
          const double px = (x0 + s * nx) / hh - ox - 0.5, py = (y0 + s * ny) / hh - oy - 0.5,
                       pz = (z0 + s * nz) / hh - oz - 0.5;
          const int i0 = static_cast<int>(Kokkos::floor(px)),
                    j0 = static_cast<int>(Kokkos::floor(py)),
                    k0 = static_cast<int>(Kokkos::floor(pz));
          if (i0 < -g || j0 < -g || k0 < -g || i0 + 1 >= e.x - g || j0 + 1 >= e.y - g ||
              k0 + 1 >= e.z - g)
            return;  // outside the block: pure gas by the block's ghost policy
          const double tx = px - i0, ty = py - j0, tz = pz - k0;
          double v = 0.0;
          for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const double w = (dx ? tx : 1.0 - tx) * (dy ? ty : 1.0 - ty) *
                                 (dz ? tz : 1.0 - tz);
                v += w * c(L3(i0 + dx + g, j0 + dy + g, k0 + dz + g, e));
              }
          if (v >= 0.5 && s > m)
            m = s;
        },
        Kokkos::Max<double>(r));
    Kokkos::fence();
    return r;
  }

  void emit(const VofEvent& ev) {
    events_.push_back(ev);
    if (events_.size() > 8192)
      events_.erase(events_.begin(), events_.begin() + 4096);
  }

  // ---- item 3: coalescence as a MODEL ----------------------------------------------------------

  void applyCoalescence() {
    std::vector<std::pair<std::size_t, std::size_t>> todo;
    for (const auto& p : pairs_) {
      bool go = false;
      if (coalescence.model == VofCoalescence::Weber) {
        go = (p.film <= coalescence.contactFilm) && (p.weber < coalescence.weCrit);
      } else if (coalescence.model == VofCoalescence::Film) {
        // Prince & Blanch (1990): the film drains in t_d = C mu_l D_eq / sigma. The clock only
        // runs while the film is thinner than h_crit, and it is RESET the moment it is not.
        const double td = coalescence.drainC * muLiquid * p.dEq / (sigma > 0.0 ? sigma : 1.0);
        go = (p.film < coalescence.hCrit) && (p.filmTime >= td);
      }
      if (go)
        todo.emplace_back(static_cast<std::size_t>(p.idA), static_cast<std::size_t>(p.idB));
    }
    std::vector<char> used(blocks_.size(), 0);
    for (const auto& q : todo) {
      if (q.first >= used.size() || q.second >= used.size())
        continue;
      if (used[q.first] || used[q.second])
        continue;  // one merge per marker per step; a triple merges over two steps
      used[q.first] = used[q.second] = 1;
      mergePair(q.first, q.second);
    }
  }

  /// Merge two markers into one. The new block's box is the bounding box of the two (in a common
  /// unwrapped frame) and its colour is the union `max` — the ONLY place in the container where a
  /// `max` CREATES colour rather than deriving the field the closures read. The two old blocks are
  /// retired; their table rows survive because `id == table index` is an invariant.
  ///
  /// The transport is one `allreduceMax` over the new box: each old master writes its own colour
  /// into the buffer and every other rank writes zero, so the union lands on every rank at once
  /// and the merge needs no point-to-point code and no ordering assumption (gate G6).
  long mergePair(std::size_t i, std::size_t j) {
    const int g[3] = {gs_.x, gs_.y, gs_.z};
    VofBox bi = blocks_[i].box, bj = blocks_[j].box;
    int shift[3] = {0, 0, 0};
    for (int d = 0; d < 3; ++d) {
      if (!per_[d] || g[d] <= 0)
        continue;
      const double ca = 0.5 * (bi.lo[d] + bi.hi[d]), cb = 0.5 * (bj.lo[d] + bj.hi[d]);
      const int k = static_cast<int>(std::round((cb - ca) / g[d]));
      shift[d] = -k * g[d];
      bj.lo[d] += shift[d];
      bj.hi[d] += shift[d];
    }
    VofBox nbx;
    for (int d = 0; d < 3; ++d) {
      nbx.lo[d] = std::min(bi.lo[d], bj.lo[d]);
      nbx.hi[d] = std::max(bi.hi[d], bj.hi[d]);
    }
    nbx = vofClampBox(nbx, gs_, per_);
    const double vA = blocks_[i].st_.volume, vB = blocks_[j].st_.volume;
    const std::size_t k = addBlock(nbx);
    std::vector<double> buf(static_cast<std::size_t>(nbx.cells()), 0.0);
    const int z3[3] = {0, 0, 0};
    placeColour(i, nbx, z3, buf);
    placeColour(j, nbx, shift, buf);
    if (exch_)
      exch_->allreduceMax(buf.data(), static_cast<int>(buf.size()));
    installColour(k, buf);
    blocks_[k].splitArmed_ = false;  // it holds two components BY CONSTRUCTION; see `splitArmed_`
    double vNew = 0.0;
    for (double q : buf)
      vNew += q;
    retire(i);
    retire(j);
    ++nMerge_;
    const VofPairStats* p = findPair(blocks_[i].id, blocks_[j].id);
    emit(VofEvent{2, step_, t_, blocks_[i].id, blocks_[j].id, p ? p->film : 0.0,
                  p ? p->approach : 0.0, p ? p->weber : 0.0, p ? p->dEq : 0.0, vA, vB, vNew, 0});
    return blocks_[k].id;
  }

  // ---- item 4: breakup -------------------------------------------------------------------------

  void applyBreakup() {
    const std::size_t nb0 = blocks_.size();
    std::vector<double> flag(nb0 * 2, 0.0);
    std::vector<std::vector<int>> comp(nb0);
    for (std::size_t i = 0; i < nb0; ++i) {
      VofBlock& b = blocks_[i];
      if (!b.alive_ || !b.mine_ || !b.allocated_)
        continue;
      const int nc = labelComponents(b, comp[i]);
      if (!b.splitArmed_) {  // a freshly merged block: wait until its components have joined
        if (nc <= 1)
          b.splitArmed_ = true;
        b.splitSteps_ = 0;
        continue;
      }
      b.splitSteps_ = (nc > 1) ? b.splitSteps_ + 1 : 0;
      flag[2 * i + 0] = nc;
      flag[2 * i + 1] = static_cast<double>(b.splitSteps_);
    }
    if (exch_)
      exch_->allreduceSum(flag.data(), static_cast<int>(flag.size()));
    for (std::size_t i = 0; i < nb0; ++i)
      if (flag[2 * i + 0] > 1.5 && flag[2 * i + 1] >= static_cast<double>(breakupSteps))
        splitBlock(i, comp[i], static_cast<int>(flag[2 * i + 0] + 0.5));
  }

  /// Connected components of `C > 1/2` on the block's INNER box, by iterative min-label
  /// propagation on the device (6-connectivity), then a host pass that (a) adopts every remaining
  /// cell with `C > bubbleEps` into the component of a labelled neighbour, so no colour is
  /// orphaned and the child volumes sum EXACTLY to the parent's, (b) orders the components by
  /// volume (ties by lowest label: deterministic), and (c) folds satellites and any components
  /// beyond `maxSplitChildren` into the largest.
  ///
  /// `comp` comes back with one entry per INNER cell in x-fastest order: the component index, or
  /// -1 for a cell with no colour. The return value is the component count.
  int labelComponents(VofBlock& b, std::vector<int>& comp) {
    const I3 e = b.adv_.extent(), n = b.adv_.inner();
    const int g = ghost_;
    const long ncell = static_cast<long>(n.x) * n.y * n.z;
    SField lab("vof::block::label", static_cast<std::size_t>(ncell));
    SField nxt("vof::block::label2", static_cast<std::size_t>(ncell));
    SField c = b.adv_.colour();
    const int nx = n.x, ny = n.y, nz = n.z;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {0, 0, 0}, {nx, ny, nz});
    Kokkos::parallel_for(
        "vof::block::label_init", pol, KOKKOS_LAMBDA(int x, int y, int z) {
          const long q = x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny;
          lab(q) = (c(L3(x + g, y + g, z + g, e)) > 0.5) ? static_cast<double>(q) : -1.0;
        });
    Kokkos::fence();
    const int maxIt = 4 * (nx + ny + nz) + 8;
    for (int it = 0; it < maxIt; ++it) {
      long changed = 0;
      Kokkos::parallel_reduce(
          "vof::block::label_prop", pol,
          KOKKOS_LAMBDA(int x, int y, int z, long& ch) {
            const long q = x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny;
            double v = lab(q);
            if (v < 0.0) {
              nxt(q) = v;
              return;
            }
            const int dx[6] = {-1, 1, 0, 0, 0, 0}, dy[6] = {0, 0, -1, 1, 0, 0},
                      dz[6] = {0, 0, 0, 0, -1, 1};
            for (int d = 0; d < 6; ++d) {
              const int ax = x + dx[d], ay = y + dy[d], az = z + dz[d];
              if (ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz)
                continue;
              const double w =
                  lab(ax + static_cast<long>(ay) * nx + static_cast<long>(az) * nx * ny);
              if (w >= 0.0 && w < v)
                v = w;
            }
            if (v != lab(q))
              ++ch;
            nxt(q) = v;
          },
          changed);
      Kokkos::fence();
      Kokkos::deep_copy(lab, nxt);
      if (changed == 0)
        break;
    }
    // host bookkeeping: the O(#components) part, and the buffers the transport needs anyway
    auto hl = Kokkos::create_mirror_view(lab);
    Kokkos::deep_copy(hl, lab);
    const std::vector<double> col = blockColourHost(static_cast<std::size_t>(&b - blocks_.data()));
    std::vector<long> root(static_cast<std::size_t>(ncell));
    for (long q = 0; q < ncell; ++q)
      root[static_cast<std::size_t>(q)] = (hl(q) < 0.0) ? -1 : static_cast<long>(hl(q) + 0.5);
    // (a) adopt the leftover colour: repeat until nothing changes (bounded by the box diameter)
    const double eps = bubbleEps;
    for (int it = 0; it < nx + ny + nz + 4; ++it) {
      long moved = 0;
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x) {
            const long q = x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny;
            if (root[static_cast<std::size_t>(q)] >= 0)
              continue;
            if (!(std::fabs(col[static_cast<std::size_t>(q)]) > eps))
              continue;
            long best = -1;
            const int dx[6] = {-1, 1, 0, 0, 0, 0}, dy[6] = {0, 0, -1, 1, 0, 0},
                      dz[6] = {0, 0, 0, 0, -1, 1};
            for (int d = 0; d < 6; ++d) {
              const int ax = x + dx[d], ay = y + dy[d], az = z + dz[d];
              if (ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz)
                continue;
              const long w = root[static_cast<std::size_t>(
                  ax + static_cast<long>(ay) * nx + static_cast<long>(az) * nx * ny)];
              if (w >= 0 && (best < 0 || w < best))
                best = w;
            }
            if (best >= 0) {
              root[static_cast<std::size_t>(q)] = best;
              ++moved;
            }
          }
      if (moved == 0)
        break;
    }
    // (b) order by volume
    std::map<long, double> vol;
    for (long q = 0; q < ncell; ++q) {
      const long r = root[static_cast<std::size_t>(q)];
      if (r >= 0)
        vol[r] += col[static_cast<std::size_t>(q)];
    }
    std::vector<std::pair<long, double>> ord(vol.begin(), vol.end());
    std::stable_sort(ord.begin(), ord.end(),
                     [](const std::pair<long, double>& a, const std::pair<long, double>& b2) {
                       return a.second > b2.second;
                     });
    std::map<long, int> idx;
    int nc = 0;
    for (const auto& q : ord) {
      const bool satellite = (satelliteVolume > 0.0 && q.second < satelliteVolume);
      if (nc == 0 || (!(satellite && absorbSatellites) && nc < maxSplitChildren))
        idx[q.first] = nc++;
      else
        idx[q.first] = 0;  // folded into the largest child; volume is preserved exactly
      if (satellite && !absorbSatellites && idx[q.first] > 0)
        ++nSatellite_;
    }
    comp.assign(static_cast<std::size_t>(ncell), -1);
    double orphan = 0.0;
    for (long q = 0; q < ncell; ++q) {
      const long r = root[static_cast<std::size_t>(q)];
      if (r >= 0)
        comp[static_cast<std::size_t>(q)] = idx[r];
      else if (col[static_cast<std::size_t>(q)] != 0.0) {
        comp[static_cast<std::size_t>(q)] = 0;  // never DROP colour: the largest child adopts it
        orphan += col[static_cast<std::size_t>(q)];
      }
    }
    orphanColour_ += orphan;
    return nc;
  }

  /// Split one block into its `nc` components. Each child is a new block whose box is its
  /// component's extent grown by the margin and whose colour is the parent's WHERE that component
  /// is and zero elsewhere — so the children's volumes sum to the parent's EXACTLY (every cell of
  /// the parent belongs to exactly one child) and the union colour is unchanged, bit for bit.
  void splitBlock(std::size_t i, const std::vector<int>& comp, int nc) {
    if (nc < 2)
      return;
    const VofBlock& b = blocks_[i];
    const VofBox pbox = b.box;
    const long pid = b.id;
    const double pvol = b.st_.volume;
    const int nx = pbox.n(0), ny = pbox.n(1), nz = pbox.n(2);
    if (nc > maxSplitChildren)
      nc = maxSplitChildren;
    // 1. the children's boxes, replicated (the master computes, one allreduce broadcasts)
    std::vector<double> meta(static_cast<std::size_t>(nc) * 8, 0.0);
    const bool mine = b.mine_ && b.allocated_ && !comp.empty();
    if (mine) {
      for (int k = 0; k < nc; ++k) {
        int lo[3] = {nx, ny, nz}, hi[3] = {-1, -1, -1};
        double v = 0.0;
        for (int z = 0; z < nz; ++z)
          for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
              const long q = x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny;
              if (comp[static_cast<std::size_t>(q)] != k)
                continue;
              const int p[3] = {x, y, z};
              for (int d = 0; d < 3; ++d) {
                lo[d] = p[d] < lo[d] ? p[d] : lo[d];
                hi[d] = p[d] > hi[d] ? p[d] : hi[d];
              }
              v += 1.0;  // extent only; the volume is measured after the child is built
            }
        if (hi[0] < lo[0])
          continue;
        for (int d = 0; d < 3; ++d) {
          meta[8 * k + d] = lo[d] + pbox.lo[d];
          meta[8 * k + 3 + d] = hi[d] + 1 + pbox.lo[d];
        }
        meta[8 * k + 6] = 1.0;
        meta[8 * k + 7] = v;
      }
    }
    if (exch_)
      exch_->allreduceSum(meta.data(), static_cast<int>(meta.size()));
    // 2. create the children and transport their colour
    long firstChild = -1;
    int made = 0;
    for (int k = 0; k < nc; ++k) {
      if (meta[8 * k + 6] < 0.5)
        continue;
      VofBox cb;
      for (int d = 0; d < 3; ++d) {
        cb.lo[d] = static_cast<int>(std::lround(meta[8 * k + d]));
        cb.hi[d] = static_cast<int>(std::lround(meta[8 * k + 3 + d]));
      }
      cb = vofClampBox(VofBox::grown(cb, margin_), gs_, per_);
      const std::size_t ci = addBlock(cb);
      std::vector<double> buf(static_cast<std::size_t>(cb.cells()), 0.0);
      if (mine)
        placeComponent(i, comp, k, cb, buf);
      if (exch_)
        exch_->allreduceMax(buf.data(), static_cast<int>(buf.size()));
      installColour(ci, buf);
      if (firstChild < 0)
        firstChild = blocks_[ci].id;
      ++made;
    }
    retire(i);
    ++nSplit_;
    emit(VofEvent{3, step_, t_, pid, firstChild, 0.0, 0.0, 0.0, 0.0, pvol, 0.0, 0.0, made});
  }

  // ---- the plumbing the two models share -------------------------------------------------------

  /// Append a table row (on EVERY rank) for a new block on `bx`, allocated (zeroed) on its master.
  std::size_t addBlock(const VofBox& bx) {
    VofBlock nb;
    nb.id = static_cast<long>(blocks_.size());
    nb.master = masterOf(nb.id);
    nb.box = bx;
    nb.mine_ = (nb.master == rank_);
    blocks_.push_back(std::move(nb));
    const std::size_t k = blocks_.size() - 1;
    for (std::size_t q = 0; q + 1 < blocks_.size(); ++q)
      if (blocks_[q].mine_ && blocks_[q].allocated_)
        installHook(q);  // push_back may have moved the advectors
    if (blocks_[k].mine_)
      allocate(k);
    return k;
  }

  /// Retire a block: free its state, empty its box, keep its row.
  void retire(std::size_t idx) {
    VofBlock& b = blocks_[idx];
    if (b.mine_ && b.allocated_)
      release(idx);
    const long id = b.id;
    b.alive_ = false;
    b.mine_ = false;
    b.box = VofBox{};
    b.st_ = VofBlockStats{};
    b.st_.id = id;
  }

  /// Write block `src`'s inner colour into `buf` (indexed over `bx`, x-fastest), offset by
  /// `shift` global cells; cells outside `bx` are dropped (there are none by construction).
  void placeColour(std::size_t src, const VofBox& bx, const int shift[3], std::vector<double>& buf) {
    if (!blocks_[src].mine_ || !blocks_[src].allocated_)
      return;
    const std::vector<double> col = blockColourHost(src);
    const VofBox sb = blocks_[src].box;
    const int nx = sb.n(0), ny = sb.n(1), nz = sb.n(2);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          const int gq[3] = {x + sb.lo[0] + shift[0], y + sb.lo[1] + shift[1],
                             z + sb.lo[2] + shift[2]};
          const double v = col[static_cast<std::size_t>(
              x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny)];
          long t[3];
          if (!targetIndex(gq, bx, t)) {
            transportLost_ += v;  // a cell of the source that the target box does not hold
            continue;
          }
          const long o = t[0] + t[1] * bx.n(0) + t[2] * static_cast<long>(bx.n(0)) * bx.n(1);
          double& d = buf[static_cast<std::size_t>(o)];
          // The union `max` keeps ONE value where both markers claim a cell; the smaller one is
          // the SHARED liquid, and it is exactly the amount by which a merged volume falls short
          // of the sum of the two. Accounted, never silent.
          transportLost_ += (v < d) ? v : d;
          if (v > d)
            d = v;
        }
  }

  /// The same, restricted to the cells of one component (the split's transport).
  void placeComponent(std::size_t src, const std::vector<int>& comp, int k, const VofBox& bx,
                      std::vector<double>& buf) {
    const std::vector<double> col = blockColourHost(src);
    const VofBox sb = blocks_[src].box;
    const int nx = sb.n(0), ny = sb.n(1), nz = sb.n(2);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          const long q = x + static_cast<long>(y) * nx + static_cast<long>(z) * nx * ny;
          if (comp[static_cast<std::size_t>(q)] != k)
            continue;
          const int gq[3] = {x + sb.lo[0], y + sb.lo[1], z + sb.lo[2]};
          long t[3];
          if (!targetIndex(gq, bx, t))
            continue;
          const long o = t[0] + t[1] * bx.n(0) + t[2] * static_cast<long>(bx.n(0)) * bx.n(1);
          const double v = col[static_cast<std::size_t>(q)];
          if (v > buf[static_cast<std::size_t>(o)])
            buf[static_cast<std::size_t>(o)] = v;
        }
  }

  /// Global cell `gq` -> index inside `bx`, trying the periodic images. False if it is not in.
  bool targetIndex(const int gq[3], const VofBox& bx, long t[3]) const {
    const int g[3] = {gs_.x, gs_.y, gs_.z};
    for (int d = 0; d < 3; ++d) {
      long q = gq[d] - bx.lo[d];
      if ((q < 0 || q >= bx.n(d)) && per_[d] && g[d] > 0) {
        if (q < 0)
          q += g[d];
        else
          q -= g[d];
      }
      if (q < 0 || q >= bx.n(d))
        return false;
      t[d] = q;
    }
    return true;
  }

  /// Install a transported host colour into block `idx` (its master only), then fill and measure.
  void installColour(std::size_t idx, const std::vector<double>& buf) {
    VofBlock& b = blocks_[idx];
    if (!b.mine_ || !b.allocated_)
      return;
    const I3 n = b.adv_.inner();
    SField c = b.adv_.colour();
    auto hc = Kokkos::create_mirror_view(c);
    Kokkos::deep_copy(hc, c);
    std::size_t q = 0;
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          hc(b.adv_.index(x, y, z)) = buf[q++];
    Kokkos::deep_copy(c, hc);
    Kokkos::fence();
    fillBlockGhosts(b);
    measure(b, 0.0);
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
  // rung W4
  std::vector<VofPairStats> pairs_;
  std::vector<VofEvent> events_;
  double t_ = 0.0;
  long nMerge_ = 0, nSplit_ = 0, nSatellite_ = 0;
  double orphanColour_ = 0.0;
  double transportLost_ = 0.0;
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_CONTAINER_HPP
