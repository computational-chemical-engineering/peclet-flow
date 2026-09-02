/// @file
/// @brief flow — VoF Part III rung W0: the block gather/scatter (`VOF_PLAN.md` §10 design items
/// 2 and 4). MPI-guarded; with `size == 1` it is a strided copy and no MPI symbol is referenced,
/// so the single-rank Python module builds this header unchanged.
///
/// ## The exchange, and why it is NOT an NBX handshake
///
/// The block table (id, box, master) is replicated on every rank and the flow decomposition is
/// replicated too (`BlockDecomposer::origins()/sizes()` are global on every rank). So **every rank
/// can compute every message size** as a pure function of those two: the overlap of a block's box
/// with a rank's owned box. There is nothing dynamic to discover, and a sparse-neighbourhood
/// handshake (`core::halo::NbxEngine`) would pay a full round of unexpected-message discovery to
/// learn what both sides already know. Plain `MPI_Isend/Irecv` with precomputed counts is the right
/// primitive here; the NBX engine stays for the genuinely dynamic case (rung W1's redistribution,
/// where the *assignment* changes).
///
/// ## The plan
///
/// A block's box lives in UNWRAPPED global indices and may hang outside `[0, gs)`. Per axis it is
/// cut into contiguous GLOBAL runs (`vofAxisRuns`): one run on a non-periodic axis (the outside
/// part has no owner and is the block's own clamp fill), one or two on a periodic axis that crosses
/// the seam. The Cartesian product of the runs gives at most 8 global sub-boxes, each with a known
/// block-local offset; intersecting each with each rank's owned box gives the pieces. For one
/// (block, rank) pair the pieces are concatenated in a canonical order (sub-box index, then
/// x-fastest inside the piece) — both sides walk the same loop, so both agree on the layout without
/// exchanging it.
///
/// **Arrival order cannot matter**, which is what makes gate G1/G4 a *bitwise* gate: the runs
/// partition the block's box, so every block-local cell is written by exactly one piece from
/// exactly one owner; and the scatter combines with `max`, which is exact and commutative.
///
/// ## Staging
///
/// Packing is host-side (`create_mirror_view_and_copy` per call). W0 is a correctness rung and the
/// block payloads are small (a 20³ extended block is 190 KB of face velocity); the device-resident
/// packing kernel + the CUDA-aware path are the W1/W2 optimisation, exactly as `core`'s grid halo
/// grew its device-resident variant after the host-staged one was proven.
#ifndef PECLET_FLOW_VOF_BLOCK_EXCHANGE_HPP
#define PECLET_FLOW_VOF_BLOCK_EXCHANGE_HPP

#include <algorithm>
#include <array>
#include <Kokkos_Core.hpp>
#include <map>
#include <stdexcept>
#include <vector>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>
#endif

#include "vof/block_container.hpp"

namespace peclet::flow::vof {

/// One contiguous transfer: the global cells `g` of a block that rank `r` owns, and where they sit
/// in the block's EXTENDED local index space.
struct VofPiece {
  int rank = 0;
  VofBox g;             ///< the overlap, in global cell indices
  int loc[3]{0, 0, 0};  ///< block-local index (0-based within the block's box) of `g.lo`
  long cells() const { return g.cells(); }
};

/// Build the pieces of one block box against every rank's owned box, in canonical order.
inline void vofBuildPieces(const VofBox& box, I3 gs, const std::array<bool, 3>& per,
                           const std::vector<VofBox>& rankBox, std::vector<VofPiece>& out) {
  out.clear();
  VofRun rx[5], ry[5], rz[5];
  const int nx = vofAxisRuns(box.lo[0], box.hi[0], gs.x, per[0], rx);
  const int ny = vofAxisRuns(box.lo[1], box.hi[1], gs.y, per[1], ry);
  const int nz = vofAxisRuns(box.lo[2], box.hi[2], gs.z, per[2], rz);
  for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix) {
        VofBox s;
        s.lo[0] = rx[ix].g0;
        s.hi[0] = rx[ix].g0 + rx[ix].len;
        s.lo[1] = ry[iy].g0;
        s.hi[1] = ry[iy].g0 + ry[iy].len;
        s.lo[2] = rz[iz].g0;
        s.hi[2] = rz[iz].g0 + rz[iz].len;
        const int base[3] = {rx[ix].loc0, ry[iy].loc0, rz[iz].loc0};
        for (std::size_t r = 0; r < rankBox.size(); ++r) {
          const VofBox o = VofBox::intersect(s, rankBox[r]);
          if (o.empty())
            continue;
          VofPiece p;
          p.rank = static_cast<int>(r);
          p.g = o;
          for (int d = 0; d < 3; ++d)
            p.loc[d] = base[d] + (o.lo[d] - s.lo[d]);
          out.push_back(p);
        }
      }
}

/// The gather / scatter / table replication of the block container.
class VofBlockExchange : public VofBlockExchangeBase {
 public:
  /// This rank's own patch: the extended block the face velocity and the union colour live on.
  struct Patch {
    I3 e{0, 0, 0};  ///< extended extent
    I3 n{0, 0, 0};  ///< inner extent
    I3 o{0, 0, 0};  ///< global index of inner cell (0,0,0)
    int g = 0;      ///< ghost width
  };

  /// @param rankBox  the owned inner box of EVERY rank, in global cells (from the flow
  ///                 `BlockDecomposer`; a single element covering the grid when serial)
  void init(I3 gs, std::array<bool, 3> per, std::vector<VofBox> rankBox, int rank) {
    gs_ = gs;
    per_ = per;
    rankBox_ = std::move(rankBox);
    rank_ = rank;
    size_ = static_cast<int>(rankBox_.size());
    if (rank_ < 0 || rank_ >= size_)
      throw std::runtime_error("peclet::flow::vof::VofBlockExchange: rank out of range");
  }

  /// The local face velocity, in the ADVECTOR's high-face convention (i.e. already bridged with
  /// `vof::copyFaceVelocity`), on the local patch.
  void setPatch(Patch p, SField uf, SField vf, SField wf) {
    patch_ = p;
    lu_ = uf;
    lv_ = vf;
    lw_ = wf;
  }

#ifdef PECLET_FLOW_MPI
  void setComm(MPI_Comm c) {
    comm_ = c;
    hasComm_ = true;
  }
#endif

  long gatherBytes() const override { return gBytes_; }
  long scatterBytes() const override { return sBytes_; }
  /// Messages posted by the last gather / scatter on this rank (sends + receives).
  long gatherMessages() const { return gMsgs_; }
  long scatterMessages() const { return sMsgs_; }

  // ================================================================== rung W1 item (b):
  // DEVICE-RESIDENT packing.
  //
  // The pack/unpack kernels run in the block's own memory space, so the only host traffic left is
  // the MPI staging copy of each MESSAGE (one `deep_copy` per peer, not one mirror of the whole
  // local patch per step) -- and for the master's OWN cells there is no host traffic at all: the
  // copy is device-to-device. That is exactly the order `core`'s grid halo grew its device-resident
  // variant (host-staged first, proven, then the device pack with a host staging buffer at the MPI
  // boundary), and the reason the staging survives is that the MPI here is not guaranteed
  // CUDA-aware. Every step is a copy of a double, so the device path is BITWISE the host path;
  // `deviceStaging = false` selects the W0 arm for the measurement.
  //
  // The four transfers of the container are one pattern with four instantiations:
  //
  //   gatherFaceVel     EXTENDED box, 3 components, patch -> block          (blockBase = 0)
  //   gatherColour      INNER box,    1 component,  patch -> block          (blockBase = ghost)
  //   scatterColourMax  INNER box,    1 component,  block -> patch, MAX     (blockBase = ghost)
  //   scatterForceSum   INNER box,    3 components, block -> patch, SUM     (blockBase = ghost)
  //
  // so `gatherImpl` / `scatterImpl` take the box selector, the component count, the block-array
  // base offset and (for the scatter) the combine rule, and nothing else differs.
  bool deviceStaging = true;

  /// patch -> block. `useExtended` picks the box; `blockBase` is 0 for an array indexed over the
  /// extended box and `ghost` for one indexed over the inner box.
  void gatherImpl(std::vector<VofBlock>& blocks, int ghost, bool useExtended, int nc,
                  const SField* loc, SField (*blockView)(VofBlock&, int), int blockBase) {
    gBytes_ = 0;
    gMsgs_ = 0;
    bufSeq_ = 0;
    struct Recv {
      std::size_t bi;
      int from;
      long cells;
      SField dbuf;
      HostView hbuf;
    };
    std::vector<Recv> recvs;
    std::vector<SField> sendD(blocks.size());
    std::vector<HostView> sendH(blocks.size());
#ifdef PECLET_FLOW_MPI
    std::vector<MPI_Request> reqs;
#endif
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      VofBlock& b = blocks[bi];
      const VofBox box = useExtended ? b.extended(ghost) : b.box;
      vofBuildPieces(box, gs_, per_, rankBox_, pieces_);
      const bool master = (b.master == rank_);
      if (master)
        for (int c = 0; c < nc; ++c)
          Kokkos::deep_copy(blockView(b, c), 0.0);
      std::map<int, long> counts;
      for (const auto& p : pieces_)
        counts[p.rank] += p.cells();
      if (master)
        for (const auto& kv : counts) {
          if (kv.first == rank_)
            continue;
          Recv r;
          r.bi = bi;
          r.from = kv.first;
          r.cells = kv.second;
          r.dbuf = buffer(nc * kv.second);
          r.hbuf = Kokkos::create_mirror_view(r.dbuf);
          recvs.push_back(std::move(r));
        }
      if (!counts.count(rank_))
        continue;
      if (master) {  // device-to-device, no host traffic at all
        for (const auto& p : pieces_)
          if (p.rank == rank_)
            movePiece(p, nc, loc, blockView, b, blockBase, /*toBlock=*/true);
      } else {
        sendD[bi] = buffer(nc * counts[rank_]);
        long off = 0;
        for (const auto& p : pieces_)
          if (p.rank == rank_)
            packFromPatch(p, nc, loc, sendD[bi], off);
        Kokkos::fence();
        sendH[bi] = Kokkos::create_mirror_view(sendD[bi]);
        Kokkos::deep_copy(sendH[bi], sendD[bi]);
      }
    }
    Kokkos::fence();
#ifdef PECLET_FLOW_MPI
    if (size_ > 1) {
      requireComm();
      for (auto& r : recvs) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(r.hbuf.data(), static_cast<int>(r.hbuf.extent(0)), MPI_DOUBLE, r.from,
                  tagOf(blocks[r.bi].id, 0), comm_, &reqs.back());
        gBytes_ += static_cast<long>(r.hbuf.extent(0)) * 8;
        ++gMsgs_;
      }
      for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        if (sendH[bi].extent(0) == 0)
          continue;
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(sendH[bi].data(), static_cast<int>(sendH[bi].extent(0)), MPI_DOUBLE,
                  blocks[bi].master, tagOf(blocks[bi].id, 0), comm_, &reqs.back());
        gBytes_ += static_cast<long>(sendH[bi].extent(0)) * 8;
        ++gMsgs_;
      }
      if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
#endif
    for (auto& r : recvs) {
      VofBlock& b = blocks[r.bi];
      Kokkos::deep_copy(r.dbuf, r.hbuf);
      vofBuildPieces(useExtended ? b.extended(ghost) : b.box, gs_, per_, rankBox_, pieces_);
      long off = 0;
      for (const auto& p : pieces_)
        if (p.rank == r.from)
          unpackToBlock(p, nc, r.dbuf, off, blockView, b, blockBase);
    }
    Kokkos::fence();
  }

  /// block -> patch, combined with `op` (0 = max, 1 = sum). The patch's INNER region is zeroed
  /// first: both combines start from an empty field (UNPACK_MAX's empty union is pure gas;
  /// UNPACK_SUM's is no force), so a cell no marker covers reads exactly 0.
  void scatterImpl(std::vector<VofBlock>& blocks, int nc, SField* loc,
                   SField (*blockView)(VofBlock&, int), int blockBase, int op) {
    sBytes_ = 0;
    sMsgs_ = 0;
    bufSeq_ = 0;
    zeroPatchInner(nc, loc);
    struct Recv {
      std::size_t bi;
      int from;
      SField dbuf;
      HostView hbuf;
    };
    std::vector<Recv> recvs;
    std::vector<SField> sendD;
    std::vector<HostView> sendH;
    std::vector<int> sendTo;
    std::vector<long> sendTag;
#ifdef PECLET_FLOW_MPI
    std::vector<MPI_Request> reqs;
#endif
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      VofBlock& b = blocks[bi];
      vofBuildPieces(b.box, gs_, per_, rankBox_, pieces_);
      const bool master = (b.master == rank_);
      std::map<int, long> counts;
      for (const auto& p : pieces_)
        counts[p.rank] += p.cells();
      if (master) {
        for (const auto& kv : counts) {
          if (kv.first == rank_) {
            for (const auto& p : pieces_)
              if (p.rank == rank_)
                combineBlockIntoPatch(p, nc, blockView, b, blockBase, loc, op);
            continue;
          }
          SField d = buffer(nc * kv.second);
          long off = 0;
          for (const auto& p : pieces_)
            if (p.rank == kv.first)
              packFromBlock(p, nc, blockView, b, blockBase, d, off);
          Kokkos::fence();
          HostView h = Kokkos::create_mirror_view(d);
          Kokkos::deep_copy(h, d);
          sendD.push_back(d);
          sendH.push_back(h);
          sendTo.push_back(kv.first);
          sendTag.push_back(b.id);
        }
      } else if (counts.count(rank_)) {
        Recv r;
        r.bi = bi;
        r.from = b.master;
        r.dbuf = buffer(nc * counts[rank_]);
        r.hbuf = Kokkos::create_mirror_view(r.dbuf);
        recvs.push_back(std::move(r));
      }
    }
    Kokkos::fence();
#ifdef PECLET_FLOW_MPI
    if (size_ > 1) {
      requireComm();
      for (auto& r : recvs) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(r.hbuf.data(), static_cast<int>(r.hbuf.extent(0)), MPI_DOUBLE, r.from,
                  tagOf(blocks[r.bi].id, 1), comm_, &reqs.back());
        sBytes_ += static_cast<long>(r.hbuf.extent(0)) * 8;
        ++sMsgs_;
      }
      for (std::size_t k = 0; k < sendH.size(); ++k) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(sendH[k].data(), static_cast<int>(sendH[k].extent(0)), MPI_DOUBLE, sendTo[k],
                  tagOf(sendTag[k], 1), comm_, &reqs.back());
        sBytes_ += static_cast<long>(sendH[k].extent(0)) * 8;
        ++sMsgs_;
      }
      if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
#endif
    for (auto& r : recvs) {
      VofBlock& b = blocks[r.bi];
      Kokkos::deep_copy(r.dbuf, r.hbuf);
      vofBuildPieces(b.box, gs_, per_, rankBox_, pieces_);
      long off = 0;
      for (const auto& p : pieces_)
        if (p.rank == rank_)
          combineBufIntoPatch(p, nc, r.dbuf, off, loc, op);
    }
    Kokkos::fence();
  }

  // ---- the four device kernels (public for nvcc's extended-lambda rule) -----------------------
  void zeroPatchInner(int nc, SField* loc) {
    const I3 e = patch_.e, n = patch_.n;
    const int g = patch_.g;
    for (int c = 0; c < nc; ++c) {
      SField f = loc[c];
      Kokkos::parallel_for(
          "vof::block::zero_patch",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n.x, n.y, n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) { f(L3(x + g, y + g, z + g, e)) = 0.0; });
    }
    Kokkos::fence();
  }

  void packFromPatch(const VofPiece& p, int nc, const SField* loc, SField buf, long& off) {
    const I3 e = patch_.e, o = patch_.o;
    const int g = patch_.g;
    const int lo0 = p.g.lo[0], lo1 = p.g.lo[1], lo2 = p.g.lo[2];
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const long base = off;
    SField s0 = loc[0], s1 = nc > 1 ? loc[1] : loc[0], s2 = nc > 2 ? loc[2] : loc[0];
    Kokkos::parallel_for(
        "vof::block::pack_patch",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(lo0 + x - o.x + g, lo1 + y - o.y + g, lo2 + z - o.z + g, e);
          const long k = base + x + static_cast<long>(y) * n0 + static_cast<long>(z) * n0 * n1;
          buf(nc * k + 0) = s0(i);
          if (nc > 1)
            buf(nc * k + 1) = s1(i);
          if (nc > 2)
            buf(nc * k + 2) = s2(i);
        });
    off += static_cast<long>(n0) * n1 * n2;
  }

  void unpackToBlock(const VofPiece& p, int nc, SField buf, long& off,
                     SField (*blockView)(VofBlock&, int), VofBlock& b, int blockBase) {
    const I3 be = b.advector().extent();
    const int lo0 = p.g.lo[0], lo1 = p.g.lo[1], lo2 = p.g.lo[2];
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const int q0 = p.loc[0] + blockBase, q1 = p.loc[1] + blockBase, q2 = p.loc[2] + blockBase;
    const long base = off;
    (void)lo0;
    (void)lo1;
    (void)lo2;
    for (int c = 0; c < nc; ++c) {
      SField d = blockView(b, c);
      const int cc = c;
      Kokkos::parallel_for(
          "vof::block::unpack_block",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long k = base + x + static_cast<long>(y) * n0 + static_cast<long>(z) * n0 * n1;
            d(L3(q0 + x, q1 + y, q2 + z, be)) = buf(nc * k + cc);
          });
    }
    off += static_cast<long>(n0) * n1 * n2;
  }

  /// The master's OWN cells: patch <-> block with no buffer and no host traffic.
  void movePiece(const VofPiece& p, int nc, const SField* loc,
                 SField (*blockView)(VofBlock&, int), VofBlock& b, int blockBase, bool toBlock) {
    const I3 e = patch_.e, o = patch_.o, be = b.advector().extent();
    const int g = patch_.g;
    const int lo0 = p.g.lo[0], lo1 = p.g.lo[1], lo2 = p.g.lo[2];
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const int q0 = p.loc[0] + blockBase, q1 = p.loc[1] + blockBase, q2 = p.loc[2] + blockBase;
    for (int c = 0; c < nc; ++c) {
      SField src = loc[c], dst = blockView(b, c);
      if (!toBlock)
        throw std::runtime_error("vof::block: movePiece is patch->block only");
      Kokkos::parallel_for(
          "vof::block::move_local",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            dst(L3(q0 + x, q1 + y, q2 + z, be)) =
                src(L3(lo0 + x - o.x + g, lo1 + y - o.y + g, lo2 + z - o.z + g, e));
          });
    }
  }

  void packFromBlock(const VofPiece& p, int nc, SField (*blockView)(VofBlock&, int), VofBlock& b,
                     int blockBase, SField buf, long& off) {
    const I3 be = b.advector().extent();
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const int q0 = p.loc[0] + blockBase, q1 = p.loc[1] + blockBase, q2 = p.loc[2] + blockBase;
    const long base = off;
    for (int c = 0; c < nc; ++c) {
      SField src = blockView(b, c);
      const int cc = c;
      Kokkos::parallel_for(
          "vof::block::pack_block",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long k = base + x + static_cast<long>(y) * n0 + static_cast<long>(z) * n0 * n1;
            buf(nc * k + cc) = src(L3(q0 + x, q1 + y, q2 + z, be));
          });
    }
    off += static_cast<long>(n0) * n1 * n2;
  }

  void combineBlockIntoPatch(const VofPiece& p, int nc, SField (*blockView)(VofBlock&, int),
                             VofBlock& b, int blockBase, SField* loc, int op) {
    const I3 e = patch_.e, o = patch_.o, be = b.advector().extent();
    const int g = patch_.g;
    const int lo0 = p.g.lo[0], lo1 = p.g.lo[1], lo2 = p.g.lo[2];
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const int q0 = p.loc[0] + blockBase, q1 = p.loc[1] + blockBase, q2 = p.loc[2] + blockBase;
    for (int c = 0; c < nc; ++c) {
      SField src = blockView(b, c), dst = loc[c];
      const int opc = op;
      Kokkos::parallel_for(
          "vof::block::combine_block",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const double v = src(L3(q0 + x, q1 + y, q2 + z, be));
            const long i = L3(lo0 + x - o.x + g, lo1 + y - o.y + g, lo2 + z - o.z + g, e);
            if (opc == 0) {
              if (v > dst(i))
                dst(i) = v;
            } else {
              dst(i) += v;
            }
          });
    }
  }

  void combineBufIntoPatch(const VofPiece& p, int nc, SField buf, long& off, SField* loc, int op) {
    const I3 e = patch_.e, o = patch_.o;
    const int g = patch_.g;
    const int lo0 = p.g.lo[0], lo1 = p.g.lo[1], lo2 = p.g.lo[2];
    const int n0 = p.g.n(0), n1 = p.g.n(1), n2 = p.g.n(2);
    const long base = off;
    for (int c = 0; c < nc; ++c) {
      SField dst = loc[c];
      const int cc = c, opc = op;
      Kokkos::parallel_for(
          "vof::block::combine_buf",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n0, n1, n2}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long k = base + x + static_cast<long>(y) * n0 + static_cast<long>(z) * n0 * n1;
            const double v = buf(nc * k + cc);
            const long i = L3(lo0 + x - o.x + g, lo1 + y - o.y + g, lo2 + z - o.z + g, e);
            if (opc == 0) {
              if (v > dst(i))
                dst(i) = v;
            } else {
              dst(i) += v;
            }
          });
    }
    off += static_cast<long>(n0) * n1 * n2;
  }

  // ---- the four public transfers ----------------------------------------------------------
  static SField faceViewOf(VofBlock& b, int c) { return b.advector().faceVel(c); }
  static SField colourViewOf(VofBlock& b, int c) {
    (void)c;
    return b.advector().colour();
  }
  static SField forceViewOf(VofBlock& b, int c) { return b.csfForce(c); }

  void gatherFaceVel(std::vector<VofBlock>& blocks, int ghost) override {
    if (!deviceStaging) {
      gatherFaceVelHost(blocks, ghost);
      return;
    }
    SField loc[3] = {lu_, lv_, lw_};
    gatherImpl(blocks, ghost, /*useExtended=*/true, 3, loc, &faceViewOf, 0);
  }

  void gatherColour(std::vector<VofBlock>& blocks, SField cLocal) override {
    SField loc[1] = {cLocal};
    gatherImpl(blocks, 0, /*useExtended=*/false, 1, loc, &colourViewOf, ghostOf(blocks));
  }

  void scatterColourMax(std::vector<VofBlock>& blocks, SField cLocal) override {
    if (!deviceStaging) {
      scatterColourMaxHost(blocks, cLocal);
      return;
    }
    SField loc[1] = {cLocal};
    scatterImpl(blocks, 1, loc, &colourViewOf, ghostOf(blocks), /*op=*/0);
  }

  void scatterForceSum(std::vector<VofBlock>& blocks, SField fx, SField fy, SField fz) override {
    SField loc[3] = {fx, fy, fz};
    scatterImpl(blocks, 3, loc, &forceViewOf, ghostOf(blocks), /*op=*/1);
  }

  /// Move a block's colour from its old master to its new one after a re-assignment (rung W1
  /// item a). The two boxes are identical (the table was replicated before the re-assignment) so
  /// the whole extended array travels as one contiguous message and lands bit for bit.
  void migrateColour(std::vector<VofBlock>& blocks, const std::vector<int>& oldMaster) override {
    mBytes_ = 0;
    mMsgs_ = 0;
    bufSeq_ = 0;
#ifdef PECLET_FLOW_MPI
    if (size_ <= 1)
      return;
    requireComm();
    std::vector<MPI_Request> reqs;
    std::vector<SField> dbuf(blocks.size());
    std::vector<HostView> hbuf(blocks.size());
    std::vector<std::vector<double>> hAux(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      const int from = oldMaster[i], to = blocks[i].master;
      if (from == to || (from != rank_ && to != rank_))
        continue;
      const I3 e = blocks[i].advector().extent();
      const long len = static_cast<long>(e.x) * e.y * e.z;
      dbuf[i] = buffer(len);
      hbuf[i] = Kokkos::create_mirror_view(dbuf[i]);
      hAux[i].resize(4, 0.0);
      if (from == rank_) {
        Kokkos::deep_copy(dbuf[i], blocks[i].advector().colour());
        Kokkos::deep_copy(hbuf[i], dbuf[i]);
        blocks[i].serializeAux(hAux[i].data());
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(hbuf[i].data(), static_cast<int>(len), MPI_DOUBLE, to, tagOf(blocks[i].id, 2),
                  comm_, &reqs.back());
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(hAux[i].data(), 4, MPI_DOUBLE, to, tagOf(blocks[i].id, 3), comm_, &reqs.back());
      } else {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(hbuf[i].data(), static_cast<int>(len), MPI_DOUBLE, from, tagOf(blocks[i].id, 2),
                  comm_, &reqs.back());
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(hAux[i].data(), 4, MPI_DOUBLE, from, tagOf(blocks[i].id, 3), comm_, &reqs.back());
      }
      mBytes_ += (len + 4) * 8;
      mMsgs_ += 2;
    }
    if (!reqs.empty())
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    for (std::size_t i = 0; i < blocks.size(); ++i)
      if (oldMaster[i] != blocks[i].master && blocks[i].master == rank_) {
        Kokkos::deep_copy(dbuf[i], hbuf[i]);
        Kokkos::deep_copy(blocks[i].advector().colour(), dbuf[i]);
        blocks[i].deserializeAux(hAux[i].data());
      }
    Kokkos::fence();
#else
    (void)blocks;
    (void)oldMaster;
#endif
  }
  long migrateBytes() const { return mBytes_; }
  long migrateMessages() const { return mMsgs_; }

  /// Reusable device buffer of exactly `n` doubles, handed out in REQUEST ORDER — two live
  /// messages of the same size must not be the same View, so the pool is keyed by the request
  /// sequence (reset at the head of every transfer) and not by the size. The request sequence of a
  /// step is deterministic, so after the first few steps nothing is allocated at all; an
  /// allocation per message would otherwise dominate the device-vs-host packing comparison.
  SField buffer(long n) {
    const std::size_t k = static_cast<std::size_t>(bufSeq_++);
    if (bufPool_.size() <= k)
      bufPool_.resize(k + 1);
    if (bufPool_[k].extent(0) != static_cast<std::size_t>(n))
      bufPool_[k] = SField(Kokkos::view_alloc("vof::block::buf", Kokkos::WithoutInitializing),
                           static_cast<std::size_t>(n));
    return bufPool_[k];
  }

  static int ghostOf(const std::vector<VofBlock>& blocks) {
    for (const auto& b : blocks)
      if (b.mine())
        return b.advector().ghost();
    return 3;
  }

  // ------------------------------------------------------- gather (rung W0, HOST-staged)
  //
  // Kept as the reference implementation and as the `deviceStaging = false` arm of the W1
  // measurement: it mirrors the whole local patch to the host, packs with host loops, and mirrors
  // the block arrays back. The device path below produces BIT-IDENTICAL buffers (every step is a
  // copy of a double), which is what the W1 ctest gates.
  void gatherFaceVelHost(std::vector<VofBlock>& blocks, int ghost) {
    gBytes_ = 0;
    gMsgs_ = 0;
    auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lu_);
    auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lv_);
    auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lw_);

    std::vector<std::vector<double>> sendBuf(blocks.size());
    std::vector<HostView> hbu(blocks.size()), hbv(blocks.size()), hbw(blocks.size());
    struct Recv {
      std::size_t bi;
      int from;
      std::vector<double> buf;
    };
    std::vector<Recv> recvs;
#ifdef PECLET_FLOW_MPI
    std::vector<MPI_Request> reqs;
#endif

    // 1. post receives (master side) and pack+send (owner side); do the master's own cells locally.
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      VofBlock& b = blocks[bi];
      const VofBox eb = b.extended(ghost);
      vofBuildPieces(eb, gs_, per_, rankBox_, pieces_);
      const bool master = (b.master == rank_);
      if (master) {
        hbu[bi] = Kokkos::create_mirror_view(b.advector().faceU());
        hbv[bi] = Kokkos::create_mirror_view(b.advector().faceV());
        hbw[bi] = Kokkos::create_mirror_view(b.advector().faceW());
        zero(hbu[bi]);
        zero(hbv[bi]);
        zero(hbw[bi]);
      }
      // per-rank aggregated counts, in the canonical piece order
      std::map<int, long> counts;
      for (const auto& p : pieces_)
        counts[p.rank] += p.cells();
      if (master) {
        for (const auto& kv : counts) {
          if (kv.first == rank_)
            continue;
          recvs.push_back(Recv{bi, kv.first, std::vector<double>(3 * kv.second)});
        }
      }
      if (counts.count(rank_)) {
        if (master) {
          // local: straight copy, no buffer
          long off = 0;
          for (const auto& p : pieces_) {
            if (p.rank != rank_)
              continue;
            copyPatchToBlock(p, hu, hv, hw, hbu[bi], hbv[bi], hbw[bi], b, ghost);
            off += p.cells();
          }
          (void)off;
        } else {
          std::vector<double>& sb = sendBuf[bi];
          sb.resize(3 * counts[rank_]);
          long off = 0;
          for (const auto& p : pieces_) {
            if (p.rank != rank_)
              continue;
            packPatch(p, hu, hv, hw, sb, off);
          }
        }
      }
    }
#ifdef PECLET_FLOW_MPI
    if (size_ > 1) {
      requireComm();
      for (auto& r : recvs) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(r.buf.data(), static_cast<int>(r.buf.size()), MPI_DOUBLE, r.from,
                  tagOf(blocks[r.bi].id, 0), comm_, &reqs.back());
        gBytes_ += static_cast<long>(r.buf.size()) * 8;
        ++gMsgs_;
      }
      for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        if (sendBuf[bi].empty())
          continue;
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(sendBuf[bi].data(), static_cast<int>(sendBuf[bi].size()), MPI_DOUBLE,
                  blocks[bi].master, tagOf(blocks[bi].id, 0), comm_, &reqs.back());
        gBytes_ += static_cast<long>(sendBuf[bi].size()) * 8;
        ++gMsgs_;
      }
      if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
#endif
    // 2. unpack
    for (auto& r : recvs) {
      VofBlock& b = blocks[r.bi];
      vofBuildPieces(b.extended(ghost), gs_, per_, rankBox_, pieces_);
      long off = 0;
      for (const auto& p : pieces_) {
        if (p.rank != r.from)
          continue;
        unpackBlock(p, r.buf, off, hbu[r.bi], hbv[r.bi], hbw[r.bi], b, ghost);
      }
    }
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      if (blocks[bi].master != rank_)
        continue;
      Kokkos::deep_copy(blocks[bi].advector().faceU(), hbu[bi]);
      Kokkos::deep_copy(blocks[bi].advector().faceV(), hbv[bi]);
      Kokkos::deep_copy(blocks[bi].advector().faceW(), hbw[bi]);
    }
  }

  // ------------------------------------------------------ scatter (rung W0, HOST-staged)
  void scatterColourMaxHost(std::vector<VofBlock>& blocks, SField cLocal) {
    sBytes_ = 0;
    sMsgs_ = 0;
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cLocal);
    // UNPACK_MAX starts from an empty union: every inner cell not covered by a block is gas.
    const I3 e = patch_.e, n = patch_.n;
    const int g = patch_.g;
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          hc(L3(x + g, y + g, z + g, e)) = 0.0;

    struct Recv {
      std::size_t bi;
      int from;
      std::vector<double> buf;
    };
    std::vector<Recv> recvs;
    std::vector<std::vector<double>> sendBuf;
    std::vector<int> sendTo;
    std::vector<long> sendTag;
#ifdef PECLET_FLOW_MPI
    std::vector<MPI_Request> reqs;
#endif
    std::vector<HostView> hbc(blocks.size());

    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      VofBlock& b = blocks[bi];
      vofBuildPieces(b.box, gs_, per_, rankBox_, pieces_);
      const bool master = (b.master == rank_);
      if (master) {
        hbc[bi] = Kokkos::create_mirror_view(b.advector().colour());
        Kokkos::deep_copy(hbc[bi], b.advector().colour());
      }
      std::map<int, long> counts;
      for (const auto& p : pieces_)
        counts[p.rank] += p.cells();
      if (master) {
        for (const auto& kv : counts) {
          if (kv.first == rank_) {
            for (const auto& p : pieces_)
              if (p.rank == rank_)
                maxBlockIntoPatch(p, hbc[bi], b, hc);
            continue;
          }
          std::vector<double> sb(kv.second);
          long off = 0;
          for (const auto& p : pieces_)
            if (p.rank == kv.first)
              packBlockColour(p, hbc[bi], b, sb, off);
          sendBuf.push_back(std::move(sb));
          sendTo.push_back(kv.first);
          sendTag.push_back(b.id);
        }
      } else if (counts.count(rank_)) {
        recvs.push_back(Recv{bi, b.master, std::vector<double>(counts[rank_])});
      }
    }
#ifdef PECLET_FLOW_MPI
    if (size_ > 1) {
      requireComm();
      for (auto& r : recvs) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(r.buf.data(), static_cast<int>(r.buf.size()), MPI_DOUBLE, r.from,
                  tagOf(blocks[r.bi].id, 1), comm_, &reqs.back());
        sBytes_ += static_cast<long>(r.buf.size()) * 8;
        ++sMsgs_;
      }
      for (std::size_t k = 0; k < sendBuf.size(); ++k) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(sendBuf[k].data(), static_cast<int>(sendBuf[k].size()), MPI_DOUBLE, sendTo[k],
                  tagOf(sendTag[k], 1), comm_, &reqs.back());
        sBytes_ += static_cast<long>(sendBuf[k].size()) * 8;
        ++sMsgs_;
      }
      if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
#endif
    for (auto& r : recvs) {
      VofBlock& b = blocks[r.bi];
      vofBuildPieces(b.box, gs_, per_, rankBox_, pieces_);
      long off = 0;
      for (const auto& p : pieces_)
        if (p.rank == rank_)
          maxBufIntoPatch(p, r.buf, off, hc);
    }
    Kokkos::deep_copy(cLocal, hc);
  }

  // -------------------------------------------------------------------------------- table
  void syncTable(std::vector<VofBlock>& blocks) override {
#ifdef PECLET_FLOW_MPI
    if (size_ <= 1)
      return;
    requireComm();
    const int nb = static_cast<int>(blocks.size());
    std::vector<int> mine(6 * nb, 0), all(6 * nb, 0);
    for (int i = 0; i < nb; ++i)
      if (blocks[i].master == rank_)
        for (int d = 0; d < 3; ++d) {
          mine[6 * i + d] = blocks[i].box.lo[d];
          mine[6 * i + 3 + d] = blocks[i].box.hi[d];
        }
    // exactly one rank contributes each block's row, so a SUM is a broadcast that needs no
    // per-block communicator and no ordering assumption.
    MPI_Allreduce(mine.data(), all.data(), 6 * nb, MPI_INT, MPI_SUM, comm_);
    for (int i = 0; i < nb; ++i)
      for (int d = 0; d < 3; ++d) {
        blocks[i].box.lo[d] = all[6 * i + d];
        blocks[i].box.hi[d] = all[6 * i + 3 + d];
      }
#else
    (void)blocks;
#endif
  }

 private:
  using HostView = decltype(Kokkos::create_mirror_view(SField()));

  static void zero(HostView v) {
    for (std::size_t i = 0; i < v.extent(0); ++i)
      v(i) = 0.0;
  }
  static int tagOf(long id, int phase) { return static_cast<int>((id % 4096) * 4 + phase) + 4096; }

#ifdef PECLET_FLOW_MPI
  void requireComm() const {
    if (!hasComm_)
      throw std::runtime_error("peclet::flow::vof::VofBlockExchange: no communicator set");
  }
#endif

  /// local patch index of a global cell (the caller has checked ownership)
  long patchIdx(int gx, int gy, int gz) const {
    return L3(gx - patch_.o.x + patch_.g, gy - patch_.o.y + patch_.g, gz - patch_.o.z + patch_.g,
              patch_.e);
  }

  template <class HV>
  void packPatch(const VofPiece& p, const HV& hu, const HV& hv, const HV& hw,
                 std::vector<double>& buf, long& off) const {
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const long i = patchIdx(x, y, z);
          buf[3 * off + 0] = hu(i);
          buf[3 * off + 1] = hv(i);
          buf[3 * off + 2] = hw(i);
          ++off;
        }
  }

  template <class HV, class HB>
  void copyPatchToBlock(const VofPiece& p, const HV& hu, const HV& hv, const HV& hw, HB bu, HB bv,
                        HB bw, const VofBlock& b, int ghost) const {
    const I3 be = b.advector().extent();
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const long i = patchIdx(x, y, z);
          // block-local EXTENDED index: `loc` is the index inside the extended box, whose cell
          // (0,0,0) sits at the advector's (0,0,0) — i.e. `ghost` cells before the inner region.
          const int lx = p.loc[0] + (x - p.g.lo[0]);
          const int ly = p.loc[1] + (y - p.g.lo[1]);
          const int lz = p.loc[2] + (z - p.g.lo[2]);
          const long j = L3(lx, ly, lz, be);
          bu(j) = hu(i);
          bv(j) = hv(i);
          bw(j) = hw(i);
        }
    (void)ghost;
  }

  template <class HB>
  void unpackBlock(const VofPiece& p, const std::vector<double>& buf, long& off, HB bu, HB bv,
                   HB bw, const VofBlock& b, int ghost) const {
    const I3 be = b.advector().extent();
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const int lx = p.loc[0] + (x - p.g.lo[0]);
          const int ly = p.loc[1] + (y - p.g.lo[1]);
          const int lz = p.loc[2] + (z - p.g.lo[2]);
          const long j = L3(lx, ly, lz, be);
          bu(j) = buf[3 * off + 0];
          bv(j) = buf[3 * off + 1];
          bw(j) = buf[3 * off + 2];
          ++off;
        }
    (void)ghost;
  }

  /// block INNER colour at the piece's cells -> a send buffer (x-fastest inside the piece)
  template <class HB>
  void packBlockColour(const VofPiece& p, const HB& bc, const VofBlock& b, std::vector<double>& buf,
                       long& off) const {
    const I3 be = b.advector().extent();
    const int g = b.advector().ghost();
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const int lx = p.loc[0] + (x - p.g.lo[0]);
          const int ly = p.loc[1] + (y - p.g.lo[1]);
          const int lz = p.loc[2] + (z - p.g.lo[2]);
          buf[off++] = bc(L3(lx + g, ly + g, lz + g, be));
        }
  }

  template <class HB, class HC>
  void maxBlockIntoPatch(const VofPiece& p, const HB& bc, const VofBlock& b, HC& hc) const {
    const I3 be = b.advector().extent();
    const int g = b.advector().ghost();
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const int lx = p.loc[0] + (x - p.g.lo[0]);
          const int ly = p.loc[1] + (y - p.g.lo[1]);
          const int lz = p.loc[2] + (z - p.g.lo[2]);
          const double v = bc(L3(lx + g, ly + g, lz + g, be));
          double& d = hc(patchIdx(x, y, z));
          if (v > d)
            d = v;
        }
  }

  template <class HC>
  void maxBufIntoPatch(const VofPiece& p, const std::vector<double>& buf, long& off, HC& hc) const {
    for (int z = p.g.lo[2]; z < p.g.hi[2]; ++z)
      for (int y = p.g.lo[1]; y < p.g.hi[1]; ++y)
        for (int x = p.g.lo[0]; x < p.g.hi[0]; ++x) {
          const double v = buf[off++];
          double& d = hc(patchIdx(x, y, z));
          if (v > d)
            d = v;
        }
  }

  I3 gs_{0, 0, 0};
  std::array<bool, 3> per_{true, true, true};
  std::vector<VofBox> rankBox_;
  int rank_ = 0, size_ = 1;
  Patch patch_;
  SField lu_, lv_, lw_;
  mutable std::vector<VofPiece> pieces_;
  std::vector<SField> bufPool_;
  long bufSeq_ = 0;
  long gBytes_ = 0, sBytes_ = 0, gMsgs_ = 0, sMsgs_ = 0, mBytes_ = 0, mMsgs_ = 0;
#ifdef PECLET_FLOW_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool hasComm_ = false;
#endif
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_EXCHANGE_HPP
