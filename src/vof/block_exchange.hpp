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

  // -------------------------------------------------------------------------------- gather
  void gatherFaceVel(std::vector<VofBlock>& blocks, int ghost) override {
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

  // -------------------------------------------------------------------------------- scatter
  void scatterColourMax(std::vector<VofBlock>& blocks, SField cLocal) override {
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
  static int tagOf(long id, int phase) { return static_cast<int>((id % 8192) * 2 + phase) + 4096; }

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
  long gBytes_ = 0, sBytes_ = 0, gMsgs_ = 0, sMsgs_ = 0;
#ifdef PECLET_FLOW_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool hasComm_ = false;
#endif
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_EXCHANGE_HPP
