// flow — the sharp viscous stress at a VoF interface (src/vof/sharp_viscous.hpp,
// doc/vof_sharp_viscous_stress.md §10 WO-2):
//
//   (a) 3-D extrusion of the 2-D harness (n_z = 0, z-independent fields, exact planes injected):
//       tau_hat, D_nn, the cell C, both midsection fractions, both face fractions and both link
//       fractions of every node with a mixed cell equal mac2d (TAU_RULE 'mixed-avg') to 1e-12
//       relative; fit fallbacks 0. Reference: tests/kokkos/sharp_viscous_mac2d.txt
//       (tests/study/sharp_viscous/wo2_reference.py);
//   (b) exact tau and D_nn for kinked-linear fields (traceless L_lo, [u] = 0, [P sigma n] = 0) at
//       the 9 orientations (beta, gamma in {0, 22.5, 45} deg), offsets 0 and 0.3, mu ratio 1e-2 /
//       1e2 / 1e-3 / 1e3, isotropic and h ratio 1 : 1.5 : 2: <= 1e-11; fallbacks 0;
//   (c) virtual planes (§5.7 Q-B): a 1-cell film normal to each axis and an ABAB stack: every edge
//       on a film face is owned at distance 0, and (m, alpha) follows the low-face rule;
//   (d) a periodic seam inside the band (45 deg slab, offset-order ties): classes, fallbacks and
//       tau/D_nn are bitwise invariant under a circular shift of the block by (5, 3, 2) cells;
//   (e) WO-3, the implicit band (sphere of tangent planes, mu ratio 1e-2 / 1e3, isotropic and
//       1 : 1.5 : 2): every band-edge entry = -mu(F_c(E)) w_a of a host oracle (owner + PLIC face
//       fraction), every non-band entry and every cell/diagonal of a row without a band edge
//       bitwise 'averaged' (VarFaceProps), per-component symmetry bitwise, M-matrix;
//   (f) WO-4, the explicit kernel against a host transcription (sample arrays, then the
//       divergence; random u, tau_hat, D_nn, row scale): <= 1e-13 relative;
//   (g) WO-4, the full operator (implicit band + band transpose + tau-terms + midsection - grad p,
//       p the exact cell-average pressure) on exact face averages of kinked-linear Couette and
//       extension at the 9 orientations, offsets 0 / 0.3, 4 ratios, both metrics: residual
//       <= 1e-11 of the stress scale;
//   (h) WO-9.2 (§5.8), the owner-set mean: a vertex-centred sphere is reflection self-symmetric
//       (F, theta, T over every band edge) to 1e-13 of the scale, with ties;
//   (i) WO-9.2: a tie-free case is bitwise the f01a251 single-owner operator (host digest).
// The host oracles of (e) and (f) transcribe the owner set and its mean (§5.8) independently.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cut_cell_ibm.hpp"
#include "full_stress.hpp"
#include "vof/sharp_viscous.hpp"

#ifndef SHARP_VISCOUS_REF
#define SHARP_VISCOUS_REF "sharp_viscous_mac2d.txt"
#endif

namespace {
int failures = 0;
#define CHECK(cond, ...)                                                              \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed: %s at %s:%d  ", #cond, __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                                              \
      std::fprintf(stderr, "\n");                                                     \
      ++failures;                                                                     \
    }                                                                                 \
  } while (0)

using namespace peclet::flow;
namespace sv = peclet::flow::vof;
using DV = Kokkos::View<double*, CCMem>;
constexpr double kPi = 3.14159265358979323846;
constexpr int kBig = 1 << 28;

DV toDev(const std::vector<double>& h, const char* name = "t") {
  DV d(name, h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i)
    m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}
std::vector<double> toHost(DV d) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  return std::vector<double>(m.data(), m.data() + m.extent(0));
}
std::vector<int> toHost(sv::SvByte d) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  std::vector<int> o(m.extent(0));
  for (std::size_t i = 0; i < o.size(); ++i)
    o[i] = m(i);
  return o;
}

// Solver block (ghost 2) and colour block (ghost 3) of n inner cells.
struct Blk {
  int n[3], G = 2, e[3], e3[3];
  long st[3], st3[3], cells, cells3;
  Blk(int nx, int ny, int nz) : n{nx, ny, nz} {
    for (int a = 0; a < 3; ++a) {
      e[a] = n[a] + 2 * G;
      e3[a] = n[a] + 2 * G + 2;
    }
    st[0] = 1;
    st[1] = e[0];
    st[2] = (long)e[0] * e[1];
    st3[0] = 1;
    st3[1] = e3[0];
    st3[2] = (long)e3[0] * e3[1];
    cells = st[2] * e[2];
    cells3 = st3[2] * e3[2];
  }
  long id(int x, int y, int z) const { return x + y * st[1] + z * st[2]; }
  // colour-block index of SOLVER coordinates (x, y, z), x in [-1, e]
  long id3(int x, int y, int z) const { return (x + 1) + (y + 1) * st3[1] + (z + 1) * st3[2]; }
};

// The device side of one configuration.
struct Geo {
  DV col, pm[3], pa, tau[3], dnn, v[3];
  sv::SvByte cls, fb;
  sv::SharpGeom g;
};

// Colour + planes on the colour block (host vectors), class pass, fit over the inner cells.
Geo makeGeo(const Blk& b, const std::vector<double>& col, const std::vector<double> (&pm)[3],
            const std::vector<double>& pa, const std::vector<double> (&v)[3], const int dlo[3],
            const int dhi[3], double eps, double muA, double muB, const double ell[3]) {
  Geo o;
  o.col = toDev(col, "col");
  for (int a = 0; a < 3; ++a) {
    o.pm[a] = toDev(pm[a], "pm");
    o.v[a] = toDev(v[a], "v");
    o.tau[a] = DV("tau", b.cells);
  }
  o.pa = toDev(pa, "pa");
  o.dnn = DV("dnn", b.cells);
  o.cls = sv::SvByte("cls", b.cells);
  o.fb = sv::SvByte("fb", b.cells);
  sv::SharpGeom& g = o.g;
  for (int a = 0; a < 3; ++a) {
    g.e[a] = b.e[a];
    g.st[a] = b.st[a];
    g.st3[a] = b.st3[a];
    g.dlo[a] = dlo[a];
    g.dhi[a] = dhi[a];
    g.ell[a] = ell[a];
    g.gm.h[a] = ell[a];
    g.pm[a] = o.pm[a];
  }
  g.d = 1;
  g.col = o.col;
  g.pa = o.pa;
  g.cls = o.cls;
  g.eps = eps;
  g.p0 = muB;
  g.p1 = muA - muB;
  g.muA = muA;
  g.muB = muB;
  g.lowA = muA < muB;
  g.mulo = g.lowA ? muA : muB;
  g.r = g.lowA ? muA / muB : muB / muA;
  sv::sharpClassify(g, o.cls);
  const CCConst vv[3] = {o.v[0], o.v[1], o.v[2]};
  sv::sharpFit(g, vv, o.tau[0], o.tau[1], o.tau[2], o.dnn, o.fb, C3{b.G, b.G, b.G},
               C3{b.e[0] - b.G, b.e[1] - b.G, b.e[2] - b.G});
  Kokkos::fence();
  return o;
}

// Edge queries on the device (§5.8): n_E (0 = not a band edge), the largest member distance, and
// the owner-set means F_row(c), F_row(a), L_ac, L_ca.
struct EdgeQ {
  int k[3], a, c;
};
std::vector<std::array<double, 6>> queryEdges(const Geo& o, const std::vector<EdgeQ>& qs) {
  const int nq = (int)qs.size();
  Kokkos::View<int*, CCMem> qd("q", 5 * nq);
  auto qh = Kokkos::create_mirror_view(qd);
  for (int t = 0; t < nq; ++t) {
    for (int s = 0; s < 3; ++s)
      qh(5 * t + s) = qs[t].k[s];
    qh(5 * t + 3) = qs[t].a;
    qh(5 * t + 4) = qs[t].c;
  }
  Kokkos::deep_copy(qd, qh);
  DV out("out", 6 * nq);
  const sv::SharpGeom g = o.g;
  Kokkos::parallel_for(
      "test::edges", Kokkos::RangePolicy<CCExec>(0, nq), KOKKOS_LAMBDA(int t) {
        const int k[3] = {qd(5 * t), qd(5 * t + 1), qd(5 * t + 2)};
        const int a = qd(5 * t + 3), c = qd(5 * t + 4), b = 3 - a - c;
        double dmax = 0.0;
        const double n =
            g.forEachOwner(k, a, c, [&](const int ow[3], const double m[3], double al, double) {
              double xi[3];
              xi[a] = k[a] - ow[a];
              xi[c] = k[c] - ow[c];
              xi[b] = 0.5;
              dmax =
                  Kokkos::fmax(dmax, Kokkos::fabs(m[0] * xi[0] + m[1] * xi[1] + m[2] * xi[2] - al) /
                                         Kokkos::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]));
            });
        out(6 * t) = n;
        if (n == 0.0)
          return;
        out(6 * t + 1) = dmax;
        double f = 0.0, tac = 0.0, tca = 0.0;
        g.edgeFaceFraction(k, a, c, c, f);
        out(6 * t + 2) = f;
        g.edgeFaceFraction(k, a, c, a, f);
        out(6 * t + 3) = f;
        g.edgeLinkFractions(k, a, c, tac, tca);
        out(6 * t + 4) = tac;
        out(6 * t + 5) = tca;
      });
  auto oh = toHost(out);
  std::vector<std::array<double, 6>> r(nq);
  for (int t = 0; t < nq; ++t)
    for (int s = 0; s < 6; ++s)
      r[t][s] = oh[6 * t + s];
  return r;
}

// Cell queries: class, plane (m, alpha), midsection fractions c = 0, 1, 2.
std::vector<std::array<double, 8>> queryCells(const Geo& o,
                                              const std::vector<std::array<int, 3>>& qs) {
  const int nq = (int)qs.size();
  Kokkos::View<int*, CCMem> qd("q", 3 * nq);
  auto qh = Kokkos::create_mirror_view(qd);
  for (int t = 0; t < nq; ++t)
    for (int s = 0; s < 3; ++s)
      qh(3 * t + s) = qs[t][s];
  Kokkos::deep_copy(qd, qh);
  DV out("out", 8 * nq);
  const sv::SharpGeom g = o.g;
  Kokkos::parallel_for(
      "test::cells", Kokkos::RangePolicy<CCExec>(0, nq), KOKKOS_LAMBDA(int t) {
        const int q[3] = {qd(3 * t), qd(3 * t + 1), qd(3 * t + 2)};
        const int k = g.cellClass(q);
        out(8 * t) = k;
        if (k == sv::kSvNone)
          return;
        double m[3], al;
        g.plane(q, k, m, al);
        out(8 * t + 1) = m[0];
        out(8 * t + 2) = m[1];
        out(8 * t + 3) = m[2];
        out(8 * t + 4) = al;
        for (int c = 0; c < 3; ++c)
          out(8 * t + 5 + c) = sv::plicMidsectionFraction(m, al, c);
      });
  auto oh = toHost(out);
  std::vector<std::array<double, 8>> r(nq);
  for (int t = 0; t < nq; ++t)
    for (int s = 0; s < 8; ++s)
      r[t][s] = oh[8 * t + s];
  return r;
}

// ---------------------------------------------------------------------------------------------
// (a) the 2-D harness, extruded
void testExtrusion() {
  std::ifstream in(SHARP_VISCOUS_REF);
  CHECK(in.good(), "cannot open %s", SHARP_VISCOUS_REF);
  if (!in.good())
    return;
  std::string tag;
  int ncases = 0;
  in >> tag >> ncases;
  double wTau = 0, wDnn = 0, wFrac = 0, wC = 0;
  long nCells = 0, nNodes = 0, nFb = 0, nNoOwner = 0, nClass = 0;
  for (int cs = 0; cs < ncases; ++cs) {
    int N, gen;
    double n0, n1, t0, t1, c0, c1, s, muA, muB;
    in >> tag >> N >> n0 >> n1 >> t0 >> t1 >> c0 >> c1 >> s >> muA >> muB >> gen;
    const double h = 1.0 / N;
    struct Cell {
      int i, j, fit;
      double C, cbu, cbv, tau, dnn;
    };
    struct Node {
      int i, j;
      double fu, fv, thy, thx;
    };
    int nc;
    in >> tag >> nc;
    std::vector<Cell> cells(nc);
    for (auto& c : cells)
      in >> c.i >> c.j >> c.C >> c.cbu >> c.cbv >> c.fit >> c.tau >> c.dnn;
    int nn;
    in >> tag >> nn;
    std::vector<Node> nodes(nn);
    for (auto& d : nodes)
      in >> d.i >> d.j >> d.fu >> d.fv >> d.thy >> d.thx;
    Blk b(N, N, 3);
    const int G = b.G;
    std::vector<double> vel[3] = {std::vector<double>(b.cells, 0.0),
                                  std::vector<double>(b.cells, 0.0),
                                  std::vector<double>(b.cells, 0.0)};
    for (int comp = 0; comp < 2; ++comp) {
      int nv;
      in >> tag >> nv;
      for (int t = 0; t < nv; ++t) {
        int i, j;
        double val;
        in >> i >> j >> val;
        for (int z = 0; z < b.e[2]; ++z)
          vel[comp][b.id(i + G, j + G, z)] = val / h;  // index velocity (t_ref = 1, h' = 1)
      }
    }
    // Colour block: mixed cells from the file, pure by the sign of phi at the centre.
    auto phi = [&](double x, double y) { return n0 * (x - c0) + n1 * (y - c1) - s; };
    std::vector<double> col(b.cells3),
        pm[3] = {std::vector<double>(b.cells3, 0.0), std::vector<double>(b.cells3, 0.0),
                 std::vector<double>(b.cells3, 0.0)},
        pa(b.cells3, 0.0);
    for (int z = -1; z <= b.e[2]; ++z)
      for (int y = -1; y <= b.e[1]; ++y)
        for (int x = -1; x <= b.e[0]; ++x)
          col[b.id3(x, y, z)] = phi((x - G + 0.5) * h, (y - G + 0.5) * h) < 0 ? 1.0 : 0.0;
    for (const auto& c : cells)
      for (int z = -1; z <= b.e[2]; ++z) {
        const long j3 = b.id3(c.i + G, c.j + G, z);
        col[j3] = c.C;
        pm[0][j3] = n0;
        pm[1][j3] = n1;
        pm[2][j3] = 0.0;
        pa[j3] = (s + n0 * c0 + n1 * c1) / h - (n0 * c.i + n1 * c.j);
      }
    const int dlo[3] = {G, G, -kBig}, dhi[3] = {N + G, N + G, kBig};
    const double ell[3] = {1.0, 1.0, 1.0};
    Geo o = makeGeo(b, col, pm, pa, vel, dlo, dhi, 0.0, muA, muB, ell);
    auto tau0 = toHost(o.tau[0]), tau1 = toHost(o.tau[1]), tau2 = toHost(o.tau[2]),
         dnn = toHost(o.dnn);
    auto fb = toHost(o.fb);
    const double mulo = std::min(muA, muB);
    double tmax = 0.0;
    for (const auto& c : cells)
      tmax = std::max(tmax, std::fabs(c.tau));
    std::vector<std::array<int, 3>> cq;
    for (const auto& c : cells)
      cq.push_back({c.i + G, c.j + G, G + 1});
    auto cr = queryCells(o, cq);
    for (std::size_t t = 0; t < cells.size(); ++t) {
      const auto& c = cells[t];
      const bool in = c.i >= 0 && c.i < N && c.j >= 0 && c.j < N;
      CHECK((cr[t][0] == sv::kSvPlane) == in, "class of (%d,%d) = %g", c.i, c.j, cr[t][0]);
      if (!in)
        continue;
      nClass += (cr[t][0] == sv::kSvPlane);
      const double mm[3] = {cr[t][1], cr[t][2], cr[t][3]};
      wC = std::max(wC, std::fabs(sv::plicVolume(mm[0], mm[1], mm[2], cr[t][4]) - c.C));
      wFrac = std::max({wFrac, std::fabs(cr[t][5] - c.cbu), std::fabs(cr[t][6] - c.cbv)});
      if (!c.fit)
        continue;
      ++nCells;
      const long i = b.id(c.i + G, c.j + G, G + 1);
      nFb += fb[i];
      const double et = std::max(
          {std::fabs(tau0[i] - c.tau * t0), std::fabs(tau1[i] - c.tau * t1), std::fabs(tau2[i])});
      wTau = std::max(wTau, et / tmax);
      wDnn = std::max(wDnn, std::fabs(dnn[i] - c.dnn) / (tmax / mulo));
    }
    std::vector<EdgeQ> eq;
    for (const auto& d : nodes)
      eq.push_back(EdgeQ{{d.i + G, d.j + G, G + 1}, 0, 1});
    auto er = queryEdges(o, eq);
    for (std::size_t t = 0; t < nodes.size(); ++t) {
      const auto& d = nodes[t];
      if (er[t][0] < 1.0) {
        ++nNoOwner;
        continue;
      }
      ++nNodes;
      // E_xy: row c = y is the v row (F_y = fv), row a = x the u row (F_x = fu); L_ac = d_x u_y
      // (thx), L_ca = d_y u_x (thy).
      wFrac = std::max({wFrac, std::fabs(er[t][2] - d.fv), std::fabs(er[t][3] - d.fu),
                        std::fabs(er[t][4] - d.thx), std::fabs(er[t][5] - d.thy)});
    }
  }
  std::printf(
      "(a) extrusion: %d cases, %ld fits, %ld band nodes: tau %.2e, D_nn %.2e, fractions "
      "%.2e, C %.2e; fallbacks %ld, ownerless %ld\n",
      ncases, nCells, nNodes, wTau, wDnn, wFrac, wC, nFb, nNoOwner);
  CHECK(nCells > 100 && nNodes > 100 && nClass >= nCells, "coverage");
  CHECK(wTau <= 1e-12 && wDnn <= 1e-12, "tau/D_nn vs mac2d");
  CHECK(wFrac <= 1e-12 && wC <= 1e-12, "fractions vs mac2d");
  CHECK(nFb == 0 && nNoOwner == 0, "fallbacks / ownerless band nodes");
}

// ---------------------------------------------------------------------------------------------
// Exact kinked-linear fields.  Internal coordinates X_a = ell_a * (index coordinate).
struct Kinked {
  double n[3], xp[3], Llo[9], j[3], muA, muB;
  bool lowA;
  double phi(const double X[3]) const {
    return n[0] * (X[0] - xp[0]) + n[1] * (X[1] - xp[1]) + n[2] * (X[2] - xp[2]);
  }
  bool hi(double ph) const { return lowA ? (ph >= 0.0) : (ph < 0.0); }  // A: phi < 0
};

// int over [0,L1]x[0,L2] of max(0, f0 + g1 s + g2 t), exact (clipped polygon, centroid rule).
double posInt(double f0, double g1, double g2, double L1, double L2) {
  double P[4][2] = {{0, 0}, {L1, 0}, {L1, L2}, {0, L2}};
  std::vector<std::array<double, 2>> poly;
  for (int k = 0; k < 4; ++k) {
    const double* A = P[k];
    const double* B = P[(k + 1) % 4];
    const double fa = f0 + g1 * A[0] + g2 * A[1], fb = f0 + g1 * B[0] + g2 * B[1];
    if (fa > 0)
      poly.push_back({A[0], A[1]});
    if ((fa > 0) != (fb > 0)) {
      const double w = fa / (fa - fb);
      poly.push_back({A[0] + w * (B[0] - A[0]), A[1] + w * (B[1] - A[1])});
    }
  }
  if (poly.size() < 3)
    return 0.0;
  double ar = 0, cx = 0, cy = 0;
  for (std::size_t k = 0; k < poly.size(); ++k) {
    const auto& p = poly[k];
    const auto& q = poly[(k + 1) % poly.size()];
    const double cr = p[0] * q[1] - q[0] * p[1];
    ar += cr;
    cx += (p[0] + q[0]) * cr;
    cy += (p[1] + q[1]) * cr;
  }
  ar *= 0.5;
  if (std::fabs(ar) < 1e-300)
    return 0.0;
  cx /= (6 * ar);
  cy /= (6 * ar);
  return std::fabs(ar) * (f0 + g1 * cx + g2 * cy);
}

// Face average of u_q over the -c face of cell (x, y, z) (index coordinates of the cell corner).
double faceAvg(const Kinked& K, const double ell[3], int c, int x, int y, int z, int q) {
  const int I[3] = {x, y, z};
  const int p1 = (c == 0) ? 1 : 0, p2 = (c == 2) ? 1 : 2;
  const double L1 = ell[p1], L2 = ell[p2];
  double X0[3];
  for (int a = 0; a < 3; ++a)
    X0[a] = ell[a] * I[a];
  double cen[3] = {X0[0], X0[1], X0[2]};
  cen[p1] += 0.5 * L1;
  cen[p2] += 0.5 * L2;
  double lin = 0.0;
  for (int p = 0; p < 3; ++p)
    lin += K.Llo[3 * p + q] * (cen[p] - K.xp[p]);
  const double f0 = K.phi(X0), g1 = K.n[p1], g2 = K.n[p2];
  const double A = L1 * L2;
  const double pos = posInt(f0, g1, g2, L1, L2) / A;
  const double all = f0 + 0.5 * (g1 * L1 + g2 * L2);  // mean of phi over the face
  // the mean of phi over the hi side: phi >= 0 (lowA) -> mean max(phi, 0); phi < 0 -> mean
  // min(phi, 0) = mean(phi) - mean max(phi, 0)
  const double kink = K.lowA ? pos : (all - pos);
  return lin + K.j[q] * kink;
}

struct KCase {
  Blk b;
  Kinked K;
  double ell[3];
  std::vector<double> col, pm[3], pa, v[3];
};

// Exact planes + colour on the colour block, exact face averages on the solver block.
void buildKinked(KCase& kc) {
  const Blk& b = kc.b;
  const Kinked& K = kc.K;
  kc.col.assign(b.cells3, 0.0);
  kc.pa.assign(b.cells3, 0.0);
  for (int a = 0; a < 3; ++a) {
    kc.pm[a].assign(b.cells3, 0.0);
    kc.v[a].assign(b.cells, 0.0);
  }
  for (int z = -1; z <= b.e[2]; ++z)
    for (int y = -1; y <= b.e[1]; ++y)
      for (int x = -1; x <= b.e[0]; ++x) {
        const long j = b.id3(x, y, z);
        double m[3];
        for (int a = 0; a < 3; ++a)
          m[a] = K.n[a] * kc.ell[a];
        const double al = K.n[0] * K.xp[0] + K.n[1] * K.xp[1] + K.n[2] * K.xp[2] -
                          (m[0] * x + m[1] * y + m[2] * z);
        const double C = sv::plicVolume(m[0], m[1], m[2], al);
        kc.col[j] = C;
        if (C > 0.0 && C < 1.0) {
          for (int a = 0; a < 3; ++a)
            kc.pm[a][j] = m[a];
          kc.pa[j] = al;
        }
      }
  for (int z = 0; z < b.e[2]; ++z)
    for (int y = 0; y < b.e[1]; ++y)
      for (int x = 0; x < b.e[0]; ++x)
        for (int c = 0; c < 3; ++c)
          kc.v[c][b.id(x, y, z)] = faceAvg(K, kc.ell, c, x, y, z, c) / kc.ell[c];
}

// L_lo traceless and its kink: L_hi = L_lo + n (x) j, j = (r - 1) P S_lo n.
void setKink(Kinked& K, double ratio) {
  K.muA = ratio;
  K.muB = 1.0;
  K.lowA = K.muA < K.muB;
  const double r = K.lowA ? K.muA / K.muB : K.muB / K.muA;
  double sn[3], nsn = 0;
  for (int p = 0; p < 3; ++p) {
    sn[p] = 0;
    for (int q = 0; q < 3; ++q)
      sn[p] += (K.Llo[3 * p + q] + K.Llo[3 * q + p]) * K.n[q];
    nsn += K.n[p] * sn[p];
  }
  for (int p = 0; p < 3; ++p)
    K.j[p] = (r - 1.0) * (sn[p] - K.n[p] * nsn);
}

void testKinked() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const double angs[3] = {0.0, 22.5, 45.0};
  double wTau = 0, wDnn = 0;
  long nFit = 0, nFb = 0;
  for (int an = 0; an < 2; ++an)
    for (int ib = 0; ib < 3; ++ib)
      for (int ig = 0; ig < 3; ++ig)
        for (double off : {0.0, 0.3})
          for (double ratio : {1e-2, 1e2, 1e-3, 1e3}) {
            KCase kc{Blk(8, 8, 8), {}, {1.0, 1.0, 1.0}, {}, {}, {}, {}};
            if (an == 1) {
              kc.ell[1] = 1.5;
              kc.ell[2] = 2.0;
            }
            const double be = angs[ib] * kPi / 180, ga = angs[ig] * kPi / 180;
            Kinked& K = kc.K;
            K.n[0] = std::cos(be) * std::cos(ga);
            K.n[1] = std::cos(be) * std::sin(ga);
            K.n[2] = std::sin(be);
            for (int a = 0; a < 3; ++a)  // through a cell corner + off along n (index units)
              K.xp[a] = kc.ell[a] * 6.0 + off * K.n[a];
            double tr = 0;
            for (int s = 0; s < 9; ++s)
              K.Llo[s] = U(rng);
            for (int p = 0; p < 3; ++p)
              tr += K.Llo[4 * p];
            for (int p = 0; p < 3; ++p)
              K.Llo[4 * p] -= tr / 3.0;
            setKink(K, ratio);
            buildKinked(kc);
            const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
            Geo o = makeGeo(kc.b, kc.col, kc.pm, kc.pa, kc.v, dlo, dhi, 0.0, K.muA, K.muB, kc.ell);
            auto t0 = toHost(o.tau[0]), t1 = toHost(o.tau[1]), t2 = toHost(o.tau[2]),
                 dn = toHost(o.dnn);
            auto fb = toHost(o.fb), cl = toHost(o.cls);
            // exact: tau = mu_lo P S_lo n, D_nn = n.L n
            double sn[3], nsn = 0, dex = 0, tex[3], lmax = 0;
            for (int p = 0; p < 3; ++p) {
              sn[p] = 0;
              for (int q = 0; q < 3; ++q) {
                sn[p] += (K.Llo[3 * p + q] + K.Llo[3 * q + p]) * K.n[q];
                dex += K.n[p] * K.Llo[3 * p + q] * K.n[q];
                lmax = std::max(lmax, std::fabs(K.Llo[3 * p + q]));
              }
              nsn += K.n[p] * sn[p];
            }
            const double mulo = std::min(K.muA, K.muB);
            for (int p = 0; p < 3; ++p)
              tex[p] = mulo * (sn[p] - K.n[p] * nsn);
            const Blk& b = kc.b;
            for (int z = b.G; z < b.e[2] - b.G; ++z)
              for (int y = b.G; y < b.e[1] - b.G; ++y)
                for (int x = b.G; x < b.e[0] - b.G; ++x) {
                  const long i = b.id(x, y, z);
                  if (cl[i] == sv::kSvNone)
                    continue;
                  ++nFit;
                  nFb += fb[i];
                  const double e = std::max({std::fabs(t0[i] - tex[0]), std::fabs(t1[i] - tex[1]),
                                             std::fabs(t2[i] - tex[2])});
                  wTau = std::max(wTau, e / (mulo * lmax));
                  wDnn = std::max(wDnn, std::fabs(dn[i] - dex) / lmax);
                }
          }
  std::printf("(b) kinked-linear: %ld fits, tau %.2e, D_nn %.2e (relative), fallbacks %ld\n", nFit,
              wTau, wDnn, nFb);
  CHECK(nFit > 1000, "coverage");
  CHECK(wTau <= 1e-11 && wDnn <= 1e-11, "kinked-linear exactness");
  CHECK(nFb == 0, "fallbacks");
}

// ---------------------------------------------------------------------------------------------
// (c) virtual planes: films and the ABAB stack, periodic box, colour only.
void testVirtual() {
  long nEdge = 0, nBad = 0, nPlaneBad = 0;
  double wDist = 0;
  for (int ax = 0; ax < 3; ++ax)
    for (int stack = 0; stack < 2; ++stack) {
      Blk b(6, 6, 6);
      std::vector<double> col(b.cells3, 0.0), pm[3], pa(b.cells3, 0.0), v[3];
      for (int a = 0; a < 3; ++a) {
        pm[a].assign(b.cells3, 0.0);
        v[a].assign(b.cells, 0.0);
      }
      // global inner coordinate along ax: X = x - G; the film is X == 2 (A, C = 1) in B; the
      // stack alternates A, B starting with A at X = 0 (6 cells: periodic-consistent).
      auto phase = [&](int X) {
        const int w = ((X % 6) + 6) % 6;
        return stack ? (w % 2 == 0) : (w == 2);
      };
      for (int z = -1; z <= b.e[2]; ++z)
        for (int y = -1; y <= b.e[1]; ++y)
          for (int x = -1; x <= b.e[0]; ++x) {
            const int I[3] = {x, y, z};
            col[b.id3(x, y, z)] = phase(I[ax] - b.G) ? 1.0 : 0.0;
          }
      const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
      const double ell[3] = {1.0, 1.0, 1.0};
      Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-8, 0.01, 1.0, ell);
      // planes: every cell with an opposite face-neighbour along ax is virtual on ax, low face
      // first
      std::vector<std::array<int, 3>> cq;
      for (int X = 0; X < 6; ++X)
        cq.push_back({ax == 0 ? X + b.G : 3, ax == 1 ? X + b.G : 3, ax == 2 ? X + b.G : 3});
      auto cr = queryCells(o, cq);
      for (int X = 0; X < 6; ++X) {
        const bool me = phase(X), lo = phase(X - 1), hi = phase(X + 1);
        int side = -1;
        if (lo != me)
          side = 0;
        else if (hi != me)
          side = 1;
        const int kExp = side < 0 ? sv::kSvNone : sv::kSvVirtual0 + 2 * ax + side;
        bool ok = (int)cr[X][0] == kExp;
        if (ok && side >= 0) {
          // sigma = +1 iff C = 1 lies at xi_ax < f, f = side
          const double sg = (side == 0) ? (lo ? 1.0 : -1.0) : (me ? 1.0 : -1.0);
          for (int a = 0; a < 3; ++a)
            ok = ok && cr[X][1 + a] == (a == ax ? sg : 0.0);
          ok = ok && cr[X][4] == sg * side;
        }
        nPlaneBad += ok ? 0 : 1;
      }
      // every edge on a phase face: (k_ax = face) for the two orientations containing ax
      std::vector<EdgeQ> eq;
      for (int X = 0; X <= 6; ++X) {
        if (phase(X) == phase(X - 1))
          continue;  // not a phase face
        for (int c = 0; c < 3; ++c) {
          if (c == ax)
            continue;
          for (int p = 3; p <= 4; ++p)
            for (int s = 3; s <= 4; ++s) {
              EdgeQ q;
              const int bb = 3 - ax - c;
              q.k[ax] = X + b.G;
              q.k[c] = p;
              q.k[bb] = s;
              q.a = std::min(ax, c);
              q.c = std::max(ax, c);
              eq.push_back(q);
            }
        }
      }
      auto er = queryEdges(o, eq);
      for (const auto& r : er) {
        ++nEdge;
        if (r[0] < 1.0)
          ++nBad;
        else
          wDist = std::max(wDist, r[1]);
      }
    }
  std::printf(
      "(c) virtual planes: %ld face edges, %ld ownerless, max owner distance %.2e, %ld bad "
      "planes\n",
      nEdge, nBad, wDist, nPlaneBad);
  CHECK(nEdge > 0 && nBad == 0 && wDist == 0.0, "film-face edges owned at distance 0");
  CHECK(nPlaneBad == 0, "virtual planes follow the low-face rule");
}

// ---------------------------------------------------------------------------------------------
// (d) a periodic seam in the band: circular-shift invariance (bitwise).
void testSeam() {
  const int n = 8, sh[3] = {5, 3, 2};
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<double> ug[3];  // global periodic face values, n^3
  for (int c = 0; c < 3; ++c) {
    ug[c].resize(n * n * n);
    for (auto& x : ug[c])
      x = U(rng);
  }
  auto wrap = [&](int i) { return ((i % n) + n) % n; };
  // the slab 2.3 <= (X + Y) / sqrt(2) < 5.1 (45 deg; ties between the two planes' owners)
  const double s2 = std::sqrt(0.5);
  auto run = [&](int shift, std::vector<double>& tauOut, std::vector<int>& clsOut) {
    Blk b(n, n, n);
    std::vector<double> col(b.cells3), pm[3], pa(b.cells3, 0.0), v[3];
    for (int a = 0; a < 3; ++a) {
      pm[a].assign(b.cells3, 0.0);
      v[a].assign(b.cells, 0.0);
    }
    for (int z = -1; z <= b.e[2]; ++z)
      for (int y = -1; y <= b.e[1]; ++y)
        for (int x = -1; x <= b.e[0]; ++x) {
          // global cell of this local cell under the shift
          const int X = wrap(x - b.G + shift * sh[0]), Y = wrap(y - b.G + shift * sh[1]);
          const int W = (X + Y) % n;  // the slab is a function of (X + Y) mod n on the torus
          // two planes: phi1 = s2 (X + Y) - 2.3 (A below), phi2 = 5.1 - s2 (X + Y) (A above)
          // -- A between them, on the periodic torus of period n in X + Y: use the cell-local
          // planes of whichever plane cuts the cell
          const long j = b.id3(x, y, z);
          const double m[3] = {s2, s2, 0.0};
          double C1 = sv::plicVolume(-m[0], -m[1], 0.0, -(2.3 - s2 * W));  // side phi1 > 0
          double C2 = sv::plicVolume(m[0], m[1], 0.0, 5.1 - s2 * W);       // side phi2 > 0
          const double C = C1 + C2 - 1.0 < 0.0 ? 0.0 : C1 + C2 - 1.0;
          col[j] = C;
          if (C > 0.0 && C < 1.0) {
            const bool first = C1 < 1.0;
            for (int a = 0; a < 2; ++a)
              pm[a][j] = first ? -m[a] : m[a];
            pa[j] = first ? -(2.3 - s2 * W) : 5.1 - s2 * W;
          }
        }
    for (int z = 0; z < b.e[2]; ++z)
      for (int y = 0; y < b.e[1]; ++y)
        for (int x = 0; x < b.e[0]; ++x)
          for (int c = 0; c < 3; ++c) {
            const int X = wrap(x - b.G + shift * sh[0]), Y = wrap(y - b.G + shift * sh[1]),
                      Z = wrap(z - b.G + shift * sh[2]);
            v[c][b.id(x, y, z)] = ug[c][X + n * (Y + n * Z)];
          }
    const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
    const double ell[3] = {1.0, 1.0, 1.0};
    Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, 0.01, 1.0, ell);
    auto t0 = toHost(o.tau[0]), t1 = toHost(o.tau[1]), t2 = toHost(o.tau[2]), dn = toHost(o.dnn);
    auto cl = toHost(o.cls), fb = toHost(o.fb);
    tauOut.assign(4 * n * n * n, 0.0);
    clsOut.assign(2 * n * n * n, 0);
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          const long i = b.id(x + b.G, y + b.G, z + b.G);
          const int X = wrap(x + shift * sh[0]), Y = wrap(y + shift * sh[1]),
                    Z = wrap(z + shift * sh[2]);
          const long gi = X + n * (Y + n * Z);
          tauOut[4 * gi] = t0[i];
          tauOut[4 * gi + 1] = t1[i];
          tauOut[4 * gi + 2] = t2[i];
          tauOut[4 * gi + 3] = dn[i];
          clsOut[2 * gi] = cl[i];
          clsOut[2 * gi + 1] = fb[i];
        }
  };
  std::vector<double> ta, tb;
  std::vector<int> ca, cb;
  run(0, ta, ca);
  run(1, tb, cb);
  long diff = 0, band = 0;
  for (std::size_t i = 0; i < ta.size(); ++i)
    diff += (ta[i] != tb[i]);
  for (std::size_t i = 0; i < ca.size(); ++i)
    diff += (ca[i] != cb[i]);
  for (std::size_t i = 0; i < ca.size(); i += 2)
    band += (ca[i] != 0);
  std::printf("(d) periodic seam: %ld band cells, %ld differing values under the shift\n", band,
              diff);
  CHECK(band > 50 && diff == 0, "shift invariance across the seam");
}

// ---------------------------------------------------------------------------------------------
// Host oracle of the band geometry: an independent transcription of §5.2/§5.4 on host arrays.
struct HostGeo {
  const Blk* b;
  std::vector<int> cls;
  std::vector<double> col, pm[3], pa;
  double eps, muA, muB, ell[3];
  int cl(const int q[3]) const { return cls[b->id(q[0], q[1], q[2])]; }
  double C(const int q[3]) const { return col[b->id3(q[0], q[1], q[2])]; }
  double mu(double f) const { return muB + (muA - muB) * f; }
  double res(double f) const { return f / muA + (1.0 - f) / muB; }
  void plane(const int q[3], double m[3], double& al) const {
    const int k = cl(q);
    if (k == 1) {
      const long j = b->id3(q[0], q[1], q[2]);
      for (int a = 0; a < 3; ++a)
        m[a] = pm[a][j];
      al = pa[j];
      return;
    }
    const int v = k - 2, ax = v / 2, side = v % 2;
    const bool one = C(q) >= 1.0 - eps;
    const double s = side == 0 ? (one ? -1.0 : 1.0) : (one ? 1.0 : -1.0);
    m[0] = m[1] = m[2] = 0.0;
    m[ax] = s;
    al = side == 0 ? 0.0 : s;
  }
  void normal(const double m[3], double n[3]) const {
    double p[3], s = 0;
    for (int a = 0; a < 3; ++a) {
      p[a] = m[a] / ell[a];
      s += p[a] * p[a];
    }
    s = std::sqrt(s);
    for (int a = 0; a < 3; ++a)
      n[a] = p[a] / s;
  }
  // the four cells of E_ac(k), sorted by local linear index
  std::vector<std::array<int, 3>> cells4(const int k[3], int a, int c) const {
    std::vector<std::pair<long, std::array<int, 3>>> v;
    for (int da = 0; da < 2; ++da)
      for (int dc = 0; dc < 2; ++dc) {
        std::array<int, 3> q = {k[0], k[1], k[2]};
        q[a] -= da;
        q[c] -= dc;
        v.push_back({b->id(q[0], q[1], q[2]), q});
      }
    std::sort(v.begin(), v.end());
    std::vector<std::array<int, 3>> o;
    for (auto& x : v)
      o.push_back(x.second);
    return o;
  }
  // the owner set of E_ac(k) (§5.8): the interfacial cells whose plane distance from the edge
  // midpoint is within 1e-6 of the closest, in local-linear-index order (empty: not a band edge)
  struct Member {
    int o[3];
    double m[3], al;
  };
  std::vector<Member> owners(const int k[3], int a, int c) const {
    std::vector<Member> all;
    std::vector<double> d;
    for (auto& q : cells4(k, a, c)) {
      if (cl(q.data()) == 0)
        continue;
      Member mb;
      plane(q.data(), mb.m, mb.al);
      double xi[3];
      for (int s = 0; s < 3; ++s) {
        mb.o[s] = q[s];
        xi[s] = k[s] - q[s];
      }
      xi[3 - a - c] = 0.5;
      const double* mm = mb.m;
      d.push_back(std::fabs(mm[0] * xi[0] + mm[1] * xi[1] + mm[2] * xi[2] - mb.al) /
                  std::sqrt(mm[0] * mm[0] + mm[1] * mm[1] + mm[2] * mm[2]));
      all.push_back(mb);
    }
    std::vector<Member> o;
    if (all.empty())
      return o;
    const double dmin = *std::min_element(d.begin(), d.end());
    for (std::size_t t = 0; t < all.size(); ++t)
      if (d[t] <= dmin + 1e-6)
        o.push_back(all[t]);
    return o;
  }
  // box x box fraction in the plane x_p = xiP over [lo_s, lo_s + 1] on the two other axes s
  static double boxFrac(const double m[3], double al, int p, double xiP, const double lo[3]) {
    const int q = p == 0 ? 1 : 0, bb = p == 2 ? 1 : 2;
    return sv::plicVolume(m[q], m[bb], 0.0, al - m[p] * xiP - m[q] * lo[q] - m[bb] * lo[bb]);
  }
  // hat (half-width 1, centred qc on axis q) x box (b from b0) in the plane x_p = xiP
  static double hatFrac(const double m[3], double al, int p, double xiP, int q, double qc,
                        double b0) {
    const int bb = 3 - p - q;
    return sv::plicVolume(m[q], m[q], m[bb], al - m[p] * xiP - m[q] * (qc - 1.0) - m[bb] * b0);
  }
  // row r's face at E_ac(k)
  double rowFace(const int k[3], int a, int c, int r, const int o[3], const double m[3],
                 double al) const {
    const int p = (r == c) ? a : c, bb = 3 - a - c;
    double lo[3];
    lo[r] = k[r] - o[r] - 0.5;
    lo[bb] = k[bb] - o[bb];
    lo[p] = 0;
    return boxFrac(m, al, p, k[p] - o[p], lo);
  }
  double link(const int k[3], int dd, int cc, const int o[3], const double m[3], double al) const {
    return hatFrac(m, al, cc, k[cc] - o[cc], dd, k[dd] - o[dd], k[3 - dd - cc] - o[3 - dd - cc]);
  }
};

HostGeo hostGeo(const Blk& b, const Geo& o, double eps, double muA, double muB,
                const double ell[3]) {
  HostGeo h;
  h.b = &b;
  h.cls = toHost(o.cls);
  h.col = toHost(o.col);
  for (int a = 0; a < 3; ++a) {
    h.pm[a] = toHost(o.pm[a]);
    h.ell[a] = ell[a];
  }
  h.pa = toHost(o.pa);
  h.eps = eps;
  h.muA = muA;
  h.muB = muB;
  return h;
}

// A sphere of tangent planes (one plane per cut cell, so owners choose between planes).
// Centre `cen` (internal coordinates) and radius R; by default the (e)/(f) sphere.
void sphereGeom(const Blk& b, const double ell[3], std::vector<double>& col,
                std::vector<double> (&pm)[3], std::vector<double>& pa,
                const double* centre = nullptr, double R = 3.1) {
  const double cen[3] = {centre ? centre[0] : 7.3 * ell[0], centre ? centre[1] : 6.7 * ell[1],
                         centre ? centre[2] : 7.1 * ell[2]};
  col.assign(b.cells3, 0.0);
  pa.assign(b.cells3, 0.0);
  for (int a = 0; a < 3; ++a)
    pm[a].assign(b.cells3, 0.0);
  for (int z = -1; z <= b.e[2]; ++z)
    for (int y = -1; y <= b.e[1]; ++y)
      for (int x = -1; x <= b.e[0]; ++x) {
        const int I[3] = {x, y, z};
        double n[3], s = 0;
        for (int a = 0; a < 3; ++a) {
          n[a] = ell[a] * (I[a] + 0.5) - cen[a];
          s += n[a] * n[a];
        }
        s = std::sqrt(s);
        double m[3], al = R;
        for (int a = 0; a < 3; ++a) {
          n[a] /= s;
          m[a] = n[a] * ell[a];
          al += n[a] * cen[a] - m[a] * I[a];
        }
        const long j = b.id3(x, y, z);
        const double C = sv::plicVolume(m[0], m[1], m[2], al);
        col[j] = C;
        if (C > 1e-12 && C < 1.0 - 1e-12) {
          for (int a = 0; a < 3; ++a)
            pm[a][j] = m[a];
          pa[j] = al;
        }
      }
}

DV muField(const Blk& b, const std::vector<double>& col, double muA, double muB) {
  std::vector<double> mu(b.cells);
  for (int z = 0; z < b.e[2]; ++z)
    for (int y = 0; y < b.e[1]; ++y)
      for (int x = 0; x < b.e[0]; ++x)
        mu[b.id(x, y, z)] = muB + (muA - muB) * col[b.id3(x, y, z)];
  return toDev(mu, "mu");
}

struct Bands {
  DV A[7];  // AC, AW, AE, AS, AN, AB, AT
  explicit Bands(long n) {
    for (auto& x : A)
      x = DV("band", n);
  }
};
template <class FP>
std::vector<std::vector<double>> buildBands(const Blk& b, FP fp, const double w[3]) {
  Bands B(b.cells);
  ibmBuildDiffusionVar(B.A[0], B.A[1], B.A[2], B.A[3], B.A[4], B.A[5], B.A[6], b.e[0], b.e[1],
                       b.e[2], b.G, fp, w[0], w[1], w[2]);
  Kokkos::fence();
  std::vector<std::vector<double>> o;
  for (auto& x : B.A)
    o.push_back(toHost(x));
  return o;
}

sv::SharpFaceProps sharpFP(const Geo& o, DV mu, int c, double rhoIdt,
                           sv::SvRole role = sv::SvRole::kImplicit) {
  VarFaceProps base;
  base.haveMu = true;
  base.mu = CCConst(mu);
  base.msc = o.g.st[c];
  base.rhoIdtC = rhoIdt;
  sv::SharpFaceProps sp;
  sp.base = base;
  sp.geo = o.g;
  sp.c = c;
  sp.role = role;
  return sp;
}

// (e) WO-3: the implicit band. WO-5b (§6.1): the cell link along c of an interfacial cell carries
// 2μ(C) in the implicit role and 0 in the explicit one; implicit + explicit = 2 × 'averaged' there
// and on every other cell link both roles equal 'averaged'.
void testBand() {
  long nBand = 0, nTie = 0, nNon = 0, badNon = 0, badSym = 0, badM = 0, badCell = 0, nCellIf = 0,
       badSplit = 0;
  double wOr = 0;
  for (int an = 0; an < 2; ++an)
    for (double ratio : {1e-2, 1e3}) {
      Blk b(12, 12, 12);
      double ell[3] = {1.0, 1.0, 1.0};
      if (an) {
        ell[1] = 1.5;
        ell[2] = 2.0;
      }
      const double w[3] = {1.0 / (ell[0] * ell[0]), 1.0 / (ell[1] * ell[1]),
                           1.0 / (ell[2] * ell[2])};
      std::vector<double> col, pm[3], pa, v[3];
      sphereGeom(b, ell, col, pm, pa);
      for (auto& x : v)
        x.assign(b.cells, 0.0);
      const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
      const double muA = ratio, muB = 1.0;
      Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, muA, muB, ell);
      HostGeo hg = hostGeo(b, o, 1e-12, muA, muB, ell);
      DV mu = muField(b, col, muA, muB);
      for (int c = 0; c < 3; ++c) {
        sv::SharpFaceProps sp = sharpFP(o, mu, c, 0.7);
        auto Av = buildBands(b, sp.base, w);
        auto As = buildBands(b, sp, w);
        auto Ae = buildBands(b, sharpFP(o, mu, c, 0.7, sv::SvRole::kExplicit), w);
        for (int z = b.G; z < b.e[2] - b.G; ++z)
          for (int y = b.G; y < b.e[1] - b.G; ++y)
            for (int x = b.G; x < b.e[0] - b.G; ++x) {
              const long i = b.id(x, y, z);
              bool rowBand = false, rowCell = false;
              double sum = 0;
              for (int f = 0; f < 6; ++f) {
                const int a = f / 2, sg = f % 2;  // AW(-x) AE(+x) AS AN AB AT
                const double sv_ = As[1 + f][i], av = Av[1 + f][i];
                sum += -sv_;
                badM += (sv_ > 0.0);
                if (a == c) {
                  int q[3] = {x, y, z};  // the cell between the two u_c faces: the lower one
                  if (!sg)
                    q[c] -= 1;
                  const bool itf = hg.cl(q) != 0;
                  nCellIf += itf;
                  rowCell = rowCell || itf;
                  badCell += (sv_ != (itf ? 2.0 * av : av));
                  badSplit += (sv_ + Ae[1 + f][i] != 2.0 * av) || (!itf && Ae[1 + f][i] != av) ||
                              (itf && Ae[1 + f][i] != 0.0);
                  continue;
                }
                badSplit += (Ae[1 + f][i] != sv_);
                int k[3] = {x, y, z};
                if (sg)
                  k[a] += 1;
                const auto ow = hg.owners(k, a, c);
                if (ow.empty()) {
                  ++nNon;
                  badNon += (sv_ != av);
                  continue;
                }
                ++nBand;
                nTie += ow.size() > 1;
                rowBand = true;
                double F = 0.0;
                for (const auto& mb : ow)
                  F += hg.rowFace(k, a, c, c, mb.o, mb.m, mb.al);
                const double ex = -hg.mu(F / ow.size()) * w[a];
                wOr = std::max(wOr, std::fabs(sv_ - ex) / std::fabs(ex));
              }
              if (!rowBand && !rowCell)
                badCell += (As[0][i] != Av[0][i]);
              badM += !(std::fabs(As[0][i] - 0.7 - sum) <= 1e-13 * As[0][i]);
              // per-component symmetry with the +a neighbour (inner)
              for (int a = 0; a < 3; ++a) {
                int q[3] = {x, y, z};
                q[a] += 1;
                if (q[a] >= b.e[a] - b.G)
                  continue;
                badSym += (As[2 + 2 * a][i] != As[1 + 2 * a][b.id(q[0], q[1], q[2])]);
              }
            }
      }
    }
  std::printf(
      "(e) implicit band: %ld band entries (%ld tied; oracle %.2e), %ld non-band (%ld differ), "
      "cells/"
      "diag %ld differ, symmetry %ld, M-matrix %ld; split: %ld interfacial cell links, %ld "
      "wrong\n",
      nBand, nTie, wOr, nNon, badNon, badCell, badSym, badM, nCellIf, badSplit);
  CHECK(nBand > 500 && wOr <= 1e-14, "band entries vs oracle");
  CHECK(badNon == 0 && badCell == 0, "non-band entries bitwise 'averaged'");
  CHECK(badSym == 0 && badM == 0, "symmetric M-matrix");
  CHECK(nCellIf > 100 && badSplit == 0,
        "WO-5b split: 2mu(C) implicit, 0 explicit on interfacial cells");
}

// Host transcription of the explicit band terms of the u_c rows (§5.4): sample arrays, then the
// divergence.
double hostCellExtra(const HostGeo& h, const int q[3], int c, const std::vector<double> (&v)[3],
                     const std::vector<double> (&tau)[3], const std::vector<double>& dnn) {
  if (h.cl(q) == 0)
    return 0.0;
  double m[3], al, n[3];
  h.plane(q, m, al);
  h.normal(m, n);
  double lo[3] = {0, 0, 0};
  const double cb = HostGeo::boxFrac(m, al, c, 0.5, lo), C = h.C(q);
  const long j = h.b->id(q[0], q[1], q[2]);
  const double mcc = v[c][j + h.b->st[c]] - v[c][j];
  return 2.0 * (h.muA - h.muB) * (cb - C) * (mcc - dnn[j]) +
         2.0 * n[c] * tau[c][j] * (1.0 - h.mu(cb) * h.res(C));
}
double hostEdgeY(const HostGeo& h, const int k[3], int a, int c,
                 const std::vector<double> (&tau)[3]) {
  const auto ow = h.owners(k, a, c);
  if (ow.empty())
    return 0.0;
  double t[3] = {0, 0, 0};
  int nI = 0;
  for (auto& q : h.cells4(k, a, c))
    nI += h.cl(q.data()) != 0;
  for (auto& q : h.cells4(k, a, c))
    if (h.cl(q.data()) != 0)
      for (int s = 0; s < 3; ++s)
        t[s] += tau[s][h.b->id(q[0], q[1], q[2])] / nI;
  double y = 0.0;
  for (const auto& mb : ow) {
    double n[3];
    h.normal(mb.m, n);
    const double mub = h.mu(h.rowFace(k, a, c, c, mb.o, mb.m, mb.al));
    const double rac = h.res(h.link(k, a, c, mb.o, mb.m, mb.al)),
                 rca = h.res(h.link(k, c, a, mb.o, mb.m, mb.al));
    y += n[a] * t[c] * (1.0 - mub * rac) + n[c] * t[a] * (1.0 - mub * rca);
  }
  return y / ow.size();
}

// (h) WO-9.2 (§5.8): reflection self-symmetry of the owner-set mean. A sphere of radius 5.3 cells
// centred on a grid vertex of a 24^3 periodic box (tangent planes), mu ratio 1e3, u^n the pure
// strain u = (X, Y, -2Z) about the centre (wrapped at the box faces: its jump lies outside every
// fit block). For each axis reflection R about the centre and every band edge E,
// |G(E) - s G(R E)| <= 1e-13 max|G| for G in {F_a, F_c, th_ac, th_ca, T_a, T_c}; s = -1 for T
// iff R reflects a or c. The case ties (census), and n_E is reflection-invariant.
void testReflection() {
  Blk b(24, 24, 24);
  const double ell[3] = {1.0, 1.0, 1.0};
  const int X0 = b.G + 12;  // the centre, a grid vertex (solver index coordinates)
  const double cen[3] = {(double)X0, (double)X0, (double)X0};
  std::vector<double> col, pm[3], pa, v[3];
  sphereGeom(b, ell, col, pm, pa, cen, 5.3);
  for (auto& x : v)
    x.assign(b.cells, 0.0);
  for (int z = 0; z < b.e[2]; ++z)
    for (int y = 0; y < b.e[1]; ++y)
      for (int x = 0; x < b.e[0]; ++x) {
        const int q[3] = {x, y, z};
        for (int a = 0; a < 3; ++a) {
          const double r = (double)(((q[a] - b.G) % 24 + 24) % 24) - 12.0;  // face a - centre
          v[a][b.id(x, y, z)] = (a == 2) ? -2.0 * r : r;
        }
      }
  const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
  Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, 1e3, 1.0, ell);
  const auto census =
      sv::sharpCensus(o.g, o.fb, C3{b.G, b.G, b.G}, C3{b.e[0] - b.G, b.e[1] - b.G, b.e[2] - b.G});
  DV out("refl", 21 * b.cells);
  const sv::SharpGeom g = o.g;
  const CCConst tc[3] = {o.tau[0], o.tau[1], o.tau[2]};
  const int G = b.G;
  Kokkos::parallel_for(
      "test::reflection", Kokkos::RangePolicy<CCExec>(0, b.cells), KOKKOS_LAMBDA(long i) {
        const int z = (int)(i / g.st[2]), rr = (int)(i - z * g.st[2]), y = rr / (int)g.st[1],
                  x = rr - y * (int)g.st[1];
        for (int ac = 0; ac < 3; ++ac)
          out(21 * i + 7 * ac) = 0.0;
        if (x < G || y < G || z < G || x >= g.e[0] - G || y >= g.e[1] - G || z >= g.e[2] - G)
          return;
        const int k[3] = {x, y, z};
        for (int ac = 0; ac < 3; ++ac) {
          const int a = (ac == 2) ? 1 : 0, c = (ac == 0) ? 1 : 2;
          const long o7 = 21 * i + 7 * ac;
          const double W =
              g.forEachOwner(k, a, c, [](const int*, const double*, double, double) {});
          if (W == 0.0)
            continue;
          double sp = 0.0, gp = 0.0;
          bool bl = false;
          out(o7) = g.edgeTieKeys(k, ac, sp, gp, bl);  // n_E: the plateau members (census)
          double f = 0.0, tac = 0.0, tca = 0.0;
          g.edgeFaceFraction(k, a, c, a, f);
          out(o7 + 1) = f;
          g.edgeFaceFraction(k, a, c, c, f);
          out(o7 + 2) = f;
          g.edgeLinkFractions(k, a, c, tac, tca);
          out(o7 + 3) = tac;
          out(o7 + 4) = tca;
          out(o7 + 5) = g.edgeTraction(k, a, c, a, tc);
          out(o7 + 6) = g.edgeTraction(k, a, c, c, tc);
        }
      });
  const auto h = toHost(out);
  double mx[6] = {0, 0, 0, 0, 0, 0}, wd[6] = {0, 0, 0, 0, 0, 0};
  long nBand = 0, nTie = 0, badN = 0;
  for (int r = 0; r < 3; ++r)
    for (int z = G; z < b.e[2] - G; ++z)
      for (int y = G; y < b.e[1] - G; ++y)
        for (int x = G; x < b.e[0] - G; ++x)
          for (int ac = 0; ac < 3; ++ac) {
            const int a = (ac == 2) ? 1 : 0, c = (ac == 0) ? 1 : 2, bb = 3 - a - c;
            const double* e = &h[21 * b.id(x, y, z) + 7 * ac];
            if (e[0] == 0.0)
              continue;
            int kr[3] = {x, y, z};
            kr[r] = (r == bb) ? 2 * X0 - 1 - kr[r] : 2 * X0 - kr[r];  // the reflected edge
            const double* f = &h[21 * b.id(kr[0], kr[1], kr[2]) + 7 * ac];
            if (r == 0) {
              ++nBand;
              nTie += e[0] > 1.0;
            }
            if (f[0] != e[0]) {
              ++badN;
              continue;
            }
            for (int q = 0; q < 6; ++q) {
              const double s = (q >= 4 && r != bb) ? -1.0 : 1.0;
              mx[q] = std::max(mx[q], std::fabs(e[1 + q]));
              wd[q] = std::max(wd[q], std::fabs(e[1 + q] - s * f[1 + q]));
            }
          }
  double worst = 0.0;
  for (int q = 0; q < 6; ++q)
    worst = std::max(worst, mx[q] > 0.0 ? wd[q] / mx[q] : 0.0);
  std::printf(
      "(h) reflection self-symmetry: %ld band edges, %ld tied (census %ld, spread %.2e, gap "
      "%.2e); worst |G(E) - s G(RE)| / max|G|: F_a %.1e F_c %.1e th_ac %.1e th_ca %.1e T_a %.1e "
      "T_c %.1e; n_E mismatches %ld; fallbacks %ld\n",
      nBand, nTie, census.tied, census.tieSpread, census.untiedGap, wd[0] / mx[0], wd[1] / mx[1],
      wd[2] / mx[2], wd[3] / mx[3], wd[4] / mx[4], wd[5] / mx[5], badN, census.fallback);
  CHECK(nBand > 1000 && nTie > 0 && census.tied == nTie, "a band with ties (%ld / %ld)", nTie,
        census.tied);
  CHECK(badN == 0, "n_E reflection-invariant");
  CHECK(worst <= 1e-13, "reflection self-symmetry %.2e", worst);
}

// (j) §5.9 item 7: the owner weight -- plateau, C¹ ramp, zero; evaluated in the execution space.
void testOwnerWeight() {
  constexpr double t0 = sv::kSvOwnerTieTol, t1 = sv::kSvOwnerRampTol;
  constexpr int nS = 100000;
  const double K = 1.5 / (t1 - t0), hh = 1e-9;
  DV out("ownerWeight", nS + 1 + 7);
  Kokkos::parallel_for(
      "test::ownerWeight", Kokkos::RangePolicy<CCExec>(0, nS + 1), KOKKOS_LAMBDA(int i) {
        out(i) = sv::ownerWeight(2.0 * t1 * (double)i / (double)nS);
        if (i == 0) {
          out(nS + 1) = sv::ownerWeight(0.0);
          out(nS + 2) = sv::ownerWeight(t0);
          out(nS + 3) = sv::ownerWeight(t1);
          out(nS + 4) = sv::ownerWeight(2.0 * t1);
          out(nS + 5) = sv::ownerWeight(Kokkos::Experimental::quiet_NaN<double>::value);
          out(nS + 6) = sv::ownerWeight(t0 + hh);
          out(nS + 7) = sv::ownerWeight(t1 - hh);
        }
      });
  const auto h = toHost(out);
  long nInc = 0;
  double slope = 0.0;
  for (int i = 0; i < nS; ++i) {
    nInc += h[i + 1] > h[i];
    slope = std::max(slope, std::fabs(h[i + 1] - h[i]) / (2.0 * t1 / nS));
  }
  const double e2 = 3.0 * (hh / (t1 - t0)) * (hh / (t1 - t0));
  std::printf(
      "(j) owner weight: w(0) %g w(t0) %g w(t1) %g w(2t1) %g w(NaN) %g; %ld increases on %d "
      "points, max slope %.6e (K %.6e); 1 - w(t0 + h) %.3e, w(t1 - h) %.3e (bound %.3e)\n",
      h[nS + 1], h[nS + 2], h[nS + 3], h[nS + 4], h[nS + 5], nInc, nS + 1, slope, K,
      1.0 - h[nS + 6], h[nS + 7], e2);
  CHECK(h[nS + 1] == 1.0 && h[nS + 2] == 1.0, "w = 1 exactly on the plateau");
  CHECK(h[nS + 3] == 0.0 && h[nS + 4] == 0.0 && h[nS + 5] == 0.0, "w = 0 from t1 on and at NaN");
  CHECK(nInc == 0, "w non-increasing (%ld increases)", nInc);
  CHECK(slope <= K * (1.0 + 1e-9), "|dw/dg| <= K (%.6e)", slope);
  CHECK(std::fabs(1.0 - h[nS + 6]) <= e2 && std::fabs(h[nS + 7]) <= e2, "C1 at both ends");
}

// FNV-1a over the bytes of a double array (the tie-free identity's digest).
std::uint64_t fnv(std::uint64_t h, const std::vector<double>& x) {
  for (double d : x) {
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    for (int s = 0; s < 8; ++s) {
      h ^= (u >> (8 * s)) & 0xffu;
      h *= 0x100000001b3ULL;
    }
  }
  return h;
}

// (i) WO-9.2 (§5.8): a tie-free case is bitwise the single-owner operator of f01a251 (host). The
// (e)/(f) sphere at mu ratio 1e3 and h ratio 1 : 1.5 : 2, random u^n: its census reads 0 tied
// edges (the isotropic sphere has 7: not this case), and every band-op output -- the fit (tau_hat,
// D_nn, fallbacks), the implicit and the explicit bands of the three components, the explicit band
// RHS -- is hashed (FNV-1a over the bytes). The digests were recorded once from the f01a251
// operator (the WO-9.1 tree, host OpenMP, 2026-11-04). On a device backend FMA contraction may
// differ: the digest is printed, not checked.
void testTieFree() {
  const std::uint64_t want = 0x453f2865e4cc0381ULL;
  constexpr bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, CCMem>::accessible;
  {
    Blk b(12, 12, 12);
    const double ell[3] = {1.0, 1.5, 2.0};
    const double w[3] = {1.0 / (ell[0] * ell[0]), 1.0 / (ell[1] * ell[1]), 1.0 / (ell[2] * ell[2])};
    std::vector<double> col, pm[3], pa, v[3];
    sphereGeom(b, ell, col, pm, pa);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (auto& x : v) {
      x.resize(b.cells);
      for (auto& y : x)
        y = U(rng);
    }
    const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
    const double muA = 1e3, muB = 1.0;
    Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, muA, muB, ell);
    const auto cen =
        sv::sharpCensus(o.g, o.fb, C3{b.G, b.G, b.G}, C3{b.e[0] - b.G, b.e[1] - b.G, b.e[2] - b.G});
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (int a = 0; a < 3; ++a)
      h = fnv(h, toHost(o.tau[a]));
    h = fnv(h, toHost(o.dnn));
    {
      const auto fb = toHost(o.fb);
      h = fnv(h, std::vector<double>(fb.begin(), fb.end()));
    }
    DV mu = muField(b, col, muA, muB);
    for (int c = 0; c < 3; ++c) {
      for (const auto& x : buildBands(b, sharpFP(o, mu, c, 0.7), w))
        h = fnv(h, x);
      for (const auto& x : buildBands(b, sharpFP(o, mu, c, 0.7, sv::SvRole::kExplicit), w))
        h = fnv(h, x);
    }
    DV rd("rs", b.cells);
    Kokkos::deep_copy(rd, 1.0);
    const CCConst tc[3] = {o.tau[0], o.tau[1], o.tau[2]}, vc[3] = {o.v[0], o.v[1], o.v[2]};
    for (int c = 0; c < 3; ++c) {
      DV bb("bb", b.cells);
      sv::addSharpViscousRhs(c, bb, CCConst(rd), vc, o.g, tc, CCConst(o.dnn), b.G, w[c], false);
      Kokkos::fence();
      h = fnv(h, toHost(bb));
    }
    std::printf("(i) tie-free identity: %ld band cells, %ld tied edges, digest 0x%016llx%s\n",
                cen.band, cen.tied, (unsigned long long)h,
                host ? (h == want ? " (= f01a251)" : " (MISMATCH)") : " (device: not checked)");
    CHECK(cen.band > 50 && cen.tied == 0, "a tie-free case");
    if (host)
      CHECK(h == want, "tie-free band-op outputs bitwise f01a251");
  }
}

// (f) WO-4: the explicit kernel against the host transcription.
void testRhsOracle() {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  double worst = 0;
  long nRow = 0;
  for (int an = 0; an < 2; ++an)
    for (double ratio : {1e-2, 1e3}) {
      Blk b(12, 12, 12);
      double ell[3] = {1.0, 1.0, 1.0};
      if (an) {
        ell[1] = 1.5;
        ell[2] = 2.0;
      }
      std::vector<double> col, pm[3], pa, v[3], tau[3], dnn(b.cells), rs(b.cells);
      sphereGeom(b, ell, col, pm, pa);
      for (int a = 0; a < 3; ++a) {
        v[a].resize(b.cells);
        tau[a].resize(b.cells);
        for (long i = 0; i < b.cells; ++i) {
          v[a][i] = U(rng);
          tau[a][i] = U(rng);
        }
      }
      for (long i = 0; i < b.cells; ++i) {
        dnn[i] = U(rng);
        rs[i] = 1.0 + 0.25 * U(rng);
      }
      const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
      const double muA = ratio, muB = 1.0;
      Geo o = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, muA, muB, ell);
      HostGeo hg = hostGeo(b, o, 1e-12, muA, muB, ell);
      DV td[3] = {toDev(tau[0]), toDev(tau[1]), toDev(tau[2])}, dd = toDev(dnn), rd = toDev(rs);
      const CCConst tc[3] = {td[0], td[1], td[2]}, vc[3] = {o.v[0], o.v[1], o.v[2]};
      for (int c = 0; c < 3; ++c) {
        DV bb("bb", b.cells);
        const double wc = 1.0 / (ell[c] * ell[c]);
        sv::addSharpViscousRhs(c, bb, CCConst(rd), vc, o.g, tc, CCConst(dd), b.G, wc, false);
        Kokkos::fence();
        auto bh = toHost(bb);
        const int a1 = c == 0 ? 1 : 0, a2 = c == 2 ? 1 : 2;
        double mx = 0, md = 0;
        for (int z = b.G; z < b.e[2] - b.G; ++z)
          for (int y = b.G; y < b.e[1] - b.G; ++y)
            for (int x = b.G; x < b.e[0] - b.G; ++x) {
              const int q[3] = {x, y, z};
              int qm[3] = {x, y, z}, p1[3] = {x, y, z}, p2[3] = {x, y, z};
              qm[c] -= 1;
              p1[a1] += 1;
              p2[a2] += 1;
              const double X =
                  hostCellExtra(hg, q, c, v, tau, dnn) - hostCellExtra(hg, qm, c, v, tau, dnn);
              const double Y1 = hostEdgeY(hg, p1, a1, c, tau) - hostEdgeY(hg, q, a1, c, tau);
              const double Y2 = hostEdgeY(hg, p2, a2, c, tau) - hostEdgeY(hg, q, a2, c, tau);
              const long i = b.id(x, y, z);
              const double ex =
                  rs[i] * (wc * X + Y1 / (ell[a1] * ell[c]) + Y2 / (ell[a2] * ell[c]));
              mx = std::max(mx, std::fabs(ex));
              md = std::max(md, std::fabs(bh[i] - ex));
              nRow += (ex != 0.0);
            }
        worst = std::max(worst, md / mx);
      }
    }
  std::printf("(f) explicit kernel vs host oracle: %ld nonzero rows, worst %.2e relative\n", nRow,
              worst);
  CHECK(nRow > 500 && worst <= 1e-13, "explicit kernel vs host oracle");
}

// (g) WO-4: the full operator on exact face averages of kinked-linear fields (Couette and
// extension, 9 orientations) -- implicit band + band transpose + tau-terms + midsection - grad p.
// Review F3: the fit fallback. The normal equations depend on the geometry only (a NaN VELOCITY
// datum gives a non-finite tau_hat, not a fallback), so a NaN in the PLIC normal of one
// interfacial cell q* is what makes its fit singular (fitSolve's pivot test is NaN-safe): q* and
// only interfacial cells within the fit's reach fall back, each with tau_hat = D_nn = 0 exactly,
// every other fit stays finite, the census counts the fallbacks, and the band terms on the CLEAN
// geometry with that fit are finite (a fallback's tau_hat = 0 enters the edge mean, §5.2).
void testFallback() {
  Blk b(12, 12, 12);
  const double ell[3] = {1.0, 1.0, 1.0};
  std::vector<double> col, pm[3], pa, v[3];
  sphereGeom(b, ell, col, pm, pa);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  for (int a = 0; a < 3; ++a) {
    v[a].resize(b.cells);
    for (long i = 0; i < b.cells; ++i)
      v[a][i] = U(rng);
  }
  const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
  const double muA = 1e-2, muB = 1.0;
  Geo clean = makeGeo(b, col, pm, pa, v, dlo, dhi, 1e-12, muA, muB, ell);
  const auto cls = toHost(clean.cls);
  const auto fb0 = toHost(clean.fb);
  long qs = -1, nFb0 = 0;
  for (int z = b.G; z < b.e[2] - b.G; ++z)
    for (int y = b.G; y < b.e[1] - b.G; ++y)
      for (int x = b.G; x < b.e[0] - b.G; ++x) {
        const long i = b.id(x, y, z);
        nFb0 += fb0[i];
        if (qs < 0 && cls[i] == sv::kSvPlane && x > 5 && y > 5 && z > 5)
          qs = i;
      }
  CHECK(qs >= 0 && nFb0 == 0, "an interior plane cell (%ld), no fallback on the clean field (%ld)",
        qs, nFb0);
  if (qs < 0)
    return;
  const int qz = (int)(qs / b.st[2]), qy = (int)((qs - qz * b.st[2]) / b.st[1]),
            qx = (int)(qs - qz * b.st[2] - qy * b.st[1]);
  std::vector<double> pmn[3] = {pm[0], pm[1], pm[2]};
  pmn[0][b.id3(qx, qy, qz)] = std::numeric_limits<double>::quiet_NaN();
  Geo o = makeGeo(b, col, pmn, pa, v, dlo, dhi, 1e-12, muA, muB, ell);
  const auto fb = toHost(o.fb);
  const std::vector<double> tau[3] = {toHost(o.tau[0]), toHost(o.tau[1]), toHost(o.tau[2])};
  const auto dnn = toHost(o.dnn);
  long nFb = 0, nFar = 0, nNonZero = 0, nNonFinite = 0, reach = 0;
  for (int z = b.G; z < b.e[2] - b.G; ++z)
    for (int y = b.G; y < b.e[1] - b.G; ++y)
      for (int x = b.G; x < b.e[0] - b.G; ++x) {
        const long i = b.id(x, y, z);
        const bool f = fb[i] != 0;
        nFb += f;
        const int d = std::max({std::abs(x - qx), std::abs(y - qy), std::abs(z - qz)});
        if (f) {
          reach = std::max<long>(reach, d);
          nFar += (d > 2 || cls[i] == sv::kSvNone);
          nNonZero += (tau[0][i] != 0.0 || tau[1][i] != 0.0 || tau[2][i] != 0.0 || dnn[i] != 0.0);
        } else
          nNonFinite += !(std::isfinite(tau[0][i]) && std::isfinite(tau[1][i]) &&
                          std::isfinite(tau[2][i]) && std::isfinite(dnn[i]));
      }
  const auto cen =
      sv::sharpCensus(o.g, o.fb, C3{b.G, b.G, b.G}, C3{b.e[0] - b.G, b.e[1] - b.G, b.e[2] - b.G});
  long nBad = 0;
  {
    const CCConst tc[3] = {o.tau[0], o.tau[1], o.tau[2]},
                  vc[3] = {clean.v[0], clean.v[1], clean.v[2]};
    DV rs("rs", b.cells);
    Kokkos::deep_copy(rs, 1.0);
    for (int c = 0; c < 3; ++c) {
      DV bb("bb", b.cells);
      sv::addSharpViscousRhs(c, bb, CCConst(rs), vc, clean.g, tc, CCConst(o.dnn), b.G, 1.0, false);
      Kokkos::fence();
      for (double x : toHost(bb))
        nBad += !std::isfinite(x);
    }
  }
  std::printf(
      "fallback: NaN normal at (%d,%d,%d): %ld fallbacks (reach %ld), census %ld, tau/D_nn nonzero "
      "on a fallback %ld, beyond reach 2 or off the band %ld, non-finite fits %ld, non-finite RHS "
      "%ld\n",
      qx, qy, qz, nFb, reach, cen.fallback, nNonZero, nFar, nNonFinite, nBad);
  CHECK(fb[qs] == 1 && nFb >= 1, "q* falls back");
  CHECK(cen.fallback == nFb, "census %ld vs %ld", cen.fallback, nFb);
  CHECK(nNonZero == 0 && nFar == 0 && nNonFinite == 0, "fallback state");
  CHECK(nBad == 0, "finite band terms");
}

void testPatch() {
  const double angs[3] = {0.0, 22.5, 45.0};
  double worst = 0;
  long nCase = 0;
  for (int an = 0; an < 2; ++an)
    for (int ib = 0; ib < 3; ++ib)
      for (int ig = 0; ig < 3; ++ig)
        for (double off : {0.0, 0.3})
          for (double ratio : {1e-2, 1e2, 1e-3, 1e3})
            for (int kind = 0; kind < 2; ++kind) {
              KCase kc{Blk(8, 8, 8), {}, {1.0, 1.0, 1.0}, {}, {}, {}, {}};
              if (an) {
                kc.ell[1] = 1.5;
                kc.ell[2] = 2.0;
              }
              const double be = angs[ib] * kPi / 180, ga = angs[ig] * kPi / 180;
              Kinked& K = kc.K;
              K.n[0] = std::cos(be) * std::cos(ga);
              K.n[1] = std::cos(be) * std::sin(ga);
              K.n[2] = std::sin(be);
              double t1[3] = {-std::sin(ga), std::cos(ga), 0.0};
              for (int a = 0; a < 3; ++a)
                K.xp[a] = kc.ell[a] * 6.0 + off * K.n[a];
              K.muA = ratio;
              K.muB = 1.0;
              const double mulo = std::min(K.muA, K.muB), muhi = std::max(K.muA, K.muB);
              for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                  K.Llo[3 * p + q] =
                      kind == 0 ? K.n[p] * t1[q] / mulo : K.n[p] * K.n[q] - t1[p] * t1[q];
              setKink(K, ratio);
              buildKinked(kc);
              const Blk& b = kc.b;
              const int dlo[3] = {-kBig, -kBig, -kBig}, dhi[3] = {kBig, kBig, kBig};
              Geo o = makeGeo(b, kc.col, kc.pm, kc.pa, kc.v, dlo, dhi, 0.0, K.muA, K.muB, kc.ell);
              DV mu = muField(b, kc.col, K.muA, K.muB);
              const double w[3] = {1.0 / (kc.ell[0] * kc.ell[0]), 1.0 / (kc.ell[1] * kc.ell[1]),
                                   1.0 / (kc.ell[2] * kc.ell[2])};
              const double dnnEx = kind == 0 ? 0.0 : 1.0;
              const double jump = 2.0 * (K.muA - K.muB) * dnnEx;  // p_A - p_B
              const double scale = kind == 0 ? 1.0 : 2.0 * muhi;
              const CCConst vc[3] = {o.v[0], o.v[1], o.v[2]};
              const CCConst tc[3] = {o.tau[0], o.tau[1], o.tau[2]};
              DV ones("rs", b.cells);
              Kokkos::deep_copy(ones, 1.0);
              double md = 0;
              for (int c = 0; c < 3; ++c) {
                sv::SharpFaceProps sp = sharpFP(o, mu, c, 0.0);
                auto A = buildBands(b, sp, w);
                DV bb("bb", b.cells);
                addFullStressRhs<sv::SharpFaceProps, false>(
                    c, bb, CCConst(ones), vc, sharpFP(o, mu, c, 0.0, sv::SvRole::kExplicit),
                    C3{b.e[0], b.e[1], b.e[2]}, b.G, w[c], false);
                sv::addSharpViscousRhs(c, bb, CCConst(ones), vc, o.g, tc, CCConst(o.dnn), b.G, w[c],
                                       false);
                Kokkos::fence();
                auto bh = toHost(bb);
                const auto& vh = kc.v[c];
                for (int z = b.G + 1; z < b.e[2] - b.G - 1; ++z)
                  for (int y = b.G + 1; y < b.e[1] - b.G - 1; ++y)
                    for (int x = b.G + 1; x < b.e[0] - b.G - 1; ++x) {
                      const long i = b.id(x, y, z);
                      double Ku = A[0][i] * vh[i];
                      for (int f = 0; f < 6; ++f)
                        Ku += A[1 + f][i] * vh[i + (f % 2 ? 1 : -1) * b.st[f / 2]];
                      int qm[3] = {x, y, z};
                      qm[c] -= 1;
                      const double pi = jump * kc.col[b.id3(x, y, z)],
                                   pm_ = jump * kc.col[b.id3(qm[0], qm[1], qm[2])];
                      const double R = -Ku + bh[i] - w[c] * (pi - pm_);
                      md = std::max(md, std::fabs(R));
                    }
              }
              ++nCase;
              worst = std::max(worst, md / scale);
            }
  std::printf("(g) operator patch test: %ld cases, worst residual %.2e of the stress scale\n",
              nCase, worst);
  CHECK(worst <= 1e-11, "kinked-linear patch test");
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    testExtrusion();
    testKinked();
    testVirtual();
    testSeam();
    testBand();
    testRhsOracle();
    testFallback();
    testPatch();
    testReflection();
    testTieFree();
    testOwnerWeight();
  }
  Kokkos::finalize();
  if (failures)
    std::printf("FAILED (%d)\n", failures);
  else
    std::printf("PASS\n");
  return failures ? 1 : 0;
}
