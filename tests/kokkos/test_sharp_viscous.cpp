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
//       tau/D_nn are bitwise invariant under a circular shift of the block by (5, 3, 2) cells.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

// Edge queries on the device: owner flag, owner distance, F_row(c), F_row(a), L_ac, L_ca.
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
        int ow[3];
        double m[3], al;
        const bool has = g.edgeOwner(k, a, c, ow, m, al);
        out(6 * t) = has ? 1.0 : 0.0;
        if (!has)
          return;
        double xi[3];
        xi[a] = k[a] - ow[a];
        xi[c] = k[c] - ow[c];
        xi[b] = 0.5;
        out(6 * t + 1) = Kokkos::fabs(m[0] * xi[0] + m[1] * xi[1] + m[2] * xi[2] - al) /
                         Kokkos::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
        out(6 * t + 2) = g.rowFaceFraction(k, a, c, c, ow, m, al);
        out(6 * t + 3) = g.rowFaceFraction(k, a, c, a, ow, m, al);
        out(6 * t + 4) = g.linkFraction(k, a, c, ow, m, al);
        out(6 * t + 5) = g.linkFraction(k, c, a, ow, m, al);
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
      if (er[t][0] != 1.0) {
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
        if (r[0] != 1.0)
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
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    testExtrusion();
    testKinked();
    testVirtual();
    testSeam();
  }
  Kokkos::finalize();
  if (failures)
    std::printf("FAILED (%d)\n", failures);
  else
    std::printf("PASS\n");
  return failures ? 1 : 0;
}
