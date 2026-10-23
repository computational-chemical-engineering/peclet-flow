/* Case A -- Esmaeeli & Tryggvason (1999) ordered bubble array, as in Innocenti et al. (2021, JFM;
   arXiv 2002.02209) Sec. 4 / App. A Table 3 (their Basilisk set-up re-created: no published script).
   Ar = sqrt(rho_l |drho| g d^3)/mu_l = 29.9, Bo = |drho| g d^2/sigma = 2, rho_b/rho_l = mu_b/mu_l = 0.1,
   one bubble in a periodic cube L = 1.6 d, uniform grid N = 1.6 d/h, fluid initially at rest,
   body force -<rho> g added (Innocenti Sec. 4) so the periodic system does not accelerate.
   Units: d = g = rho_l = 1 (time unit sqrt(d/g)). Vertical = x, gravity along -x.
   Phase 1 (f = 1) is the bubble. Full viscous stress div(2 mu D) (centered.h + viscosity.h).
   Compile-time ablations: -DRHO_RATIO=1 (density jump off), -DMU_RATIO=1 (viscosity jump off).
   Log (stdout every DTOUT): t ub umix uzm xcent vg, ub = sum f u_x / sum f (cell-centred u),
   umix = volume mean, uzm = mass mean (rho linear in f), xcent = unwrapped gas centroid. */
#include "grid/multigrid3D.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

#ifndef RHO_RATIO
# define RHO_RATIO 0.1
#endif
#ifndef MU_RATIO
# define MU_RATIO 0.1
#endif
#ifndef AR
# define AR 29.9
#endif
#ifndef BO
# define BO 2.
#endif
#ifndef TEND
# define TEND 20.
#endif
#ifndef DTOUT
# define DTOUT 0.05
#endif

int LEVEL = 4;              // N = 2^LEVEL; d/h = N/1.6
double rbmean = 0., xcent = 0., G0 = 1.;   // G0 = g
/* buoyancy density: the CASE density rho_b = 0.1 rho_l whatever RHO_RATIO is. At RHO_RATIO = 0.1 it is
   the inertial density and the force is exactly -(rho - <rho>) g; at RHO_RATIO = 1 (density-jump
   ablation) inertia is uniform and the buoyancy is Boussinesq, so Ar and Bo keep their case values. */
#define rbuoy(f) (clamp(f,0.,1.)*(0.1 - 1.) + 1.)

int main (int argc, char * argv[])
{
  if (argc > 1) LEVEL = atoi (argv[1]);
  if (argc > 2) TOLERANCE = atof (argv[2]);
  size (1.6);
  origin (-0.8, -0.8, -0.8);
  periodic (right);
  periodic (top);
  periodic (front);
  init_grid (1 << LEVEL);
  rho1 = RHO_RATIO, rho2 = 1.;
  mu2 = sqrt (0.9)/AR;                  // mu_l from Ar with the case's |drho| = 0.9 rho_l (always)
  mu1 = MU_RATIO*mu2;
  f.sigma = 0.9/BO;                     // from Bo, same |drho|
  run();
}

event init (t = 0)
{
  fraction (f, sq(0.5) - sq(x) - sq(y) - sq(z));
  double rs = 0.;
  foreach (reduction(+:rs))
    rs += rbuoy(f[])*dv();
  rbmean = rs/cube (1.6);
  fprintf (stderr, "# N %d d/h %g rho1 %g rho2 %g mu1 %g mu2 %g sigma %g rbmean %.10g TOL %g\n",
	   1 << LEVEL, (1 << LEVEL)/1.6, rho1, rho2, mu1, mu2, f.sigma, rbmean, TOLERANCE);
}

/* body force per unit mass on faces: -(rb_face - <rb>) g / rho_face, alpha = 1/rho_face (fm = 1),
   rb_face from the face-mean f as two-phase.h forms rho_face (so sum_faces of the force is zero) */
event acceleration (i++)
{
  face vector av = a;
  foreach_face(x)
    av.x[] += G0*(rbmean - rbuoy((f[] + f[-1])/2.))*alpha.x[]/fm.x[];
}

event logfile (t = 0; t += DTOUT; t <= TEND)
{
  double vg = 0., fu = 0., um = 0., ru = 0., rs = 0., fx = 0.;
  foreach (reduction(+:vg) reduction(+:fu) reduction(+:um) reduction(+:ru) reduction(+:rs)
	   reduction(+:fx)) {
    double r = rho(f[]);
    vg += f[]*dv(); fu += f[]*u.x[]*dv(); um += u.x[]*dv();
    ru += r*u.x[]*dv(); rs += r*dv();
    double dx = x - xcent;
    dx -= 1.6*round (dx/1.6);
    fx += f[]*dx*dv();
  }
  xcent += fx/vg;
  printf ("%.6f %.10e %.10e %.10e %.10e %.10e %d\n", t, fu/vg, um/cube(1.6), ru/rs, xcent, vg, i);
  fflush (stdout);
}

event dumpfields (t = TEND)
{
  char name[80];
  sprintf (name, "fields_N%d.bin", 1 << LEVEL);
  FILE * fp = fopen (name, "w");
  foreach (serial) {    // serial: fwrite is not thread-safe; positions written, so order does not matter
    double pos[3] = {x, y, z}, vof = f[], vel[3] = {u.x[], u.y[], u.z[]};
    fwrite (pos, sizeof(double), 3, fp);   // record: x y z f u v w
    fwrite (&vof, sizeof(double), 1, fp);
    fwrite (vel, sizeof(double), 3, fp);
  }
  fclose (fp);
}
