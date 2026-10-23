# Case A in Basilisk (cross-check; validation/BASILISK_XCHECK.md)
Innocenti et al. 2021 published no script; `array.c` re-creates their stated set-up (centered + two-phase +
tension, periodic 1.6 d cube, uniform multigrid, fluid at rest, body force -<rho> g). Basilisk: basilisk.fr
tarball of 2026-10-23, `src/config.gcc`.
    qcc -O2 -fopenmp [-DMU_RATIO=1 | -DRHO_RATIO=1] array.c -o arr -lm
    ./runall.sh arr TAG LEVEL TOL THREADS     # LEVEL 4/5/6 = d/h 10/20/40; prints post.py's summary line
`post.py` applies run_case_a.py's definitions (drift Re, last-2L window, centroid Re).
