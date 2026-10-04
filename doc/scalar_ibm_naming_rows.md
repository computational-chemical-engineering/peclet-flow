# Proposed NAMING rows: cut-cell scalar transport (WO-10, 2026-10-04)

This is PROPOSED text for the `### peclet.flow` table of `suite/docs/NAMING.md` §2, to be placed when
branch `scalar-ibm` lands. The umbrella is a shared checkout on `main` and is not edited from this
branch. Delete this file once the rows are placed.

**Every name below is unreleased.** None has shipped in a release, so a rename now is free: no alias
and no deprecation ladder (NAMING §0), and that holds for keyword arguments too (§0.1). **No code is
renamed in this work order.** A row marked PENDING waits for Frank's naming decision, together with
§13 Q5 (the `cutcell=` spelling).

**Checked against the canon:**
- §1.2: a stored value is a property; a value computed on call is a bare-name method; an array
  copied out takes `get_`;
- §1.3: counts are `num_<plural>`; dictionary keys are exempt but follow the canon where new;
- §1.6: American spelling;
- the flow conventions of the two tiers: strings for enumerations, and `instance=` as in
  `set_instance_motion`;
- one spelling per concept across the existing flow surface.

## Rows for `### peclet.flow`

| former | canonical | status |
|---|---|---|
| `add_scalar(name, diffusivity, scheme, iters=None, cutcell=False)` | — | **canon**, new keyword (scalar-ibm). `cutcell=` follows `set_solid(..., cutcell_pressure=)`. `iters` keeps its name; its default became `None` (= the legacy 50; it must stay `None` when `cutcell=True`). **PENDING Frank: the spelling of the opt-in itself (design §13 Q5).** |
| `set_scalar_bc(name, face, type, value)` with `value` a per-face profile `(N_t1, N_t2)` | — | **canon**, new overload (scalar-ibm), cut-cell scalars only. Shape in the face's tangential axes in x, y, z order, this rank's slice, as `set_domain_bc_profile`. |
| `set_scalar_wall(name, type, value=0.0, coefficient=0.0, instance=None)` | — | **canon**, new (scalar-ibm). `type` is `'neumann'`, `'dirichlet'` or `'robin'`; `coefficient` is the Robin k (L/T). |
| `set_scalar_solid(name, diffusivity, capacity=1.0, partition=1.0, contact_resistance=0.0, instance=None)` | — | **canon**, new (scalar-ibm): D_s, C_s, K, R_c, all physical. |
| `get_scalar_solid(name)` | — | **canon**, new (scalar-ibm): an array copied out (§1.2), c_s = Kψ_s, NaN outside solid unknowns. |
| `set_scalar_source(name, source)` (float or `(nx, ny, nz)` array) | — | **canon**, new (scalar-ibm). |
| `set_scalar_mean_gradient(name, gradient)` (3-sequence) | — | **canon**, new (scalar-ibm). |
| `scalar_mean_flux(name)` → `(3,)` | — | **canon**, new (scalar-ibm): computed on call, bare-name method (§1.2). The array-valued precedent is `hydro_force_torque_reaction()`. |
| `scalar_wall_flux(name)` → `(num_bodies,)` | — | **canon**, new (scalar-ibm): computed (a reduction), bare-name method. |
| `solve_scalar_steady(name)` | — | **canon**, new (scalar-ibm): a verb. |
| `set_scalar_tolerance(name, rtol)` | `set_scalar_residual_tolerance(name, rtol)` | **PENDING Frank: possible divergence.** The criterion, max\|b − A c\| ≤ rtol · max(max\|b\|, max\|A c\|), is exactly that of `set_velocity_residual_tolerance(rtol)` (design §5.1: "the velocity solver's form"), so one concept has two spellings. The other precedent, `set_outer_tolerance`, names a different quantity. Proposed: rename to the velocity spelling while still unreleased. Keep it only if Frank prefers the shorter name. |
| `diagnostics.set_scalar_max_iterations(name, maxit)` | `diagnostics.set_scalar_max_iterations(name, max_iter)` | **PENDING Frank: argument divergence.** The function name is fine. Its argument `maxit` is the only one in flow's bindings. The Krylov iteration cap is `max_iter` in the three pressure drivers (`set_pressure_pcg` / `_fcg` / `_chebyshev(on, max_iter, rtol)`); `iters` is a sweep count (`set_*_solver_params`, the legacy `add_scalar(iters=)`). Proposed: `max_iter` (§0.1: free to rename before release). |
| `diagnostics.scalar_census(name)` → dict | — | **canon**, new (diagnostics `*_census` family). |
| `diagnostics.scalar_budget(name)` → dict | — | **canon**, new (diagnostics `*_budget` family). |
| `diagnostics.scalar_facets(name)` → dict of per-facet arrays | — | **canon**, new (diagnostics tier, dict-returning like `vof_diagnostics()`). |
| `diagnostics.scalar_geometry(name)` → dict of `(nx, ny, nz)` F-order arrays | — | **canon**, new (diagnostics tier). |
| census keys `num_unknowns`, `num_solid_unknowns`, `num_cut_cells`, `num_facets`, `num_two_sided`, `num_thin_solid`, `num_sealed`, `num_small_cells`, `num_implicit_faces`, `num_flux_faces`, `num_guarded_flux_faces` | — | **canon** (§1.3 spelling, although keys are exempt). |
| census keys `sealed_volume`, `solid_volume`, `sealed_solid_volume` (physical), `probe_rungs` (`{'fluid': (R0, R1a, R1b, R2), 'solid': …}`), `bulk_courant`, `max_cell_peclet`, `steady_incompatibility`, `krylov_iterations`, `krylov_residual`, `krylov_converged` | — | **canon** (data keys). |
| census key `mg_levels` | `num_mg_levels` | **PENDING Frank (minor).** It is a count, so §1.3 would spell it `num_mg_levels`. Dictionary keys are exempt (§1.3), and nothing outside the branch reads this one yet. Proposed: rename now while free, or record it as an exempt key. |
| budget keys `mass`, `mass_solid`, `d_mass`, `wall_in`, `boundary_in`, `source_in`, `defect`, `identity_error` | — | **canon** (data keys). |
| facet keys `centroid` (physical), `normal`, `area`, `instance`, `wall_value`, `flux`, `probe_rung` | — | **canon** (data keys; American `centroid`). |
| geometry keys `kappa`, `aperture_x`, `aperture_y`, `aperture_z`, `unknown`, `kappa_solid`, `solid_unknown` | — | **canon** (data keys; one name per axis, as `get_ox` / `get_oy` / `get_oz`). |
| registered field `<name>_solid` (ψ_s, internal), via `get_field` / `field_view` | — | **canon** (the registry convention). It holds ψ_s = c_s/K; the converted copy is `get_scalar_solid`. |

**Summary:** 3 PENDING rows (`set_scalar_tolerance`, the `maxit` argument, the `mg_levels` key), plus
Q5 on `cutcell=`. Every other new name follows the canon as it stands.
