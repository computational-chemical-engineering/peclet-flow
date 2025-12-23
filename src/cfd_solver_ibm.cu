
#include "cfd_solver.cuh"
#include <cstdio>

// --------------------------------------------------------
// IBM Geometry Helper: Compute Coefficients
// --------------------------------------------------------
// phi_c: SDF at center (should be > 0)
// phi_n: SDF at neighbor (should be < 0)
// inv_h2: 1 / (spacing * spacing)
// delta_diag: Output, contribution to diagonal (to be ADDED)
// rhs_factor: Output, factor for boundary value (to be ADDED to RHS)
__device__ void get_ibm_coeffs(float phi_c, float phi_n, float inv_h2,
                               float &delta_diag, float &rhs_factor) {
  float theta = phi_c / (phi_c - phi_n);
  // Clamp theta to avoid division by zero (e.g. 0.01 to 0.999)
  if (theta < 1e-4f)
    theta = 1e-4f;

  // u_ghost = u_bc / theta - u_c * (1/theta - 1)
  // Contribution to Laplacian (d^2u/dx^2 term at Center):
  // Standard: (u_E + u_W - 2u_C) * inv_h2
  // Replaced neighbor (u_W becomes ghost): (u_bc / theta - u_c * (1/theta - 1))
  // * inv_h2 New term in sum: - u_c * (1/theta - 1) * inv_h2 Factor of u_C
  // changes by: -(1/theta - 1) * inv_h2 Since Diagonal in LHS ( Au=b ) collects
  // -coeffs of u_C: Old Diag Contribution: -(-2 * inv_h2) = +2 * inv_h2 ? Wait,
  // the standard diagonal element A_ii is -2/dx^2 - 2/dy^2 ... If we move
  // diagonal to RHS for GS update: u_C = (sum - rhs) / diag. Here Diag is
  // usually Positive (Sum of 1/h^2). Let's assume Diag = Sum(neighbors coeffs).
  // Standard neighbor coeff is 1*inv_h2.
  // Ghost neighbor coeff becomes -(1/theta - 1)*inv_h2.
  // So we subtract 1*inv_h2 (remove standard neighbor) and add -(1/theta -
  // 1)*inv_h2. Net change to Neighbor Sum part: - inv_h2 - (1/theta - 1)*inv_h2
  // = -inv_h2 * (1 + 1/theta - 1) = -inv_h2 / theta.
  //
  // EQUIVALENT VIEW:
  // Modify Diagonal A_ii?
  // A_ii (standard) = -2 * inv_h2.
  // New A_ii = -1 * inv_h2 (one standard neighbor) - (1/theta - 1) * inv_h2 ?
  // No. The eqn is: (u_E + u_ghost - 2u_C)*inv_h2 = u_E*inv_h2 + [u_bc/theta -
  // u_C(1/theta - 1)]*inv_h2 - 2u_C*inv_h2 = u_E*inv_h2 + u_bc*inv_h2/theta -
  // u_C*inv_h2*(1/theta - 1 + 2) = u_E*inv_h2 + u_bc*inv_h2/theta -
  // u_C*inv_h2*(1/theta + 1) Coeff of u_C is -(1/theta + 1)*inv_h2. Standard
  // Coeff was -2*inv_h2. Change is: -(1/theta + 1) - (-2) = 1 - 1/theta.
  //
  // So if solving A u = b, Diagonal A_ii changes by (1 - 1/theta)*inv_h2.
  //
  // If Using RB-GS Update rule: u_C = (Extension terms) / Diag_Pos.
  // Diag_Pos (Standard) = 2*inv_h2.
  // Diag_Pos (New) = (1/theta + 1)*inv_h2.
  // Delta Diag = (1/theta - 1)*inv_h2.

  delta_diag = (1.0f / theta - 1.0f) * inv_h2;
  rhs_factor = (1.0f / theta) * inv_h2;
}

struct IBM_Direction_Local {
  float D;
  float N_C;
  float N_bc;
  float val_bc;
  int nb_idx;
};

// Trilinear Interpolation of periodic SDF
__device__ float sample_sdf_interp(float x, float y, float z,
                                   const float *__restrict__ sdf, int3 res) {
  // 1. Floor to get base index
  float fx = floorf(x);
  float fy = floorf(y);
  float fz = floorf(z);

  // 2. Fraction
  float wx = x - fx;
  float wy = y - fy;
  float wz = z - fz;

  // 3. Base integer indices (wrapped)
  int ix = (int)fx;
  int iy = (int)fy;
  int iz = (int)fz;

  // 4. Neighbors
  int x0 = (ix % res.x + res.x) % res.x;
  int y0 = (iy % res.y + res.y) % res.y;
  int z0 = (iz % res.z + res.z) % res.z;

  int x1 = (x0 + 1) % res.x;
  int y1 = (y0 + 1) % res.y;
  int z1 = (z0 + 1) % res.z;

  // 5. Fetch 8 corners
  // Order: 000, 100, 010, 110, 001, 101, 011, 111
  float c000 = sdf[z0 * res.y * res.x + y0 * res.x + x0];
  float c100 = sdf[z0 * res.y * res.x + y0 * res.x + x1];
  float c010 = sdf[z0 * res.y * res.x + y1 * res.x + x0];
  float c110 = sdf[z0 * res.y * res.x + y1 * res.x + x1];
  float c001 = sdf[z1 * res.y * res.x + y0 * res.x + x0];
  float c101 = sdf[z1 * res.y * res.x + y0 * res.x + x1];
  float c011 = sdf[z1 * res.y * res.x + y1 * res.x + x0];
  float c111 = sdf[z1 * res.y * res.x + y1 * res.x + x1];

  // 6. Interpolate (Lerp X, then Y, then Z)
  float c00 = c000 * (1.0f - wx) + c100 * wx;
  float c10 = c010 * (1.0f - wx) + c110 * wx;
  float c01 = c001 * (1.0f - wx) + c101 * wx;
  float c11 = c011 * (1.0f - wx) + c111 * wx;

  float c0 = c00 * (1.0f - wy) + c10 * wy;
  float c1 = c01 * (1.0f - wy) + c11 * wy;

  return c0 * (1.0f - wz) + c1 * wz;
}

__global__ void compute_ibm_geometry_kernel(IBM_Data ibm_data, int *ibm_id_map,
                                            const float *__restrict__ sdf,
                                            int3 res, float3 spacing,
                                            int *counter, float3 offset) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Sample SDF at the staggered location (idx + offset)
  // Note: idx_x is integer. We pass floats to sample_sdf_interp.
  float sdf_c = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);

  // Only process Fluid cells
  if (sdf_c < 0.0f) {
    ibm_id_map[idx] =
        -1; // Solid (inside SDF < 0 usually means outside? No.
            // Wait. Usually SDF>0 is fluid, SDF<0 is solid?
            // Let's check existing code:
            // line 80: if (sdf_c < 0.0f) { ibm_id_map[idx] = -1; return; }
            // This suggests sdf_c < 0 is "Standard/Nothing to do" or "Solid"?
            // The comment says "Only process Fluid cells".
            // If (sdf_c < 0.0f) return... implies sdf < 0 is NOT fluid
            // boundary? In many codes SDF>0 is fluid. Or SDF<0 is fluid (inside
            // object is +?). Let's look at `get_ibm_coeffs`: `theta = phi_c /
            // (phi_c - phi_n)`. If phi_c > 0 and phi_n < 0 -> crossing. So
            // phi_c must be fluid (>0) and phi_n solid (<0). So line 80 `if
            // (sdf_c < 0) return` is conserving the logic "If I am solid,
            // skip". Yes.
    ibm_id_map[idx] = -1;
    return;
  }

  IBM_Direction_Local local_dirs[6];
  int num_boundaries = 0;

  float inv_dx2 = 1.0f / (spacing.x * spacing.x);
  float inv_dy2 = 1.0f / (spacing.y * spacing.y);
  float inv_dz2 = 1.0f / (spacing.z * spacing.z);

  float total_delta_diag = 0.0f;

  auto check_dir = [&](int dx, int dy, int dz, float inv_h2, int dir_code) {
    // Neighbor coordinate in integer grid
    // But we need SDF at neighbor staggered location:
    // (idx_x + dx) + offset.x

    // We utilize dx, dy, dz as integer steps (+1 / -1)

    float sdf_n =
        sample_sdf_interp(idx_x + dx + offset.x, idx_y + dy + offset.y,
                          idx_z + dz + offset.z, sdf, res);

    if (sdf_n < 0.0f) { // Neighbor is solid
      float d_diag, r_fac;
      get_ibm_coeffs(sdf_c, sdf_n, inv_h2, d_diag, r_fac);
      local_dirs[num_boundaries++] = {d_diag, 0.0f, r_fac, 0.0f, dir_code};
      total_delta_diag += d_diag;
    }
  };

  // +X
  check_dir(1, 0, 0, inv_dx2, 0);
  // -X
  check_dir(-1, 0, 0, inv_dx2, 1);
  // +Y
  check_dir(0, 1, 0, inv_dy2, 2);
  // -Y
  check_dir(0, -1, 0, inv_dy2, 3);
  // +Z
  check_dir(0, 0, 1, inv_dz2, 4);
  // -Z
  check_dir(0, 0, -1, inv_dz2, 5);

  if (num_boundaries > 0) {
    int list_idx = atomicAdd(counter, 1);

    float base_diag = 2.0f * (inv_dx2 + inv_dy2 + inv_dz2);
    float total_diag = base_diag + total_delta_diag;

    // Write to SoA
    ibm_data.cell_index[list_idx] = idx;
    ibm_data.S_row[list_idx] = 1.0f / total_diag;
    ibm_data.num_boundaries[list_idx] = num_boundaries;

    for (int k = 0; k < num_boundaries; k++) {
      int entry = 6 * list_idx + k;
      ibm_data.N_bc[entry] = local_dirs[k].N_bc;
      ibm_data.val_bc[entry] = local_dirs[k].val_bc;
      ibm_data.nb_idx[entry] = local_dirs[k].nb_idx;
    }
    ibm_id_map[idx] = list_idx;
  } else {
    ibm_id_map[idx] = -1;
  }
}

// --------------------------------------------------------
// Ghost Cell Extrapolation Kernel
// --------------------------------------------------------
__global__ void populate_ghost_cells_kernel(float *u, float *v, float *w,
                                            int3 res, IBM_Data ibm_data,
                                            int num_ibm_cells) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_ibm_cells)
    return;

  int c_idx = ibm_data.cell_index[idx];
  int num_b = ibm_data.num_boundaries[idx];

  // Decompose linear index
  // res must be passed correctly
  int z = c_idx / (res.x * res.y);
  int rem = c_idx % (res.x * res.y);
  int y = rem / res.x;
  int x = rem % res.x;

  float u_c = u[c_idx];
  float v_c = v[c_idx];
  float w_c = w[c_idx];

  for (int k = 0; k < num_b; k++) {
    int entry = 6 * idx + k;
    int dir_code = ibm_data.nb_idx[entry];

    // Mirror fallback (theta = 0.5)
    float theta = 0.5f;

    int nx = x, ny = y, nz = z;
    if (dir_code == 0)
      nx += 1;
    else if (dir_code == 1)
      nx -= 1;
    else if (dir_code == 2)
      ny += 1;
    else if (dir_code == 3)
      ny -= 1;
    else if (dir_code == 4)
      nz += 1;
    else if (dir_code == 5)
      nz -= 1;

    int n_idx = get_idx(nx, ny, nz, res);
    float val_bc = ibm_data.val_bc[entry];

    u[n_idx] = (val_bc - u_c * (1.0f - theta)) / theta;
    v[n_idx] = (val_bc - v_c * (1.0f - theta)) / theta;
    w[n_idx] = (val_bc - w_c * (1.0f - theta)) / theta;
  }
}

// --------------------------------------------------------
// IBM-RBGS Solver Helper
// --------------------------------------------------------
// Logic:
// Standard update: u_new = (rhs + sum(coeff*u_neigh)) / diag
// IBM update:
// 1. Scale Row: factor = S_row * diag (Standard).
//    Wait, S_row is "1 / Total_Diag".
//    Total_Diag = Standard_Diag + Sum(Delta_Diag).
//    So new Diag_eff = 1.0 / S_row.
//
// 2. Adjust Neighbors:
//    We DROP the solid neighbor (term = 0).
//    (Handled by implicit masking or checking sdf in main kernel).
//
// 3. Adjust RHS:
//    RHS_eff = RHS + Sum(N_bc * val_bc).
//
// 4. Update Rule:
//    u_new = (RHS_eff + Sum_Fluid_Neighbors) * S_row.
//
// Function returns the modified update value

// --------------------------------------------------------
// Fluid Volume/Area Fraction Kernel
// --------------------------------------------------------
// Computes fractions based on User Formula:
// Vf = clamp(0.5 + phi / L_proj, 0, 1)
//
// Type 0: Volume (L_proj = |nx|dx + |ny|dy + |nz|dz)
// Type 1: Area X (L_proj = |ny|dy + |nz|dz)
// Type 2: Area Y (L_proj = |nx|dx + |nz|dz)
// Type 3: Area Z (L_proj = |nx|dx + |ny|dy)
//
// Input:
//   sdf: Centered SDF Field
//   res: Grid Resolution
//   spacing: Grid Spacing (dx, dy, dz)
//   offset: {0,0,0} for Center/Vol, {0.5,0,0} for U-Face, etc.
//   type: 0..3
// Output:
//   fractions: Output array [num_elements]
//
__global__ void compute_fluid_fraction_kernel(const float *__restrict__ sdf,
                                              float *fractions, int3 res,
                                              float3 spacing, float3 offset,
                                              int type) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // 1. Sample SDF at location (Center or Face)
  float phi = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                idx_z + offset.z, sdf, res);

  // 2. Compute normal components |n|*delta
  // We approximate |n_x|*dx with |d_phi/dx| * dx?
  // User Formula: |n_x| * dx.
  // n = grad(phi) / |grad(phi)|.
  // grad_x = (phi(x+h) - phi(x-h)) / 2h.
  // n_x * dx = (grad_x / |grad|) * dx.
  //
  // Alternative interpretation: "Compact formulation" usually uses
  // |grad_x| * dx directly if |grad| is normalized?
  // Or maybe |n_x| implies unit normal.
  //
  // Let's compute Gradient vector G.
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  float epsilon = 1.0f; // 1 grid cell spacing for finite difference step

  float phi_xp = sample_sdf_interp(idx_x + offset.x + epsilon, idx_y + offset.y,
                                   idx_z + offset.z, sdf, res);
  float phi_xm = sample_sdf_interp(idx_x + offset.x - epsilon, idx_y + offset.y,
                                   idx_z + offset.z, sdf, res);
  gx = (phi_xp - phi_xm) / (2.0f * spacing.x);

  float phi_yp = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y + epsilon,
                                   idx_z + offset.z, sdf, res);
  float phi_ym = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y - epsilon,
                                   idx_z + offset.z, sdf, res);
  gy = (phi_yp - phi_ym) / (2.0f * spacing.y);

  float phi_zp = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                   idx_z + offset.z + epsilon, sdf, res);
  float phi_zm = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                   idx_z + offset.z - epsilon, sdf, res);
  gz = (phi_zp - phi_zm) / (2.0f * spacing.z);

  // Magnitude of Gradient
  float g_mag = sqrtf(gx * gx + gy * gy + gz * gz);
  if (g_mag < 1e-6f)
    g_mag = 1e-6f; // Avoid division by zero

  // Normalize Normal
  float nx = gx / g_mag;
  float ny = gy / g_mag;
  float nz = gz / g_mag;

  // 3. Compute Denominator
  float denom = 0.0f;
  if (type == 0) { // Volume
    denom =
        fabsf(nx) * spacing.x + fabsf(ny) * spacing.y + fabsf(nz) * spacing.z;
  } else if (type == 1) { // Area X (U)
    denom = fabsf(ny) * spacing.y + fabsf(nz) * spacing.z;
  } else if (type == 2) { // Area Y (V)
    denom = fabsf(nx) * spacing.x + fabsf(nz) * spacing.z;
  } else if (type == 3) { // Area Z (W)
    denom = fabsf(nx) * spacing.x + fabsf(ny) * spacing.y;
  }

  // Avoid singularity if denom is tiny (e.g. flat region aligned with grid?)
  // If denom ~ 0, imply very sharp interface or nothing?
  // clamp(0.5 + phi/epsilon) -> Heaviside.
  if (denom < 1e-9f)
    denom = 1e-9f;

  float fraction = 0.5f + phi / denom;

  // Clamp
  if (fraction < 0.0f)
    fraction = 0.0f;
  if (fraction > 1.0f)
    fraction = 1.0f;

  fractions[idx] = fraction;
}
