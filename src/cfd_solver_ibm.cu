
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

// --------------------------------------------------------
// Robust Scaled IBM Geometry Helper: Polynomials
// --------------------------------------------------------

// Table 1: Point-value schemes
__device__ inline float poly_D(float xi) { return xi * (1.0f + xi); }
__device__ inline float poly_N_nb(float xi) { return xi * (1.0f - xi); }
__device__ inline float poly_N_c(float xi) { return 2.0f * (xi * xi - 1.0f); }

// Table 2: Sandwiched (Double-sided) Point-value
// Neighbors at xi_m (minus) and xi_p (plus)
__device__ inline float poly_D_sandwich(float xi_m, float xi_p) {
  return (xi_m + xi_p) * xi_m * xi_p;
}
__device__ inline float poly_N_c_sandwich(float xi_m, float xi_p) {
  // N_{c,+} term (factor for Plus-side ghost)
  return (xi_m + xi_p) * (xi_m + 1.0f) * (xi_p - 1.0f);
}

// --------------------------------------------------------
// Update Dense Fractions Kernel
// --------------------------------------------------------
// Sets frac[idx] = 1.0 if sample_sdf(idx+offset) > 0, else 0.0
// This is required for Projection mask and Pressure Stencil base setup.
__global__ void update_fractions_kernel(float *__restrict__ frac,
                                        const float *__restrict__ sdf, int3 res,
                                        float3 offset) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  float val = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                idx_z + offset.z, sdf, res);

  // Simple binary fraction for now.
  // Future: Could compute exact line fraction for cut cells.
  frac[idx] = (val >= 0.0f) ? 1.0f : 0.0f;
}

// --------------------------------------------------------
// Compute IBM Geometry Kernel (Robust Scaled)
// --------------------------------------------------------
// --------------------------------------------------------
// Compute IBM Geometry Kernel (Robust Scaled)
// --------------------------------------------------------
// bc_type: 0 = Dirichlet (Robust Scaled Polynomials)
//          1 = Neumann (Zero Gradient: phi_g = phi_c => K=1, M=0)
__global__ void compute_ibm_geometry_kernel(IBM_Data ibm_data, int *ibm_id_map,
                                            const float *__restrict__ sdf,
                                            int3 res, float3 spacing,
                                            int *counter, float3 offset,
                                            int bc_type) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Sample SDF
  float sdf_c = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);

  // Mark solid cells with -1 and skip
  if (sdf_c < 0.0f) {
    ibm_id_map[idx] = -1;
    return;
  }

  // Identify neighbors
  float D_vals[6];
  float xi_vals[6];
  bool is_ghost[6];
  bool is_sandwich[3];

  int num_solid_neighbors = 0;

  // Check Directions
  auto check_dir = [&](int k, int dx, int dy, int dz) {
    float sdf_n =
        sample_sdf_interp(idx_x + dx + offset.x, idx_y + dy + offset.y,
                          idx_z + dz + offset.z, sdf, res);
    if (sdf_n < 0.0f) {
      is_ghost[k] = true;
      if (bc_type == 0) {
        float theta = sdf_c / (sdf_c - sdf_n);
        if (theta < 1e-4f)
          theta = 1e-4f;
        if (theta > 1.0f)
          theta = 1.0f;
        xi_vals[k] = theta;
        D_vals[k] = poly_D(theta);
      } else {
        // Neumann: D not used in same way, strictly K=1
        // But we can set dummy values
        xi_vals[k] = 0.5f;
        D_vals[k] = 1.0f;
      }
      num_solid_neighbors++;
    } else {
      is_ghost[k] = false;
      xi_vals[k] = 1.0f;
      D_vals[k] = 1e9f;
    }
  };

  check_dir(0, 1, 0, 0);  // +X
  check_dir(1, -1, 0, 0); // -X
  check_dir(2, 0, 1, 0);  // +Y
  check_dir(3, 0, -1, 0); // -Y
  check_dir(4, 0, 0, 1);  // +Z
  check_dir(5, 0, 0, -1); // -Z

  if (num_solid_neighbors == 0) {
    ibm_id_map[idx] = -1;
    return;
  }

  // Allocate
  int list_idx = atomicAdd(counter, 1);
  ibm_id_map[idx] = list_idx;
  ibm_data.cell_index[list_idx] = idx;
  ibm_data.num_boundaries[list_idx] = 6;

  // Dirichlet (Robust Scaled) Calculation
  if (bc_type == 0) {
    is_sandwich[0] = is_ghost[0] && is_ghost[1];
    is_sandwich[1] = is_ghost[2] && is_ghost[3];
    is_sandwich[2] = is_ghost[4] && is_ghost[5];

    float D_sandwich[3] = {0, 0, 0};
    if (is_sandwich[0])
      D_sandwich[0] = poly_D_sandwich(xi_vals[1], xi_vals[0]);
    if (is_sandwich[1])
      D_sandwich[1] = poly_D_sandwich(xi_vals[3], xi_vals[2]);
    if (is_sandwich[2])
      D_sandwich[2] = poly_D_sandwich(xi_vals[5], xi_vals[4]);

    float min_D_abs = 1e30f;
    float D_rescale = 1.0f;
    auto update_min = [&](float val) {
      if (fabsf(val) < min_D_abs) {
        min_D_abs = fabsf(val);
        D_rescale = val;
      }
    };

    if (is_sandwich[0])
      update_min(D_sandwich[0]);
    else {
      if (is_ghost[0])
        update_min(D_vals[0]);
      if (is_ghost[1])
        update_min(D_vals[1]);
    }
    if (is_sandwich[1])
      update_min(D_sandwich[1]);
    else {
      if (is_ghost[2])
        update_min(D_vals[2]);
      if (is_ghost[3])
        update_min(D_vals[3]);
    }
    if (is_sandwich[2])
      update_min(D_sandwich[2]);
    else {
      if (is_ghost[4])
        update_min(D_vals[4]);
      if (is_ghost[5])
        update_min(D_vals[5]);
    }

    ibm_data.D_rescale[list_idx] = D_rescale;

    for (int axis = 0; axis < 3; axis++) {
      int km = 2 * axis + 1;
      int kp = 2 * axis;

      float D_axis = 0.0f;
      bool sandwich = is_sandwich[axis];
      bool g_p = is_ghost[kp];
      bool g_m = is_ghost[km];

      if (sandwich)
        D_axis = D_sandwich[axis];
      else if (g_p)
        D_axis = D_vals[kp];
      else if (g_m)
        D_axis = D_vals[km];
      else
        D_axis = D_rescale;

      float R = D_rescale / D_axis;
      if (fabsf(D_axis) < 1e-9f)
        R = 1.0f;
      ibm_data.R_val[list_idx * 6 + kp] = R;
      ibm_data.R_val[list_idx * 6 + km] = R;

      if (sandwich) {
        float N_c_plus = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]);
        ibm_data.K_val[list_idx * 6 + kp] = N_c_plus * R;
        ibm_data.M_val[list_idx * 6 + kp] = 0.0f;
        ibm_data.X_val[list_idx * 6 + kp] = 0.0f;
        ibm_data.B_val[list_idx * 6 + kp] = 0.0f;

        float N_c_minus = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]);
        ibm_data.K_val[list_idx * 6 + km] = N_c_minus * R;
        ibm_data.M_val[list_idx * 6 + km] = 0.0f;
        ibm_data.X_val[list_idx * 6 + km] = 0.0f;
        ibm_data.B_val[list_idx * 6 + km] = 0.0f;
      } else {
        if (is_ghost[kp]) {
          ibm_data.K_val[list_idx * 6 + kp] =
              poly_N_c(xi_vals[kp]) * R;
          ibm_data.M_val[list_idx * 6 + kp] = 0.0f;
          ibm_data.X_val[list_idx * 6 + kp] =
              poly_N_nb(xi_vals[kp]) * R;
          ibm_data.B_val[list_idx * 6 + kp] = 0.0f;
        } else {
          ibm_data.K_val[list_idx * 6 + kp] = 0.0f;
          ibm_data.M_val[list_idx * 6 + kp] = 1.0f;
          ibm_data.X_val[list_idx * 6 + kp] = 0.0f;
          ibm_data.B_val[list_idx * 6 + kp] = 0.0f;
        }
        if (is_ghost[km]) {
          ibm_data.K_val[list_idx * 6 + km] =
              poly_N_c(xi_vals[km]) * R;
          ibm_data.M_val[list_idx * 6 + km] = 0.0f;
          ibm_data.X_val[list_idx * 6 + km] =
              poly_N_nb(xi_vals[km]) * R;
          ibm_data.B_val[list_idx * 6 + km] = 0.0f;
        } else {
          ibm_data.K_val[list_idx * 6 + km] = 0.0f;
          ibm_data.M_val[list_idx * 6 + km] = 1.0f;
          ibm_data.X_val[list_idx * 6 + km] = 0.0f;
          ibm_data.B_val[list_idx * 6 + km] = 0.0f;
        }
      }
      ibm_data.dir_code[list_idx * 6 + kp] = kp;
      ibm_data.dir_code[list_idx * 6 + km] = km;
    }

  } else {
    // Neumann (Simple K=1, M=0)
    ibm_data.D_rescale[list_idx] = 1.0f;
    for (int k = 0; k < 6; k++) {
      ibm_data.dir_code[list_idx * 6 + k] = k;
      ibm_data.R_val[list_idx * 6 + k] = 1.0f;
      if (is_ghost[k]) {
        // Ghost: phi_g = phi_c
        // Term in stencil: A_nb * phi_g = A_nb * phi_c.
        // Modify: A_c += A_nb * 1. A_nb *= 0.
        ibm_data.K_val[list_idx * 6 + k] = 1.0f;
        ibm_data.M_val[list_idx * 6 + k] = 0.0f;
        ibm_data.X_val[list_idx * 6 + k] = 0.0f;
        ibm_data.B_val[list_idx * 6 + k] = 0.0f;
      } else {
        // Fluid: No Change
        ibm_data.K_val[list_idx * 6 + k] = 0.0f;
        ibm_data.M_val[list_idx * 6 + k] = 1.0f;
        ibm_data.X_val[list_idx * 6 + k] = 0.0f;
        ibm_data.B_val[list_idx * 6 + k] = 0.0f;
      }
    }
  }
}

// --------------------------------------------------------
// Ghost Cell Extrapolation Kernel
// --------------------------------------------------------
// --------------------------------------------------------
// Ghost Cell Extrapolation Kernel
// --------------------------------------------------------
__global__ void populate_ghost_cells_kernel(float *u, float *v, float *w,
                                            int3 res, IBM_Data ibm_data,
                                            int num_ibm_cells) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_ibm_cells)
    return;

  // Placeholder: robust scheme usually eliminates ghost cells from the system.
  // Visualization might be incorrect at boundaries but simulation is safe.
  // For now we do nothing or could set to 0.
}

// --------------------------------------------------------
// Modify Stencil IBM Kernel (The "Bake-in" Step)
// --------------------------------------------------------
// Applies the geometric factors to the global stencil arrays
// A_c, A_nb, RHS
__global__ void modify_stencil_ibm_kernel(
    float *__restrict__ A_C, float *__restrict__ A_W, float *__restrict__ A_E,
    float *__restrict__ A_S, float *__restrict__ A_N, float *__restrict__ A_B,
    float *__restrict__ A_T, float *__restrict__ B_RHS, IBM_Data ibm_data) {

  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm_data.num_active_cells)
    return;

  int c_idx = ibm_data.cell_index[list_idx];
  float descale = ibm_data.D_rescale[list_idx];

  // 1. Scale Center & RHS
  // Atomic? Multiple threads might update same cell if accessing neighbors,
  // but here we update the cell c itself uniquely (one thread per IBM cell).
  // So updating A_C[c] and B_RHS[c] is safe.

  // Save original coefficient values BEFORE any modifications
  // This prevents cascading corruption in the direction loop
  float orig_AE = A_E[c_idx];
  float orig_AW = A_W[c_idx];
  float orig_AN = A_N[c_idx];
  float orig_AS = A_S[c_idx];
  float orig_AT = A_T[c_idx];
  float orig_AB = A_B[c_idx];
  float orig_vals[6] = {orig_AE, orig_AW, orig_AN, orig_AS, orig_AT, orig_AB};

  A_C[c_idx] *= descale;
  B_RHS[c_idx] *= descale;

  // Initialize neighbor modifications (will be accumulated)
  float mod_AE = 0.0f, mod_AW = 0.0f, mod_AN = 0.0f;
  float mod_AS = 0.0f, mod_AT = 0.0f, mod_AB = 0.0f;

  // 2. Compute all modifications using ORIGINAL values
  for (int k = 0; k < 6; k++) {
    int entry = list_idx * 6 + k;

    float K = ibm_data.K_val[entry];
    float M = ibm_data.M_val[entry];
    float X = ibm_data.X_val[entry];
    float B = ibm_data.B_val[entry];

    float val_nb = orig_vals[k];

    // Update Center: A_c += A_nb * K (using original A_nb)
    A_C[c_idx] += val_nb * K;

    // Update RHS: b_c += B
    B_RHS[c_idx] += B;

    // Accumulate neighbor modifications
    // M multiplies the neighbor in direction k
    // X adds to the opposite neighbor
    switch (k) {
    case 0: // East
      mod_AE += orig_AE * (descale * M - 1.0f); // Apply row scaling + mask
      mod_AW += orig_AE * X;                    // Cross term to West
      break;
    case 1: // West
      mod_AW += orig_AW * (descale * M - 1.0f);
      mod_AE += orig_AW * X;
      break;
    case 2: // North
      mod_AN += orig_AN * (descale * M - 1.0f);
      mod_AS += orig_AN * X;
      break;
    case 3: // South
      mod_AS += orig_AS * (descale * M - 1.0f);
      mod_AN += orig_AS * X;
      break;
    case 4: // Top
      mod_AT += orig_AT * (descale * M - 1.0f);
      mod_AB += orig_AT * X;
      break;
    case 5: // Bottom
      mod_AB += orig_AB * (descale * M - 1.0f);
      mod_AT += orig_AB * X;
      break;
    }
  }

  // Apply accumulated modifications
  A_E[c_idx] = orig_AE + mod_AE;
  A_W[c_idx] = orig_AW + mod_AW;
  A_N[c_idx] = orig_AN + mod_AN;
  A_S[c_idx] = orig_AS + mod_AS;
  A_T[c_idx] = orig_AT + mod_AT;
  A_B[c_idx] = orig_AB + mod_AB;
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
//   offset: {0,0,0} for Center/Vol, {-0.5,0,0} for U-Face, etc.
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
