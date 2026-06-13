/// @file
/// @brief pnm_backend: immersed-boundary (cut-cell) assembly for the momentum solve.
#include "cfd_solver.cuh"
#include "cfd_solver_ibm_kernels.cuh"
#include <cstdio>

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
__global__ void
modify_stencil_ibm_kernel(float *__restrict__ A_C, float *__restrict__ A_W,
                          float *__restrict__ A_E, float *__restrict__ A_S,
                          float *__restrict__ A_N, float *__restrict__ A_B,
                          float *__restrict__ A_T, float *__restrict__ a_inhom,
                          IBM_Data ibm_data, float u_bc_val) {

  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm_data.num_active_cells)
    return;

  int c_idx = ibm_data.cell_index[list_idx];
  float descale = ibm_data.D_rescale[list_idx];

  // 1. Scale Center
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

  // Initialize neighbor modifications (will be accumulated)
  float mod_AE = 0.0f, mod_AW = 0.0f, mod_AN = 0.0f;
  float mod_AS = 0.0f, mod_AT = 0.0f, mod_AB = 0.0f;
  float inhom_accum = 0.0f;

  // 2. Compute all modifications using ORIGINAL values
  for (int k = 0; k < 6; k++) {
    int entry = list_idx * 6 + k;

    float K = ibm_data.K_val[entry];
    float M = ibm_data.M_val[entry];
    float X = ibm_data.X_val[entry];
    float Nbc = ibm_data.Nbc_val[entry];

    float val_nb = orig_vals[k];

    // Update Center: A_c += A_nb * K (using original A_nb)
    A_C[c_idx] += val_nb * K;

    // Calculate Inhomogeneous Term Correction
    // Was: B_RHS[c_idx] -= Nbc * u_bc_val * val_nb;
    // Now: a_inhom (term to be subtracted from B) = sum(Nbc * u_bc * a_nb)
    if (a_inhom != nullptr) {
      inhom_accum += Nbc * u_bc_val * val_nb;
    }

    // Accumulate neighbor modifications
    // M multiplies the neighbor in direction k
    // X adds to the opposite neighbor
    switch (k) {
    case 0:                                     // East
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

  // Output Inhomogeneous Term
  if (a_inhom != nullptr) {
    // Accumulate into the buffer (assuming initialized to 0 by caller)
    // Atomic not needed if one thread per cell
    // But verify: ONE list_idx maps to ONE c_idx. Yes.
    // However, multiple ibm_data entries for the SAME c_idx?
    // No, ibm_data is 1-to-1 with cut cells.
    a_inhom[c_idx] += inhom_accum;
  }
}

// --------------------------------------------------------
// Apply IBM Scaling Kernel (Standalone)
// --------------------------------------------------------
// Applies D_rescale Factor to a vector field
__global__ void apply_ibm_scaling_kernel(float *vector, IBM_Data ibm_data,
                                         int num_elements) {
  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm_data.num_active_cells)
    return;

  int c_idx = ibm_data.cell_index[list_idx];
  float descale = ibm_data.D_rescale[list_idx];

  vector[c_idx] *= descale;
}

// --------------------------------------------------------
// Fluid Volume/Area Fraction Kernel
// --------------------------------------------------------
// Computes fractions based on User Formula:
// Vf = clamp(0.5 + phi / L_proj, 0, 1)
// Type 0: Volume (L_proj = |nx|dx + |ny|dy + |nz|dz)
// Type 1: Area X (L_proj = |ny|dy + |nz|dz)
// Type 2: Area Y (L_proj = |nx|dx + |nz|dz)
// Type 3: Area Z (L_proj = |nx|dx + |ny|dy)
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
  float sd = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                               idx_z + offset.z, sdf, res);

  // 2. Compute normal components |n|*delta
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  float epsilon = 1.0f; // 1 grid cell spacing for finite difference step

  float sd_xp = sample_sdf_interp(idx_x + offset.x + epsilon, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);
  float sd_xm = sample_sdf_interp(idx_x + offset.x - epsilon, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);
  gx = (sd_xp - sd_xm) / (2.0f * spacing.x);

  float sd_yp = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y + epsilon,
                                  idx_z + offset.z, sdf, res);
  float sd_ym = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y - epsilon,
                                  idx_z + offset.z, sdf, res);
  gy = (sd_yp - sd_ym) / (2.0f * spacing.y);

  float sd_zp = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z + epsilon, sdf, res);
  float sd_zm = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z - epsilon, sdf, res);
  gz = (sd_zp - sd_zm) / (2.0f * spacing.z);

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

  if (denom < 1e-9f)
    denom = 1e-9f;

  float fraction = 0.5f + sd / denom;

  // Clamp
  if (fraction < 0.0f)
    fraction = 0.0f;
  if (fraction > 1.0f)
    fraction = 1.0f;

  fractions[idx] = fraction;
}

// --------------------------------------------------------
// Populate IBM Scaling Kernel
// --------------------------------------------------------
__global__ void populate_ibm_scaling_kernel(float *__restrict__ d_rescale,
                                            IBM_Data ibm_data,
                                            int num_elements) {
  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm_data.num_active_cells)
    return;

  int c_idx = ibm_data.cell_index[list_idx];
  float scale = ibm_data.D_rescale[list_idx];

  // Cells not in list retain initialized value (1.0)
  d_rescale[c_idx] = scale;
}