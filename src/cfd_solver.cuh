#pragma once

#include "sdf_reader.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

// Device helper for periodic indexing
// Index = i + j*nx + k*nx*ny
__device__ inline int get_idx(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

// SoA Structure for IBM Data (Robust Scaled)
struct IBM_Data {
  int num_active_cells;

  // Per-Cell Data
  int *cell_index;     // [num_active] - Global Grid Index
  float *D_rescale;    // [num_active] - Scaling factor D_rescale (Eq. 17)
  int *num_boundaries; // [num_active] - Number of modified directions

  // Per-Direction Data (Size: 6 * num_active)
  // Access: cells[list_idx * 6 + k]
  // We store modification factors for the "baked-in" update
  // Direction Indices k=0..5 correspond to directions (X+, X-, Y+, ...) or just
  // sequential list? Paper says: Loop over directions d. We need to know WHICH
  // direction `d` this modification applies to (to update a_nb,d). So we store
  // `nb_idx` (direction code or neighbor index).

  int *dir_code; // [num_active * 6] - Direction Code (0:X+, 1:X-, etc.)

  // Modification Factors (Eq 23-26):
  // a_c += a_nb * K
  // a_nb = a_nb * M + a_other * X (Cross-term for sandwich)
  // b_c += B_val

  float *K_val; // [num_active * 6] - Factor K (Add neighbor to center)
  float *M_val; // [num_active * 6] - Factor M (Scale neighbor)
  float *X_val; // [num_active * 6] - Factor X (Cross-term, usually 0 unless
                // sandwich)
  float *B_val; // [num_active * 6] - Factor B (Correction to RHS)
  float *R_val; // [num_active * 6] - D_rescale / D_axis ratio per direction

  // Note: Standard N_bc, val_bc are subsumed into B_val calculation
  // or kept for explicit flux calculation if needed?
  // Paper Eq 28: b'_c = D*b_c - sum( ... u_bc ... )
  // So B_val pre-calculates the u_bc term.
};

struct MacGrid {
  int3 res;         // Resolution (cells)
  float3 spacing;   // dx, dy, dz
  int num_elements; // res.x * res.y * res.z

  // Grid indexing convention (origin at 0,0,0):
  // - Cell-centered fields (SDF, pressure): (i+0.5, j+0.5, k+0.5) * spacing
  // - u(i,j,k): (i, j+0.5, k+0.5) * spacing
  // - v(i,j,k): (i+0.5, j, k+0.5) * spacing
  // - w(i,j,k): (i+0.5, j+0.5, k) * spacing

  // Fields
  float *u, *v, *w; // Velocity (staggered)
  float *p;         // Pressure (centered)
  float *rhs;       // RHS for pressure solve / Temporary RHS for Momentum
  float *sdf;       // Signed Distance Field (centered)

  // Surface Fractions (Face-Centered)
  float *frac_u, *frac_v, *frac_w;

  // Stencil Arrays (7-point) - For Generic RB-GS Solver
  // A_C (Center), A_W (-X), A_E (+X), A_S (-Y), A_N (+Y), A_B (-Z), A_T (+Z)
  float *A_C;
  float *A_W, *A_E;
  float *A_S, *A_N;
  float *A_B, *A_T;
  float *B_RHS; // Unified Source Term Storage

  // IBM Data (SoA) - Centered (Pressure/SDF)
  IBM_Data ibm_data;
  int *ibm_id_map; // Map from grid index to ibm_cells index (-1 if fluid)
  int num_ibm_cells;

  // Staggered IBM Data (Velocity)
  IBM_Data ibm_data_u, ibm_data_v, ibm_data_w;
  int *ibm_id_map_u, *ibm_id_map_v, *ibm_id_map_w;
  int num_ibm_cells_u, num_ibm_cells_v, num_ibm_cells_w;

  // Newton-Raphson Buffers
  float *u_old, *v_old, *w_old; // Previous time step values (u^n)
  float *p_old;                 // Previous time step pressure (p^n)
  float *res_u, *res_v, *res_w; // Momentum residuals f^(k)
  float *phi;                   // Pressure correction auxiliary variable
  float *du, *dv, *dw;          // Velocity increments (delta u*, delta v*, delta w*)

  float3 body_force_;
  float3 body_accel_;
};

class CFDSolver {
public:
  CFDSolver(int3 res, float3 spacing);
  ~CFDSolver();

  // Disable copying
  CFDSolver(const CFDSolver &) = delete;
  CFDSolver &operator=(const CFDSolver &) = delete;

  // Initialize from host data (if needed, or just zero out)
  void initialize(const SDFData &sdf_data);

  // Set Body Force
  void set_body_force(float3 force);

  // Pre-process IBM Geometry
  void update_ibm_geometry();

  // Set Diffusion Scheme (0.5 = Crank-Nicolson, 1.0 = Fully Implicit)
  void set_diffusion_theta(float theta);

  // Getters for visualization (copy to host)
  std::vector<float> get_u() const;
  std::vector<float> get_v() const;
  std::vector<float> get_w() const;
  std::vector<float> get_p() const;

  void set_u(const std::vector<float> &u);
  void set_v(const std::vector<float> &v);
  void set_w(const std::vector<float> &w);

private:
  MacGrid grid;
  size_t num_elements;
  float diffusion_theta;
  float current_dt_;
  float target_cfl_;
  float rho_;
  float mu_;
  float nu_;
  int p_max_iter_;
  float p_tol_;
  int v_max_iter_;
  float v_tol_;
  int outer_iterations_ = 1;

  // Helper to compute max velocity magnitude on device
  float compute_max_velocity();

public:
  // CFL Control
  void set_cfl(float cfl);
  float get_cfl() const;
  float get_dt() const;

  // Material Properties
  void set_rho(float rho);
  void set_mu(float mu);

  // Solver Parameters
  void set_pressure_solver_params(int max_iter, float tol);
  void set_velocity_solver_params(int max_iter, float tol);

  // Unified Solver Step (Picard Iteration)
  // Replaces step_newton
  void step(float dt);

  // Perform Pressure Projection only (for Unit Testing/Splitting)
  void project(float dt, bool incremental = false);

  // Compute Volume/Area Fractions
  // type: 0=Vol, 1=Ax, 2=Ay, 3=Az
  // offset: {0,0,0} or staggered (e.g., {-0.5,0,0} for u-faces)
  std::vector<float> get_fluid_fraction(int type, float3 offset);

protected:
  int pin_idx;
};

// Start of Kernel Declarations
__global__ void compute_fluid_fraction_kernel(const float *__restrict__ sdf,
                                              float *fractions, int3 res,
                                              float3 spacing, float3 offset,
                                              int type);

__global__ void compute_convection_defect_kernel(const float *__restrict__ u,
                                                 const float *__restrict__ v,
                                                 const float *__restrict__ w,
                                                 float *__restrict__ rhs,
                                                 int3 res, float3 spacing,
                                                 int component_idx);

// HELPER IMPLEMENTATION
// (Removed get_ibm_update_rbgs as we switched to Stencil Solver)
__global__ void solve_velocity_implicit_kernel(
    float *__restrict__ u, const float *__restrict__ rhs,
    const float *__restrict__ u_old, const float *__restrict__ v_old,
    const float *__restrict__ w_old, const float *__restrict__ sdf,
    IBM_Data ibm_data, const int *__restrict__ ibm_id_map, int3 res,
    float3 spacing, float dt, float nu, int component_idx, bool is_red);
