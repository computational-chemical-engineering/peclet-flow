#pragma once

#include "sdf_reader.h"
#include <cuda_runtime.h>
#include <vector>

// Device helper for periodic indexing
// Index = i + j*nx + k*nx*ny
__device__ inline int get_idx(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
  return z * res.y * res.x + y * res.x + x;
}

// SoA Structure for IBM Data
struct IBM_Data {
  int num_active_cells;

  // Per-Cell Data (Size: num_elements or compacted)
  // We use compacted list indexing via ibm_id_map
  // Size = num_elements (max potential) or Reallocated?
  // Let's allocate num_elements for safety/simplicity first.

  int *cell_index;     // [num_active] - Global Grid Index
  float *S_row;        // [num_active]
  int *num_boundaries; // [num_active]

  // Per-Direction Data (Size: 6 * num_active)
  // Access: cells[list_idx].dirs[k] -> arrays[6 * list_idx + k]
  float *N_bc;
  float *val_bc;
  int *nb_idx;
};

// HELPER IMPLEMENTATION
__device__ inline float
get_ibm_update_rbgs(int idx, float current_sum_neighbors, float current_rhs,
                    const IBM_Data &ibm_data, const int *ibm_id_map) {
  int list_idx = ibm_id_map[idx];
  if (list_idx == -1)
    return -1e9f;

  float s_row = ibm_data.S_row[list_idx];
  int num_b = ibm_data.num_boundaries[list_idx];

  float ibm_rhs = current_rhs;
  for (int k = 0; k < num_b; k++) {
    int entry = 6 * list_idx + k; // Stride 6
    float n_bc = ibm_data.N_bc[entry];
    float v_bc = ibm_data.val_bc[entry];
    ibm_rhs += n_bc * v_bc;
  }

  return (ibm_rhs + current_sum_neighbors) * s_row;
}

struct MacGrid {
  int3 res; // Resolution (cells). Face counts match cell counts (Periodic)
  float3 spacing;   // dx, dy, dz
  int num_elements; // res.x * res.y * res.z

  // Fields
  float *u, *v, *w; // Velocity (staggered)
  float *p;         // Pressure (centered)
  float *rhs;       // RHS for pressure solve
  float *sdf;       // Signed Distance Field (centered)

  // IBM Data (SoA) - Centered (Pressure/SDF)
  IBM_Data ibm_data;
  int *ibm_id_map; // Map from grid index to ibm_cells index (-1 if fluid)
  int num_ibm_cells;

  // Staggered IBM Data (Velocity)
  IBM_Data ibm_data_u, ibm_data_v, ibm_data_w;
  int *ibm_id_map_u, *ibm_id_map_v, *ibm_id_map_w;
  int num_ibm_cells_u, num_ibm_cells_v, num_ibm_cells_w;

  // Double buffering temps
  float *u_temp, *v_temp, *w_temp;

  float3 body_force_; // Force per Unit Volume (if user intends f) or
                      // Mass-Specific Force? User said "set_body_force sets
                      // acceleration". F=ma => a=F/m. In CFD, usually f =
                      // Force/Volume. Then a = f/rho.
  float3 body_accel_; // Derived: body_force_ / rho_
};

class CFDSolver {
public:
  CFDSolver(int3 res, float3 spacing);
  ~CFDSolver();

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

  // Run one time step
  // dt: time step size (optional/auto)
  void step(float dt);

  // Temporary storage for Red-Black Gauss Seidel could go here if needed
  // or just reusing existing pointers

  // Compute Volume/Area Fractions
  // type: 0=Vol, 1=Ax, 2=Ay, 3=Az
  // offset: {0,0,0} or staggered
  std::vector<float> get_fluid_fraction(int type, float3 offset);

  // Run one implicit Convection-Diffusion time step
  // dt: must be provided given implicit nature usually desires large dt
  void step_implicit(float dt);
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

__global__ void solve_velocity_implicit_kernel(
    float *__restrict__ u, const float *__restrict__ rhs,
    const float *__restrict__ u_old, const float *__restrict__ v_old,
    const float *__restrict__ w_old, const float *__restrict__ sdf,
    IBM_Data ibm_data, const int *__restrict__ ibm_id_map, int3 res,
    float3 spacing, float dt, float nu, int component_idx, bool is_red);
