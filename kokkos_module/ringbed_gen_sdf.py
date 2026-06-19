#!/usr/bin/env python3
# Generate a ring (hollow-cylinder) packed-bed SDF with the CUDA packing engine (demgpu), exactly the
# RingBed-CFD-Surrogate protocol, and save it as .npy. Run as a SUBPROCESS (demgpu owns Kokkos init via
# ArborX, so it must not share a process with sdflow_kokkos). Usage:
#   python ringbed_gen_sdf.py <out.npy> <res> [num_particles] [target_phi] [growth_rate] [steps] [seed]
import math, sys
import numpy as np

sys.path.insert(0, "/home/frankp/Codes/suite/packing-gpu/build")
import demgpu


def pack_rings(num_particles, target_phi, growth_rate, steps, seed,
               outer_diameter=1.0, aspect_ratio=1.5, wall_thickness=0.18, friction=0.05,
               dt=0.001, solver_iterations=24, initial_scale=0.05, temperature=0.5,
               jam_overlap=0.001, cooling_step=None):
    radius = 0.5 * outer_diameter
    height = outer_diameter * aspect_ratio
    r_in = radius - wall_thickness
    pvol = math.pi * height * (radius * radius - r_in * r_in)
    side = (num_particles * pvol / target_phi) ** (1.0 / 3.0)
    half = 0.5 * side
    if cooling_step is None:
        cooling_step = int(0.96 * steps)
    rng = np.random.default_rng(seed)
    s = demgpu.Simulation(num_particles)
    s.initialize(shape_type=2, radius=radius, height=height, thickness=wall_thickness)
    s.set_domain((-half, -half, -half), (half, half, half))
    s.enable_periodicity(True, True, True)
    s.set_gravity(0.0, 0.0, 0.0)
    s.set_material_params(1.0, 1.0, friction)
    s.set_solver_iterations(solver_iterations, solver_iterations)
    margin = 0.92 * half
    pos = rng.uniform(-margin, margin, (num_particles, 4)).astype(np.float32); pos[:, 3] = 1.0
    s.set_positions(pos)
    s.set_velocities(rng.normal(0.0, math.sqrt(temperature), (num_particles, 3)).astype(np.float32))
    q = rng.normal(0.0, 1.0, (num_particles, 4)).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    s.set_quaternions(q)
    s.set_angular_velocities(np.zeros((num_particles, 3), np.float32))
    s.set_scales(np.full(num_particles, 1.0, np.float32))
    gr = growth_rate
    s.set_growth_params(gr, initial_scale)
    s.set_thermostat(temperature, dt)
    for step in range(steps):
        if step == cooling_step:
            s.set_material_params(0.28, 1.0, friction)
            s.set_thermostat(0.0, 18000.0 * dt)
        s.step(dt)
        ov = float(s.get_max_overlap())
        if ov > jam_overlap:
            for _ in range(4):
                s.step(0.0)
                ro = float(s.get_max_overlap())
                if ro >= 0.98 * ov: break
                ov = ro
            gf = float(s.get_growth_factor()) * math.exp(-gr * dt)
            gr *= 0.99
            s.set_growth_params(gr, gf)
        else:
            gr = min(gr * 1.02, growth_rate)
            s.set_growth_params(gr, float(s.get_growth_factor()))
    phi = target_phi * float(np.mean(s.get_scales() ** 3))
    return s, phi


def main():
    out = sys.argv[1]; res = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    phi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.55
    gr = float(sys.argv[5]) if len(sys.argv) > 5 else 2.4
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 2600
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 42
    s, phi_got = pack_rings(n, phi, gr, steps, seed)
    sdf = np.asarray(s.get_sdf_grid((res, res, res)), dtype=np.float32)
    porosity = float((sdf > 0).mean())
    np.save(out, sdf)
    print(f"GEN ok: res={res} phi_pack={phi_got:.4f} porosity={porosity:.4f} sdf_range=[{sdf.min():.3f},{sdf.max():.3f}]", flush=True)


if __name__ == "__main__":
    main()
