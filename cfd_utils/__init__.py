"""
CFD Utilities Package

Common utilities for CFD simulations including:
- VTI file I/O
- SDF generation
- Grid operations
- Reference data from literature
- Analytical solutions for verification
"""

from .vti import save_vti, save_vti_labeled
from .sdf import (
    generate_sphere_sdf,
    generate_sc_sphere_sdf,
    generate_slab_sdf,
    generate_angled_slab_sdf,
)
from .grid import (
    interpolate_to_cell_centers,
    compute_divergence,
    compute_velocity_magnitude,
)
from .reference import (
    get_zick_homsy_k_sc,
    get_sangani_acrivos_k_sc,
)
from .analytical import (
    poiseuille_profile,
    poiseuille_velocity_component,
)

__all__ = [
    # VTI
    "save_vti",
    "save_vti_labeled",
    # SDF
    "generate_sphere_sdf",
    "generate_sc_sphere_sdf",
    "generate_slab_sdf",
    "generate_angled_slab_sdf",
    # Grid
    "interpolate_to_cell_centers",
    "compute_divergence",
    "compute_velocity_magnitude",
    # Reference
    "get_zick_homsy_k_sc",
    "get_sangani_acrivos_k_sc",
    # Analytical
    "poiseuille_profile",
    "poiseuille_velocity_component",
]
