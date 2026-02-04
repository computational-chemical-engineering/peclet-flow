# Tech Stack

## Overview
This project leverages a multi-language and multi-paradigm technology stack designed for high-performance scientific computing and seamless integration with modern data analysis workflows. The core computational components are developed in C++ and CUDA for maximum performance, while Python serves as the primary interface for user interaction, scripting, and orchestration.

## Key Technologies

### Programming Languages
- **C++:** Used for developing performance-critical algorithms and data structures within the CFD solver.
- **CUDA:** Employed for parallelizing computationally intensive tasks on NVIDIA GPUs, enabling high-performance fluid flow simulations.
- **Python:** Serves as the high-level scripting language for setting up simulations, controlling the solver, post-processing results, and providing a user-friendly API.

### Libraries and Frameworks
- **CMake:** Used as the build system generator for managing the compilation of C++ and CUDA source code.
- **pybind11:** Facilitates seamless interoperability between C++/CUDA code and Python, allowing the exposure of high-performance components as Python modules.
- **CUDA Toolkit:** Provides the necessary development environment, libraries, and tools for CUDA programming.

### Domains and Methodologies
- **Computational Fluid Dynamics (CFD):** The primary scientific domain of the project, focusing on simulating fluid flow.
- **Immersed Boundary Method (IBM):** A key numerical method implemented to handle complex fluid-solid interactions and geometries within the CFD simulations.
- **Signed Distance Functions (SDF):** Utilized for efficient and accurate representation of intricate geometries, particularly relevant for porous media structures.
- **Pore Network Modeling (PNM):** Implied by the project's structure and components (e.g., `pnm_extraction`, `pore_extraction`), suggesting an application area involving the analysis and simulation of flow through pore networks.
