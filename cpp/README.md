# Machine Learning Algorithms - C++

This folder contains implementations of various machine learning algorithms in **C++**. Each algorithm is structured within its own subfolder, with `[algo]/src/main.cpp` serving as the entry for each of them.

### Installation
    
For Nix users, you can enter a development shell with the necessary dependencies using:

```bash
nix develop
```
    
For other users refer to the [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html) to install HIP and it's dependencies (you'll probably have to additionnally install [rocrand](https://github.com/ROCm/rocRAND)).

## Building and Running the Project
A `Makefile` is provided to compile and run each algorithm. To build and execute go to the the folder of an aglorithm and run:

```bash
make
```

If the build succeeds, the program will automatically run.

## Algorithms
🔄 **Particle Swarm Optimization (PSO) – In Progress**
   - **Dependencies**: HIP 