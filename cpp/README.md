# Machine Learning Algorithms - C++

This folder contains implementations of various machine learning algorithms in **C++**. Each algorithm is structured within its own subfolder, with `[algo]/src/main.cpp` serving as the entry for each of them.

## Dependencies

 - [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html)
 - [rocrand](https://github.com/ROCm/rocRAND)
 - [meson](https://mesonbuild.com/)
 - [ninja](https://ninja-build.org/)

For Nix users, you can enter a development shell with the necessary dependencies using:

```bash
nix develop
```

## Building and running an algorithm

```bash
CXX=hipcc meson setup builds
meson compile -C builds [algo(optional)]
./builds/[algo]
```

## Algorithms
🔄 **Particle Swarm Optimization (PSO) – In Progress**