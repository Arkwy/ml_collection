# Machine Learning Algorithms - Python

This folder contains implementations of various machine learning algorithms in **Python**. Each algorithm is structured within its own subfolder, with `[algo]/main.py` serving as the entry point for each of them.

## Dependencies
To run the implementations, you need to install the required dependencies. The main libraries used in this repository include:

- **PyTorch** (for computation)
- **OpenCV** (for visualization)

### Installation
Ensure you have Python installed (only version: **3.13** has been tested). Then, install the dependencies manually:

```bash
pip install torch opencv-python
```

For Nix users, a `flake.nix` file is provided. You can enter a development shell with the necessary dependencies using:

```bash
nix develop
```

## Running the Project
To test an algorithm, modify then run the `main.py` file from the corresponding folder:

```bash
python main.py
```

This will initialize and execute the selected machine learning algorithm.

## Available Algorithms
✅ **Particle Swarm Optimization (PSO)**

🚧 **More algorithms coming soon!**
