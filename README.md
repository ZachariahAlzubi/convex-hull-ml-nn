# Convex Hull Model

## Overview
This project generates a set of non-coplanar points within an octahedron, then trains a neural network to predict a convex hull of the points. The project also includes visualization of the learned convex hull and the actual convex hull.

## Files
- `generate_points.py`: Code to generate non-coplanar points.
- `convex_hull.py`: Defines the neural network and loss functions.
- `plotting.py`: Functions to visualize the convex hull.
- `main.py`: Main script to run the model.

## Requirements
- TensorFlow
- NumPy
- SciPy
- Plotly

## How to Run
1. Install the dependencies:
pip install tensorflow numpy scipy plotly
2. Run the main script:
python main.py