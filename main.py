from generate_points import generate_non_coplanar_points_octahedron
from convex_hull import create_convex_hull_model, train_model
from plotting import plot_convex_hull

# Generate points and create the model
num_input_points = 15
input_points = generate_non_coplanar_points_octahedron(num_input_points)

# Create and train the model
model = create_convex_hull_model()
train_model(model, input_points)
