import numpy as np

def generate_non_coplanar_points_octahedron(num_points):
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    
    if num_points <= 6:
        return vertices[:num_points]

    num_interior_points = num_points - 6
    points = vertices.tolist()

    while len(points) < num_points:
        point = np.random.uniform(-1, 1, (3,))
        if np.sum(np.abs(point)) <= 1:
            perturbation = np.random.normal(scale=0.01, size=3)
            point += perturbation

            coplanar = False
            if len(points) >= 3:
                for i in range(len(points) - 2):
                    for j in range(i + 1, len(points) - 1):
                        for k in range(j + 1, len(points)):
                            p1, p2, p3, p4 = np.array(points[i]), np.array(points[j]), np.array(points[k]), point
                            volume = np.abs(np.dot(np.cross(p2 - p1, p3 - p1), p4 - p1))
                            if volume < 1e-6:
                                coplanar = True
                                break
                        if coplanar:
                            break
                    if coplanar:
                        break

            if not coplanar:
                points.append(point.tolist())

    return np.array(points)
