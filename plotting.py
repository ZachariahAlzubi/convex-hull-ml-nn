import plotly.graph_objects as go
from scipy.spatial import ConvexHull

def plot_convex_hull(points, original_points, color, title):
    hull = ConvexHull(points)
    x, y, z = points.T
    orig_x, orig_y, orig_z = original_points.T

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z, i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
            opacity=0.5, name='Convex Hull'
        ),
        go.Scatter3d(
            x=orig_x, y=orig_y, z=orig_z,
            mode='markers', marker=dict(size=2, color='red'), name='Original Points'
        )
    ])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=title)
    fig.show()
    return hull
