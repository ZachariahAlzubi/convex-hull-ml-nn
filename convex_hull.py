import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.spatial import ConvexHull

def create_convex_hull_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(3)
    ])
    return model

def convex_hull_loss(predictions, input_points, alpha=1.0):
    distance_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(predictions - input_points), axis=1)))
    convexity_loss_value = alternative_convexity_loss(predictions)
    total_loss = distance_loss + alpha * convexity_loss_value
    return total_loss

def alternative_convexity_loss(predictions):
    loss = 0
    num_points = tf.shape(predictions)[0]
    for i in range(num_points):
        for j in range(num_points):
            for k in range(j):
                for l in range(k):
                    if i != j and i != k and i != l:
                        distance = signed_distance_to_plane(predictions[i], [predictions[j], predictions[k], predictions[l]])
                        loss += tf.maximum(0.0, tf.minimum(-distance, 1e6))
    return loss / tf.cast(num_points * (num_points - 1) * (num_points - 2) * (num_points - 3), tf.float32)

def signed_distance_to_plane(point, plane_points):
    p1, p2, p3 = tf.unstack(plane_points)
    normal = tf.linalg.cross(p2 - p1, p3 - p1)
    normal += 1e-6
    normal = normal / tf.norm(normal)
    distance = tf.tensordot(point - p1, normal, axes=1)
    return distance

def train_model(model, input_points, epochs=25, learning_rate=0.01, alpha=1.0):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
    best_loss = float('inf')
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(input_points)
            loss = convex_hull_loss(predictions, input_points, alpha)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
