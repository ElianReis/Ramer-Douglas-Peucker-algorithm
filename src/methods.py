import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _dist2_fun(points, start, end):
    """
    Calculate the squared perpendicular distance from each point in the array 'points'
    to the line formed by the start and end points.

    Parameters:
    - points (numpy.ndarray): Array of points to calculate distances for.
    - start (numpy.ndarray): Starting point of the line.
    - end (numpy.ndarray): Ending point of the line.

    Returns:
    - distances (numpy.ndarray): Array of squared distances for each point.
    """
    if np.allclose(start, end):
        # If start and end points are close, calculate squared distances directly.
        return np.sum((points - start) ** 2, axis=1)

    # Calculate unit vector along the line segment.
    d = np.divide(end - start, np.sqrt(np.sum((end - start) ** 2)))

    # Calculate the maximum distances in both directions from the line.
    max_p1 = np.dot(start - points, d).max()
    max_p2 = np.dot(points - end, d).max()

    # Calculate squared perpendicular distances.
    return (
        max(max_p1, max_p2, 0) ** 2
        + np.cross(points - np.expand_dims(start, 0), np.expand_dims(d, 0)) ** 2
    )


def get_reduced_points(points, threshold):
    """
    Apply the Ramer-Douglas-Peucker algorithm to simplify a polygon by recursively
    removing points that are below a certain threshold of perpendicular distance
    from the line segment between the first and last points.

    Parameters:
    - points (numpy.ndarray): Array of points representing the polygon.
    - threshold (float): Threshold for the squared distance to decide if a point is kept.

    Returns:
    - mask (numpy.ndarray): Boolean mask indicating whether each point should be kept.
    """
    if points.shape[0] <= 2:
        # If there are only two or fewer points, keep all points.
        return np.array([True] * points.shape[0])

    start = points[0]
    end = points[-1]

    # Calculate squared distances for interior points.
    d = _dist2_fun(points[1:-1], start, end)

    # Find the index of the point with the maximum squared distance.
    i = np.argmax(d) + 1
    d_max = d[i - 1]

    if d_max > threshold**2:
        mask = np.concatenate(
            [
                # Recursively filter points before and after the point with max distance.
                get_reduced_points(points[: i + 1], threshold)[:-1],
                get_reduced_points(points[i:], threshold),
            ]
        )
    else:
        mask = np.array(
            # If max distance is below threshold, keep only the first and last points.
            [True]
            + [False] * (points.shape[0] - 2)
            + [True],
        )

    return mask


def plot_polygon(points, reduced_points):
    """
    Generate a plot view to identify if the boundary
    is related to the previous points array.

    Parameters:
    - points (numpy.ndarray): Input points.
    - reduced_points (numpy.ndarray): Points after applying the algorithm for simplification.
    """
    fig, ax = plt.subplots()

    # Plot original polygon.
    polygon = patches.Polygon(
        points, edgecolor="b", fill=None, linewidth=2, label="Original Polygon"
    )
    ax.add_patch(polygon)

    # Plot reduced polygon after simplification.
    reduced_polygon = patches.Polygon(
        reduced_points,
        edgecolor="r",
        fill=None,
        linewidth=2,
        label="Reduced Polygon",
    )
    ax.add_patch(reduced_polygon)

    # Scatter plot for original points.
    x, y = zip(*points)
    plt.scatter(x, y, color="b", marker="o", label="Points")

    # Scatter plot for reduced points.
    x_r, y_r = zip(*reduced_points)
    plt.scatter(x_r, y_r, color="r", marker="x", label="Reduced Points")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Polygon and Points")
    plt.legend()
    plt.grid(True)
    plt.show()
