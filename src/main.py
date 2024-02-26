from methods import get_reduced_points
import pandas as pd


if __name__ == "__main__":
    # Utilizing the array provided in the exercise example
    data = pd.read_csv("data/polygons_example.csv")

    points = data[data["name"] == "Polygon1"]
    points = points[["x", "y"]].to_numpy()
    # We can even increase the threshold here to be more restrict
    mask = get_reduced_points(points=points, threshold=0.4)

    reduced_points = points[mask]

    print(
        f"Points before optimization: {points}\n Points after optimization: {reduced_points}"
    )

    pd.DataFrame(reduced_points, columns=["x", "y"]).to_csv(
        "output/polygons_example.csv", index=False
    )
