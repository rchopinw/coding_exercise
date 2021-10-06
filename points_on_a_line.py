from collections import defaultdict


# maximum points on a line
def max_points(points):
    slopes = [defaultdict(int) for _ in range(len(points))]
    optimal = 0
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x_diff, y_diff = points[i][0] - points[j][0], points[i][1] - points[j][1]
            if x_diff == 0:
                slopes[i]['-'] += 1
            else:
                slopes[i][y_diff/x_diff] += 1
        optimal = max(optimal, max(slopes[i].values()))
    return optimal + 1