# union 2 rectangle
def union_two_rectangles(rectangle1, rectangle2):
    rectangle1.sort()
    rectangle2.sort()
    if rectangle1[0][0] <= rectangle2[0][0]:
        (x1, y1), (x2, y2) = rectangle1
        (x3, y3), (x4, y4) = rectangle2
    else:
        (x1, y1), (x2, y2) = rectangle2
        (x3, y3), (x4, y4) = rectangle1
    if x3 < x2 and not y4 >= y1 and not y3 <= y2:
        mask = abs(max(x1, x3) - min(x2, x4)) * abs(max(y2, y4) - min(y1, y3))
    else:
        mask = 0
    return abs(x1 - x2) * abs(y1 - y2) + abs(x3 - x4) * abs(y3 - y4) - mask


assert union_two_rectangles([[1, 4], [3, 1]], [[2, 3], [4, 2]]) == 7


def union_three_rectangles(rectangles):
    x = set()
    queries = []
    for x1, y1, x2, y2 in rectangles:
        x.add(x1)
        x.add(x2)
        queries.append((y1, 1, x1, x2))
        queries.append((y2, -1, x1, x2))
    # array of x coordinates in left to right order
    i_to_x = list(sorted(x))
    # inverse dictionary maps x coordinate to its rank
    x_to_i = {xi: i for i, xi in enumerate(i_to_x)}
    # number of current rectangles intersected by the sweepline in interval [i_to_x[i], i_to_x[i+1]]
    num_current_rectangles = [0] * (len(i_to_x) - 1)
    area = 0
    length_union_intervals = 0
    previous_y = 0  # arbitrary initial value. because length is 0 at first iteration
    for y, offset, x1, x2 in sorted(queries):
        area += (y - previous_y) * length_union_intervals
        i1, i2 = x_to_i[x1], x_to_i[x2]  # update number of current rectangles that are intersected
        for j in range(i1, i2):
            length_interval = i_to_x[j + 1] - i_to_x[j]
            if num_current_rectangles[j] == 0:
                length_union_intervals += length_interval
            num_current_rectangles[j] += offset
            if num_current_rectangles[j] == 0:
                length_union_intervals -= length_interval
        previous_y = y
    return area


def union_n_rectangles(rectangles):
    # Populate events
    OPEN, CLOSE = 0, 1
    events = []
    for x1, y1, x2, y2 in rectangles:
        events.append((y1, OPEN, x1, x2))
        events.append((y2, CLOSE, x1, x2))
    events.sort()

    def query():
        ans = 0
        cur = -1
        for x1, x2 in active:
            cur = max(cur, x1)
            ans += max(0, x2 - cur)
            cur = max(cur, x2)
        return ans
    active = []
    cur_y = events[0][0]
    ans = 0
    for y, typ, x1, x2 in events:
        # For all vertical ground covered, update answer
        ans += query() * (y - cur_y)
        # Update active intervals
        if typ is OPEN:
            active.append((x1, x2))
            active.sort()
        else:
            active.remove((x1, x2))
        cur_y = y
    return ans % (10 ** 9 + 7)