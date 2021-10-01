# sum with 1: [1, 1, 1] -> [1, 1, 3] -> [1, 5, 3] -> [1, 5, 9], 3 times of transformation
def sum_with_one(queries):
    import heapq
    optimal = float('+Inf')
    for query in queries:
        count, valid = 0, True
        query = [-x for x in query[::-1]]
        cur_sum = sum(query)
        while cur_sum + len(query) != 0:
            cur_max = heapq.heappop(query)
            cur_gap = cur_sum - cur_max
            cur_max -= cur_gap
            cur_sum -= cur_gap
            heapq.heappush(query, cur_max)
            print(cur_gap, cur_max, cur_sum, query)
            if cur_max >= 0:
                valid = False
                break
            count += 1
        if valid:
            optimal = min(optimal, count)
    return optimal


# from point (x, y) to (x + y, y) or (x, y + x)
def reaching_point(sx, sy, tx, ty):
    while tx >= sx and ty >= sy:
        if tx == ty:
            break
        elif tx > ty:  # when tx > ty
            if ty > sy:  # if ty is larger than sy, then reduce tx since we have tx > ty
                tx %= ty
            else:
                return (tx - sx) % ty == 0
        else:
            if tx > sx:
                ty %= tx
            else:
                return (ty - sy) % tx == 0
    return tx == sx and ty == sy