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

