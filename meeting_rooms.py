import heapq

# meeting rooms i
def meeting_rooms_i(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False
    return True


def meeting_rooms_ii(intervals):
    intervals.sort()
    queue, count = [intervals[0][1]], 0
    for interval in intervals[1:]:
        if queue[0] <= interval[0]:
            heapq.heappop(queue)
        heapq.heappush(queue, interval[1])
    return len(queue)
