from collections import deque


# max distance in an array [9, 5, 1, 22, 8] -> [9, 1, 22, 5, 8]
def max_distance_in_array(nums):
    nums.sort()
    nums = deque(nums)
    cur_queue, cur_v = deque([nums.popleft()]), 0
    i = 1
    while nums:
        pending = nums.popleft() if i % 2 == 0 else nums.pop()
        left, right = abs(cur_queue[0] - pending), abs(cur_queue[-1] - pending)
        if left > right:
            cur_queue.appendleft(pending)
            cur_v += left
        else:
            cur_queue.append(pending)
            cur_v += right
        i += 1
