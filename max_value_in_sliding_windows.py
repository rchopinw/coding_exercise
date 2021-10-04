from collections import deque


# max values in the sliding windows
def sliding_window_max(nums, k):
    if not nums:
        return []
    if k == 1:
        return nums
    queue = deque([])
    max_nums = []
    for i in range(k):
        while queue and nums[i] > nums[queue[-1]]:
            queue.pop()
        queue.append(i)
    max_nums.append(nums[queue[0]])
    for i in range(k, len(nums)):
        if queue and queue[0] == i - k:
            queue.popleft()
        while queue and nums[i] > nums[queue[-1]]:
            queue.pop()
        queue.append(i)
        max_nums.append(nums[queue[0]])
    return max_nums


if __name__ == '__main__':
    nums, k = [5, 4, 3, 2, 1, 2, 3, 2, 1], 3
    assert sliding_window_max(nums, k) == [5, 4, 3, 2, 3, 3, 3]

    nums, k = [3, 2, 1], 1
    assert sliding_window_max(nums, k) == nums










































