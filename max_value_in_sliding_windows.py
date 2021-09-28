from collections import deque


# max values in the sliding windows
def max_sliding_window(nums, k):
    if not nums:
        return []
    if k == 1:
        return nums

    def clean_queue(queue, i):
        if queue and queue[0] == i - k:  # window moves forward
            queue.popleft()
        while queue and nums[i] > nums[queue[-1]]:
            queue.pop()
        return queue

    deq = deque([])

    for i in range(k):
        deq = clean_queue(deq, i)
        deq.append(i)

    max_values = [nums[deq[0]]]

    for i in range(k, len(nums)):
        deq = clean_queue(deq, i)
        deq.append(i)
        max_values.append(nums[deq[0]])

    return max_values
