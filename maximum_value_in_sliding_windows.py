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


# stock price input and output
class StockPrice:
    def __init__(self, k):
        self.window_size = k
        self.size = 0
        self.price_flow = deque([])

    def put(self, price):
        if self.window_size <= len(self.price_flow):
            self.price_flow.popleft()
        while self.price_flow and price > self.price_flow[-1]:
            self.price_flow.pop()
        self.price_flow.append(price)

    def get_price(self):
        return self.price_flow[0]


