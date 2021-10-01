import heapq
from collections import defaultdict, deque


def maximum_median_value(nums, k):
    smalls, bigs = [], nums[:k]
    heapq.heapify(bigs)
    while len(smalls) < len(bigs):
        heapq.heappush(smalls, -heapq.heappop(bigs))
    i = k - 1
    removals, medians = defaultdict(int), []
    while i < len(nums):
        print(smalls, bigs, len(smalls) + len(bigs))
        medians.append((bigs[0] - smalls[0])/2.0 if k%2 == 0 else -smalls[0])
        i += 1
        if i == len(nums):
            break
        balance, in_item, out_item = 0, nums[i], nums[i - k]
        balance += -1 if out_item <= -smalls[0] else 1
        removals[out_item] += 1
        if smalls and in_item <= -smalls[0]:
            balance += 1
            heapq.heappush(smalls, -in_item)
        else:
            balance -= 1
            heapq.heappush(bigs, in_item)

        if balance > 0:
            heapq.heappush(bigs, -heapq.heappop(smalls))
            balance -= 1
        if balance < 0:
            heapq.heappush(smalls, -heapq.heappop(bigs))
            balance += 1

        while smalls and removals[-smalls[0]]:
            removals[-smalls[0]] -= 1
            heapq.heappop(smalls)
        while bigs and removals[bigs[0]]:
            removals[bigs[0]] -= 1
            heapq.heappop(bigs)

    return medians


class MedianValueInFlow:
    def __init__(self,):
        self.smalls = []
        self.bigs = []

    def put_i(self, value):
        heapq.heappush(self.bigs, value)
        heapq.heappush(self.smalls, -heapq.heappop(self.bigs))
        if len(self.smalls) - len(self.bigs) > 1:
            heapq.heappush(self.bigs, -heapq.heappop(self.smalls))

    def put_ii(self, value):
        if self.smalls and value <= self.smalls[0]:
            heapq.heappush(self.smalls, -value)
        else:
            heapq.heappush(self.bigs, value)
        self._balance()

    def get(self, ):
        if len(self.smalls) > len(self.bigs):
            return -self.smalls[0]
        return (self.bigs[0] - self.smalls[0]) / 2.0

    def _balance(self):
        if len(self.smalls) > len(self.bigs):
            heapq.heappush(self.bigs, -heapq.heappop(self.smalls))
        if len(self.bigs) > len(self.smalls):
            heapq.heappush(self.smalls, -heapq.heappop(self.bigs))


class MedianValueInFixedSizeFlow:
    def __init__(self, k):
        self.capacity = k
        self.smalls = []
        self.bigs = []
        self.queue = deque([])
        self.removal = defaultdict(int)

    def put(self, value):
        self.queue.append(value)
        if len(self.queue) > self.capacity:
            balance = 0
            out_item = self.queue.popleft()
            balance += -1 if out_item <= -self.smalls[0] else 1
            if self.smalls and value <= -self.smalls[0]:
                balance += 1
                heapq.heappush(self.smalls, -value)
            else:
                balance -= 1
                heapq.heappush(self.bigs, value)
            if balance > 0:
                heapq.heappush(self.bigs, -heapq.heappop(self.smalls))
            if balance < 0:
                heapq.heappush(self.smalls, -heapq.heappop(self.bigs))
            self.removal[out_item] += 1
            while self.smalls and self.removal[-self.smalls[0]]:  # lazy removal
                self.removal[-self.smalls[0]] -= 1
                heapq.heappop(self.smalls)
            while self.bigs and self.removal[self.bigs[0]]:
                self.removal[self.bigs[0]] -= 1
                heapq.heappop(self.bigs)
        else:
            heapq.heappush(self.bigs, value)
            heapq.heappush(self.smalls, -heapq.heappop(self.bigs))
            if len(self.smalls) - len(self.bigs) > 1:
                heapq.heappush(self.bigs, -heapq.heappop(self.smalls))
        print(self.smalls, self.bigs, len(self.smalls) + len(self.bigs))

    def get(self):
        if len(self.queue) < self.capacity:
            if len(self.smalls) > len(self.bigs):
                return -self.smalls[0]
            return (self.bigs[0] - self.smalls[0]) / 2.0
        else:
            if self.capacity%2 == 0:
                return (self.bigs[0] - self.smalls[0]) / 2.0
            return -self.smalls[0]
