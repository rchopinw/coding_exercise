from collections import deque
import heapq
from time import time


class AverageFinder:
    def __init__(self, ):
        self.sample = []

    def add_sample(self, sample: float, timestamp=None, ) -> None:
        if not timestamp:
            timestamp = time()
        if not self.sample:
            self.sample.append((sample, timestamp))
        else:
            self.sample.append((sample + self.sample[-1][0], timestamp))

    def get_avg_in_past_hour(self, time_span, timestamp=None, ):
        if not self.sample:
            return 0.0
        if not timestamp:
            timestamp = time()
        prev_timestamp = timestamp - time_span
        lower_bound = self._binary_search(self.sample, prev_timestamp)
        upper_bound = self._binary_search(self.sample, timestamp)
        if self.sample[upper_bound][1] == timestamp:
            if lower_bound == 0:
                range_sum = self.sample[upper_bound][0]
            else:
                range_sum = self.sample[upper_bound][0] - self.sample[lower_bound - 1][0]
            return range_sum / (upper_bound - lower_bound + 1)
        else:
            if upper_bound == 0:
                return 0.0
            if lower_bound == 0:
                range_sum = self.sample[upper_bound - 1][0]
            else:
                range_sum = self.sample[upper_bound - 1][0] - self.sample[lower_bound - 1][0]
            return range_sum / (upper_bound - lower_bound)

    def _binary_search(self, nums, target, ):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid][1] < target:
                left = mid + 1
            else:
                right = mid
        return left


af = AverageFinder()
for i, j in zip([1, 2, 5, 8, 12, 30], [10, 15, 18, 24, 48, 51]):
    af.add_sample(i, j)
af.get_avg_in_past_hour(17, 50)
af.get_avg_in_past_hour(10, 51)
af.get_avg_in_past_hour(0, 1)
af.get_avg_in_past_hour(12, 48)
af.get_avg_in_past_hour(10, 49)
af.get_avg_in_past_hour(18, 18)
af.get_avg_in_past_hour(10, 10)
af.get_avg_in_past_hour(51, 51)


class AverageFinderConstantSpace:
    def __init__(self, ):
        self.sample = deque()
        self.total = 0

    def add_sample(self, sample, timestamp=None):
        if timestamp is None:
            timestamp = time()
        self.sample.append((sample, timestamp))
        self.total += sample

    def get_avg_in_past_hour(self, timestamp=None):
        if not self.sample:
            return 0.0

        if timestamp is None:
            timestamp = time()

        while self.sample:
            diff = timestamp - self.sample[0][1]
            if diff >= 3600:
                es = self.sample.popleft()
                self.total -= es[0]
            else:
                break
        return self.total / len(self.sample)


class AverageFinderHeap:
    def __init__(self, ):
        self.sample = []
        self.sum = 0.0

    def add_sample(self, sample):
        self.sum += sample
        heapq.heappush(self.sample, (time(), sample))

    def get_avg_in_past_hour(self, ):
        self._trace()
        return self.sum / len(self.sample)

    def _trace(self):
        while time() - self.sample[0][0] > 3600:
            self.sum -= heapq.heappop(self.sample)[1]
