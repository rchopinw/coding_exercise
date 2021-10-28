import heapq
import bisect
from collections import defaultdict


# class room:
class ClassRoom:
    def __init__(self, n):
        self.intervals = defaultdict(set)
        self.pq = [(0, 0, n - 1)]
        self.n = n
        self.zero_deleted = True
        self.n_deleted = True

    def seat(self) -> int:
        while self.pq:
            _, i, j = heapq.heappop(self.pq)
            if (i, j) in self.intervals[i]:
                continue
            break

        if self.zero_deleted and i == 0:
            self.zero_deleted = False
            mid = 0
            self._put_right(mid, j)
        elif self.n_deleted and j == self.n - 1:
            self.n_deleted = False
            mid = self.n - 1
            self._put_left(i, mid)
        else:
            mid = (i + j) // 2
            self._put_left(i, mid)
            self._put_right(mid, j)
        return mid

    def _put_left(self, i, mid):
        heapq.heappush(self.pq, (-((mid - i) // 2), i, mid))
        self._clean_left(i, mid)
        self._clean_right(i, mid)

    def _put_right(self, mid, j):
        heapq.heappush(self.pq, (-((j - mid) // 2), mid, j))
        self._clean_left(mid, j)
        self._clean_left(mid, j)

    def _clean_left(self, key, y):
        evict = set([(i, j) for i, j in self.intervals[key] if i == key])
        self.intervals[key] = self.intervals[key].difference(evict)
        self.intervals[key].add((key, y))

    def _clean_right(self, x, key):
        evict = set([(i, j) for i, j in self.intervals[key] if j == key])
        self.intervals[key] = self.intervals[key].difference(evict)
        self.intervals[key].add((x, key))

    def leave(self, p):
        vals = sorted(self.intervals[p])
        l, r = vals[0][0], vals[0][1]
        if p == 0:
            self.zero_deleted = True
            heapq.heappush(self.pq, (-(r - l), l, r))
            self._clean_right(p, r)
            self._clean_left(p, r)
        elif p == self.n - 1:
            self.n_deleted = True
            heapq.heappush(self.pq, (-(r - l), l, r))
            self._clean_right(l, p)
            self._clean_left(l, p)
        else:
            l, r = vals[0][0], vals[1][1]
            self._clean_right(l, r)
            self._clean_left(l, r)
            if (self.zero_deleted and l == 0) or (self.n_deleted and r == self.n - 1):
                heapq.heappush(self.pq, (-(r - l), l, r))
            else:
                heapq.heappush(self.pq, (-((r - l) // 2), l, r))
            self.intervals.pop(p)


class ExamRoom:
    def __init__(self, n):
        self.students = []
        self.n_seats = n

    def seat(self):
        pos = 0
        if self.students:
            optimal_distance = self.students[0]
            for prev_, next_ in zip(self.students, self.students[1:]):
                if (next_ - prev_) // 2 > optimal_distance:
                    optimal_distance, pos = (next_ - prev_) // 2, prev_ + (next_ - prev_) // 2
            if self.n_seats - 1 - self.students[-1] > optimal_distance:
                pos = self.n_seats - 1
            if self.students[0] - 0 > optimal_distance:
                pos = 0
        ins_pos = self._binary_search(self.students, pos)
        self.students.insert(ins_pos, pos)
        return pos

    def leave(self, p):
        self.students.remove(p)

    def _binary_search(self, nums, k):
        if nums[-1] < k:
            return len(nums)
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < k:
                left = mid + 1
            else:
                right = mid
        return left


