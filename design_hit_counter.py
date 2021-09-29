# design a hit counter
from collections import deque


class HitCounter:
    def __init__(self):
        self.queue = deque([])

    def hit(self, timestamp):
        self.queue.append(timestamp)

    def get_hits(self, timestamp):
        while self.queue and timestamp - self.queue[0] >= 300:
            self.queue.popleft()
        return len(self.queue)

