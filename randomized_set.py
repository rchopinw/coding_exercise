from random import choice
from collections import defaultdict


# random return with O(1)
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.list = []
        self.dict = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            idx = self.dict[val]
            self.list[idx], self.dict[self.list[-1]] = self.list[-1], idx
            self.list.pop()
            del self.dict[val]
            return True
        return False

    def get_random(self) -> int:
        """
        Get a random element from the set.
        """
        return choice(self.list)


class RandomizedCollection:
    def __init__(self):
        self.dict = defaultdict(set)
        self.list = []

    def insert(self, val):
        self.dict[val].add(len(self.list))
        self.list.append(val)
        return len(self.list[val]) == 1

    def remove(self, val):
        if not self.dict[val]:
            return False
        last_element = self.list[-1]
        remove_idx = self.dict[val].pop()
        self.list[remove_idx] = last_element
        self.dict[last_element].add(remove_idx)
        self.dict[last_element].discard(len(self.list) - 1)
        self.list.pop()
        return True

    def get_random(self):
        return choice(self.list)
