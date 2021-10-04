# LRU cache
class Node:
    def __init__(self, ):
        self.key = 0
        self.val = 0
        self.next = None
        self.prev = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.size = 0
        self.capacity = capacity
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._to_top(node)
        return node.val

    def put(self, key, value):
        node = self.cache.get(key)
        if not node:
            new_node = Node()
            new_node.key, new_node.val = key, value
            self.size += 1
            self.cache[key] = new_node
            if self.size > self.capacity:
                del_node = self._pop()
                del self.cache[del_node.key]
            self._add(new_node)
        else:
            node.val = value
            self._to_top(node)

    def _add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node):
        _next = node.next
        _prev = node.prev

        node.prev.next = _next
        node.next.prev = _prev

    def _to_top(self, node):
        self._remove(node)
        self._add(node)

    def _pop(self):
        _tail = self.tail.prev
        self._remove(_tail)
        return _tail


if __name__ == '__main__':
    cache = LRUCache(capacity=4)
    cache.put(2, 10)
    cache.put(3, 12)
    cache.put(1, 8)
    cache.put(4, 10)
    cache.put(12, 111)
    assert cache.head.next.val == 111
    assert cache.tail.prev.val == 12
    assert cache.get(1) == 8
    assert cache.head.next.val == 8





