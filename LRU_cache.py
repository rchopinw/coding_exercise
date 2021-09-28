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
        return node

    def put(self, key, value):
        node = self.cache.get(key)
        if not node:
            new_node = Node()
            new_node.key, new_node.val = key, value
            self.size += 1
            if self.size > self.capacity:
                self._pop()
                del self.cache[key]
                self._add(new_node)
            else:
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
