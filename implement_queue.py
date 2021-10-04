# implement a queue using stack
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        # O(1)
        self.s1.append(x)

    def pop(self):
        # amortized O(1), worst O(n)
        self.peek()  # move the elements in s1 to s2 in a reversed order
        return self.s2.pop()

    def peek(self):
        # O(1), worst O(n)
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]

    def empty(self):
        return not self.s1 and not self.s2


if __name__ == '__main__':
    q = MyQueue()

    q.push(1)  # [1]
    q.push(2)  # [1, 2]
    q.push(3)  # [1, 2, 3]
    q.push(4)  # [1, 2, 3, 4]
    q.push(5)  # [1, 2, 3, 4, 5]
    assert q.peek() == 1
    assert q.s1 == []
    assert q.s2 == [5, 4, 3, 2, 1]
    assert q.pop() == 1
    assert q.pop() == 2
    q.push(8)
    assert q.s1 == [8]
    assert q.s2 == [5, 4, 3]
    assert q.peek() == 3
    assert q.s2 == [5, 4, 3]




