from collections import deque


# implement a stack using queue
class MyStack:
    def __init__(self):
        self.queue = deque([])

    def push(self, x):
        self.queue.append(x)
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self):
        return self.queue.popleft()

    def top(self):
        return self.queue[0]

    def empty(self):
        return not not self.queue


if __name__ == '__main__':
    stack = MyStack()
    stack.push(1)
    stack.push(2)
    stack.push(10)
    stack.push(8)
    assert stack.pop() == 8
    assert stack.top() == 10
    assert stack.empty()






