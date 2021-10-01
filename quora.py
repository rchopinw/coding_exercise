# Quora round 2

# find peak element
def binary_search(nums, left, right, value):
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


def subset_sum_dp(nums, k):
    dp = [[False for _ in range(k + 1)] for _ in range(len(nums) + 1)]
    for i in range(len(nums) + 1):
        dp[i][0] = True
    for i in range(1, len(nums) + 1):
        for j in range(1, k + 1):
            if nums[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
    return dp[-1][-1]


# excel column name
def excel_column_number(column):
    return sum((ord(column[i]) - 64) * 26 ** (len(column) - i - 1) for i in range(len(column)))


def excel_column_name(num):
    ans = ''
    while num > 0:
        num -= 1
        ans += chr(num % 26 + 65)
        num //= 26
    return ans[::-1]


# implement a queue
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        self.s1.append(x)

    def pop(self):
        self.peek()  # move the elements in s1 to s2 in a reversed order
        return self.s2.pop()

    def peek(self):
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]

    def empty(self):
        return not self.s1 and not self.s2


# implement a stack using queue
class MyStack:
    def __init__(self):
        self.queue = deque([])

    def push(self, x):
        self.queue.append(x)
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self):
        self.queue.popleft()

    def top(self):
        return self.queue[0]

    def empty(self):
        return not self.queue


# partition of palindromes
def partition_of_palindromes(s):
    results = []

    def backtrack(cur, path):
        if not cur:
            results.append(path)
            return
        for i in range(1, len(cur) + 1):
            if cur[:i] == cur[:i][::-1]:
                backtrack(cur[i:], path + [cur[:i]])

    backtrack(s, [])

    return results


# shortest palindrome
def shortest_palindrome(s):
    s_dual = s + '*' + s[::-1]
    dp = [0] * len(s_dual)
    for i in range(1, len(dp)):
        j = dp[i - 1]
        while j > 0 and s_dual[i] != s_dual[j]:
            j = dp[j - 1]
        if s_dual[i] == s_dual[j]:
            dp[i] = j + 1
    return s[::-1][:len(s) - dp[-1]] + s


# solve for the binary representation of given number n - 1
def binary_representation_loop(n):
    s = ''
    n = n - 1
    while n > 0:
        s += str(n%2)
        n //= 2
    return str(int(s))[::-1]


def binary_representation_recursion(n, flag=True):
    if flag:
        n = n - 1
    if n == 0:
        return '0'
    else:
        return binary_representation_recursion(n // 2, False) + str(n % 2)


class QueryForEventCount:
    def __init__(self, intervals):
        self.intervals = intervals
        self.size = len(intervals)

    def request(self, query):
        count = 0
        for interval in self.intervals:
            if query < interval[0] or query > interval[1]:
                count += 1
        return self.size - count


# from point (x, y) to (x + y, y) or (x, y + x)
def reaching_point(sx, sy, tx, ty):
    while tx >= sx and ty >= sy:
        if tx == ty:
            break
        elif tx > ty:  # when tx > ty
            if ty > sy:  # if ty is larger than sy, then reduce tx since we have tx > ty
                tx %= ty
            else:
                return (tx - sx) % ty == 0
        else:
            if tx > sx:
                ty %= tx
            else:
                return (ty - sy) % tx == 0
    return tx == sx and ty == sy


def parse_string(s):
    result, tmp_s, i = [], '', 0
    while i < len(s):
        if s[i] == ' ':
            result.append(tmp_s)
            tmp_s = ''
        elif s[i] == '"':
            k = i + 1
            while k < len(s) and s[k] != '"':
                k += 1
            result.append(s[i+1:k])
            i = k + 1
            continue
        else:
            tmp_s += s[i]
        i += 1



