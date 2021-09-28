# odd occurrences of zero
def num_occurrence_of_zero(nums):
    total_count = 0
    duplicated_container = set()
    for num in nums:
        if num in duplicated_container:
            total_count += 1
        else:
            zero_count = 0
            for digit in str(num):
                if digit == '0':
                    zero_count += 1
            if zero_count // 2 == 1:
                total_count += 1
                duplicated_container.add(num)
    return total_count


assert 2 == num_occurrence_of_zero([5, 10, 200, 10070, 56])


# broken keyboard
# You have a passage of text that needs to be typed out, but some of the letter keys on your keyboard are broken!
# You are given an array letters representing the working letter keys, as well as a string text, and your task is
# to determine how many of the words from text can be typed using the broken keyboard. It is guaranteed that all
# of the non-letter keys are working (including all punctuation and special characters).
# A word is defined as a sequence of consecutive characters which does not contain any spaces.
# The given text is a string consisting of words, each separated by exactly one space.
# It is guaranteed that text does not contain any leading or trailing spaces.
# Note that the characters in letters are all lowercase, but since the shift key is working,
# it's possible to type the uppercase versions also.


def broken_keyboard(text, letters):
    count = 0
    text = text.split(' ')
    letters = set(letters)
    for word in text:
        flag = True
        for s in word:
            if s.isalpha() and s.lower() not in letters:
                flag = False
                break
        count += 1 if flag else 0
    return count


assert 2 == broken_keyboard("Hello, this is CodeSignal!", ['e', 'i', 'h', 'l', 'o', 's'])
assert 5 == broken_keyboard("3 + 2 = 5", [])
assert 0 == broken_keyboard(text="Hi, this is Chris!", letters=['r', 's', 't', 'c', 'h'])


# memory allocator 2
# Given an array of 0s and 1s, every 8 bits is considered 1 block and a contigous subarray of 0's
# starting with the starting index of each block is considered "free memory". For ex: [0011111111111100]. Block 1: [
# 00111111] Free memory size 2 Block 2: [11111100] Free memory: 0 Input is a list of queries of format (i,j) ,
# output list with size = size of queries list ie: 1 output per query, if i = 0, allocate memory, ex: (0,5) -> Find
# the earliest memory block where we have size 5 memory available ie: 5 consecutive 0's starting at the starting
# index of any given block; Return the starting index where free memory was found. This will be one of the starting
# indexes of each block. If no free memory found, return -1. Also when memory is successfully assigned, uniquely name
# this assignment, for ex: Assignment1. Once assigned those bits should be marked 1 or in other words that block be
# unavilable to use until memory is released. if i = 1, release memory, ex (1,3) -> Release the 3rd successfull
# memory assignement. This will always be valid, ie, There will always be 3 successful queries of type 0 before a
# release memory query with value 3. Return the starting index of memory being released. Once release those bits
# should be marked 0 or in other words that block is again avilable to use.
class MemoryAllocator:
    def __init__(self):
        self.valid_status = {x: [] for x in range(9)}
        self.success_count = 1
        self.log = {}

    def main(self, bits, queries):
        execute_results = []
        self._process_bits(bits)
        for i, j in queries:
            if i == 0:
                execute_results.append(self.allocate(j))
            else:
                execute_results.append(self.release(j))
        return execute_results

    def _process_bits(self, bits):
        for i in range(len(bits) // 8 + 1):
            cur = bits[i * 8:(i + 1) * 8]
            if not cur:
                continue
            num_free_slots = self._cal_free_slots(cur)
            self.valid_status[num_free_slots].append(i)

    def allocate(self, size):
        for s in range(size, 9):
            if self.valid_status[size]:
                idx = self.valid_status[size].pop(0)
                self.log[self.success_count] = idx
                self.success_count += 1
                return idx
        return -1

    def release(self, sc):
        idx = self.log[sc]
        self.valid_status[8].append(idx).sorted()

    @staticmethod
    def _cal_free_slots(chunk):
        count = 0
        idx = 0
        while chunk[idx] != 0:
            count += 1
            idx += 1
        return count


# longest arithmetic sequence
def las(nums, difference):
    r = {}
    optimal = 1
    for num in nums:
        if num - difference in r:
            r[num] = r[num - difference] + 1
            optimal = max(optimal, r[num])
        else:
            r[num] = 1
    return optimal


# substring divided by 3
def divisible_by_3(s):
    ps = 0
    count = 0
    prefix = []
    for num in s:
        ps += int(num)
        prefix.append(ps)
    for i in range(len(s)):
        for j in range(i, len(s)):
            if i == j and prefix[i] % 3 == 0:
                count += 1
            if i != j and (prefix[j] - prefix[i]) % 3 == 0 and s[i] != '0':
                count += 1
    return count


def f(i, rem, s, memoize):
    if i == len(s):
        return 0
    if memoize[i][rem] != -1:
        return memoize[i][rem]
    x = ord(s[i]) - ord('0')
    res = ((x + rem) % 3 == 0) + f(i + 1, (x + rem) % 3, s, memoize)
    memoize[i][rem] = res
    return memoize[i][rem]


def countDivBy3(s):
    n = len(s)
    memoize = [[-1] * 3 for i in range(n)]
    ans = 0
    for i in range(len(s)):
        if s[i] == '0':
            ans += 1
        else:
            ans += f(i, 0, s, memoize)
    return ans


assert 1 == divisible_by_3('011')
assert 5 == divisible_by_3('603')


# matrix rotation
def matrix_rotation(matrix):
    # rotate 90 degrees
    matrix = [list(reversed(x)) for x in zip(*matrix)]

    # flip 1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = matrix[j][i]

    def rotate_diag(m):
        mat = [[0 for _ in range(len(m[0]))] for _ in range(len(m))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                if i == j or i + j == len(m) - 1:
                    mat[i][j] = m[i][j]
                    continue
                mat[i][j] = m[j][len(m) - i - 1]
        return mat

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_diag(matrix)


# merge string:
def merge_string(s1, s2):
    res, p1, n1, p2, n2 = '', 0, len(s1), 0, len(s2)
    while p1 < n1 and p2 < n2:
        res += s1[p1] + s2[p2]
        p1 += 1
        p2 += 1
    while p1 < n1:
        res += s1[p1]
        p1 += 1
    while p2 < n2:
        res += s2[p2]
        p2 += 1
    return res


assert 'abababbb' == merge_string(s1='aaa', s2='bbbbb')
assert '' == merge_string('', '')


# flip multiplication
def reverse_sum(nums):
    s = 0
    nums = [list(num) for num in nums]
    for num in nums:
        zero_rec = ''
        while num[-1] == '0':
            zero_rec += num.pop()
        num.reverse()
        s += int(''.join(num) + zero_rec)
    return s


# largest rectangle/square area
def largest_rectangle(heights):
    stack = [-1]
    optimal = 0
    for i in range(len(heights)):
        while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
            optimal = max(optimal, heights[stack.pop()] * (i - stack[-1] - 1))
        stack.append(i)
    print(stack)
    while stack[-1] != -1:
        optimal = max(optimal, heights[stack.pop()] * (len(heights) - stack[-1] - 1))
    return optimal


def largest_square(heights):
    pass


# problem 1: 1*2*3*4 - (1+2+3+4)
def product_sum(num):
    p, s = 1, 0
    for digit in str(num):
        p *= int(digit)
        s += int(digit)
    return p - s


# problem 2: fit in rectangular


# problem 3: exchange a b
def exchange_string(s1, s2):
    pass


# problem 51:
class SolutionP51:
    def __init__(self):
        self.count = 0

    def find_duplicate_arrays(self, nums):
        self.nums = nums
        if len(self.nums) <= 1:
            return len(self.nums)
        self._double_pointers(nums)
        return self.count

    def _double_pointers(self, nums):
        for i in range(len(nums)):
            single = 0
            cur = {}
            for j in range(i, len(nums)):
                if nums[j] in cur:
                    cur[nums[j]] += 1
                else:
                    cur[nums[j]] = 1
                if cur[nums[j]] == 1:
                    single += 1
                elif cur[nums[j]] > 1:
                    single -= 1
                if single == 0:
                    self.count += 1


s = SolutionP51()
s.find_duplicate_arrays([0, 0, 0])
s = SolutionP51()
assert 3 == s.find_duplicate_arrays([1, 2, 2, 2, 3])


# problem 50
class SolutionP50:
    def __init__(self):
        pass

    def diagonal_sort(self, matrix):
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])):
        #         matrix[i][j] = str(matrix[i][j])
        nrow, ncol = len(matrix), len(matrix[0])
        max_len = min(nrow, ncol)
        diag_record = {}
        row_range = [i for i in range(nrow - 1, -1, -1)]
        col_range = [i for i in range(1, ncol)]
        for idx, row in enumerate(row_range):
            cur_len = nrow - row if nrow - row <= max_len else max_len
            diag_record[idx + 1] = ''.join([matrix[row + i][i] for i in range(cur_len)])
            if cur_len != 3:
                num_duplicate, num_remain = max_len // len(diag_record[idx + 1]), max_len % len(diag_record[idx + 1])
                diag_record[idx + 1] = diag_record[idx + 1] * num_duplicate + diag_record[idx + 1][:num_remain]
        for idx, col in enumerate(col_range):
            total_idx = idx + nrow + 1
            cur_len = ncol - col if ncol - col <= max_len else max_len
            diag_record[total_idx] = ''.join([matrix[i][col + i] for i in range(cur_len)])
            if cur_len != 3:
                num_duplicate, num_remain = max_len // len(diag_record[total_idx]), max_len % len(
                    diag_record[total_idx])
                diag_record[total_idx] = diag_record[total_idx] * num_duplicate + diag_record[total_idx][:num_remain]
        return [i[0] for i in sorted(diag_record.items(), key=lambda x: x[1])]


s50 = SolutionP50()
matrix = [['a', 'b', 'c'], ['c', 'a', 'd'], ['m', 'n', 'k']]
assert [3, 4, 5, 2, 1] == s50.diagonal_sort(matrix)

matrix = [['a', 'a', 'a'], ['a', 'a', 'a'], ['a', 'a', 'a']]
assert [1, 2, 3, 4, 5] == s50.diagonal_sort(matrix)


# problem 49:
def sum_less_k(s, k):
    while len(s) > k:
        sl = [s[i * k:(i + 1) * k] for i in range(len(s) // k + 1) if s[i * k:(i + 1) * k]]
        s = ''.join([str(sum(int(x) for x in num)) for num in sl])
    return s


# problem 8:
def string_subtract(s):
    s = [int(x) for x in s]
    s.reverse()
    total_sum = 0
    while s:
        i = -1
        cur_subtract = s.pop()
        total_sum += cur_subtract
        while i >= -len(s) and s[i] >= cur_subtract:
            if i == -1 and s[i] == cur_subtract:
                s.pop()
            else:
                s[i] -= cur_subtract
                i -= 1
    return total_sum


# problem 9: monotonic
def is_3_monotonic(nums):
    for i in range(2, len(nums)):
        if nums[i] >= nums[i - 1] >= nums[i - 2] or nums[i] <= nums[i - 1] <= nums[i - 2]:
            continue
        else:
            return False
    return True


from collections import defaultdict


# longest Arithmetic Subsequence of Given Difference
def longest_arithmetic_subsequence1(nums, k):
    rec = defaultdict(int)
    optimal = 0
    for num in nums:
        if num - k in rec:
            rec[num] = rec[num - k] + 1
            optimal = max(optimal, rec[num])
        else:
            rec[num] = 1
    return optimal


def longest_arithmetic_subsequence2(nums):
    optimal, dp = 0, [{} for _ in range(len(nums))]
    for i in range(1, len(nums)):
        for j in range(0, i):
            diff = nums[i] - nums[j]
            if diff in dp[j]:
                dp[i][diff] = dp[j][diff] + 1
            else:
                dp[i][diff] = 2
            optimal = max(optimal, dp[i][diff])
    return optimal


# rotate and fall
def rotate_and_fall(matrix):
    n, d = len(matrix), len(matrix[0])
    for row in range(n):
        stack = []
        for col in range(d):
            if matrix[row][col] == '#':
                stack.append((row, col))
            elif (matrix[row][col] == '*' or (matrix[row][col] == '.' and col == d - 1)) and stack:
                for cur_row, cur_col in stack:
                    matrix[cur_row][cur_col] = '.'
                if matrix[row][col] == '*':
                    matrix[row][col - len(stack):col] = ['#'] * len(stack)
                else:
                    matrix[row][d - len(stack):d] = ['#'] * len(stack)
                stack = []

    return [list(reversed(x)) for x in zip(*matrix)]


matrix = [['#', '#', '.', '.', '#', '.', '*', '.'],
          ['#', '#', '#', '.', '.', '#', '*', '*'],
          ['#', '#', '#', '*', '.', '#', '.', '.']]
rotate_and_fall(matrix)


# border sort
def border_sort(matrix):
    # matrix size n x n
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    n = len(matrix)
    layers = n // 2 + 1
    matrix_t = [list(x) for x in zip(*matrix)]
    for layer in range(layers):
        cur_border = matrix[layer][layer:n - layer] + \
                     matrix[n - layer - 1][layer:n - layer] + \
                     matrix_t[layer][layer + 1:n - layer - 1] + \
                     matrix_t[n - layer - 1][layer + 1:n - layer - 1]
        print(cur_border)
        cur_border.sort()
        matrix[layer][layer:n - layer] = cur_border[0:n - 2 * layer]
        for i, j in zip(range(layer + 1, n - layer - 1), cur_border[n - 2 * layer:2 * n - 4 * layer - 2]):
            matrix[i][n - layer - 1] = j
        matrix[n - layer - 1][layer:n - layer] = cur_border[2 * n - 4 * layer - 2:3 * n - 6 * layer - 2][::-1]
        for i, j in zip(range(n - layer - 2, layer, -1), cur_border[3 * n - 6 * layer - 2:]):
            matrix[i][layer] = j
    return matrix


# problem 38 count sub segments
def count_sub_segments(nums):
    # calculate the prefix sum
    prefix = [0 for _ in range(len(nums))]
    prefix[0] = nums[0]
    for i in range(1, len(prefix)):
        prefix[i] = prefix[i - 1] + nums[i]

    # calculate the suffix sum
    suffix = [0 for _ in range(len(nums))]
    suffix[-1] = nums[-1]
    for i in range(len(suffix) - 1, 0, -1):
        suffix[i - 1] = suffix[i] + nums[i - 1]

    p = 1
    s = 1
    count = 0
    cur_sum = 0
    while p < len(prefix) - 1 and s < len(suffix) - 1:
        while s < len(suffix) - 1:
            cur_sum += nums[s]
            s += 1
            if prefix[p - 1] <= cur_sum <= suffix[s]:
                count += 1
        cur_sum = 0
        p += 1
        s = p
    return count


count_sub_segments([2, 4, 1, 1, 6, 15])
count_sub_segments([0, 1, 2])
count_sub_segments([2, 4, 1, 6, 3])


# problem 12
def int_sum(num):
    return sum((-1) ** i * int(j) for i, j in enumerate(str(num)))


# problem 16
def largest_square_in_histogram(heights):
    optimal = 0
    for i in range(len(heights)):
        step = 0
        min_height = heights[i]
        while i + step < len(heights) and min_height >= step + 1:
            step += 1
            if i + step >= len(heights):
                break
            min_height = min(min_height, heights[i + step])
        optimal = max(optimal, step ** 2)
    return optimal


assert largest_square_in_histogram([4, 3, 3, 3, 2]) == 9
assert largest_square_in_histogram([4, 5, 5, 4, 3, 3, 3, 2, 1]) == 16
assert largest_square_in_histogram([1, 1]) == 1
assert largest_square_in_histogram([1]) == 1
assert largest_square_in_histogram([2, 2]) == 4
assert largest_square_in_histogram([3, 3, 3]) == 9
assert largest_square_in_histogram([]) == 0


# problem 41
def maximum_arithmetic_sequence_length(a, b):
    min_diff, optimal = min(a[i + 1] - a[i] for i in range(len(a) - 1)), 0
    for d in range(min_diff + 1):
        cur_sequence, i, j = [a[0]], 1, 0
        while a[0] > b[j]:
            j += 1
        while i < len(a) or j < len(b):
            if i == len(a):
                if len(cur_sequence) < len(a):
                    cur_sequence = []
                    break
                if len(cur_sequence) >= len(a) and b[j] - cur_sequence[-1] == d:
                    cur_sequence.append(b[j])
                    j += 1
                else:
                    j += 1
            elif j == len(b):
                if a[i] - cur_sequence[-1] == d:
                    cur_sequence.append(a[i])
                    i += 1
                else:
                    cur_sequence = []
                    break
            elif a[i] - cur_sequence[-1] == d:
                cur_sequence.append(a[i])
                i += 1
            elif b[j] - cur_sequence[-1] == d:
                cur_sequence.append(b[j])
                j += 1
            else:
                j += 1
        current_len = len(cur_sequence)
        if cur_sequence:
            j = 0
            while a[0] > b[j]:
                j += 1
            if j != 0:
                last = a[0]
                for item in b[:j][::-1]:
                    if last - item == d:
                        last = item
                        current_len += 1
        optimal = max(optimal, current_len)
    return optimal if optimal else -1


assert maximum_arithmetic_sequence_length([1, 4, 6, 8, 9], [1, 2, 3, 5, 7, 10, 12]) == 10
assert maximum_arithmetic_sequence_length([3, 5, 9], [1, 7, 11, 13]) == 7


# prefix abc
def prefix_appearance(a, b):
    rec = set()
    [rec.add(''.join(a[i:j])) for i in range(len(a)) for j in range(i + 1, len(a))]
    for s in b:
        if s not in rec:
            return False
    return True


# problem 22: sum in lower and upper range
def sum_in_range(a, b, lower, upper):
    if len(a) <= len(b):
        s_nums = sorted(x ** 2 for x in a)
        uns_nums = [x ** 2 for x in b]
    else:
        s_nums = sorted(x ** 2 for x in b)
        uns_nums = [x ** 2 for x in a]

    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > target:
                right = mid - 1
            elif arr[mid] < target:
                left = mid + 1
            elif arr[mid] == target:
                return mid
        return right

    count = 0
    for num in uns_nums:
        if lower <= num <= upper:
            lower_s = binary_search(s_nums, lower - num)
            upper_s = binary_search(s_nums, upper - num)
            if lower_s != -1 and upper_s != -1 and (s_nums[lower_s] == lower - num or s_nums[upper_s] == upper - num):
                count += upper_s - lower_s + 1
            else:
                count += upper_s - lower_s
    return count


# problem 23: dividing string
def divide_string(s):
    if len(s) % 2 != 0:
        return 0
    f, s = set(), set()
    for i, j in enumerate(s):
        if i < len(s) // 2:
            if j in f:
                return 0
            else:
                f.add(j)
        else:
            if j in s:
                return 0
            else:
                s.add(j)
    return s[:len(s) // 2], s[len(s) // 2:]


# problem 21 - (1):
def jump_sum(a, b):
    count = 0
    for i in range(len(a)):
        count += 1 if a[i] + (a[i + 2] if i + 2 < len(a) else '') + (a[i + 4] if i + 4 < len(a) else '') == b else 0
    return count


# problem 21 - (2):
def count_arithmetic_mean(nums):
    count = 0
    for i in range(len(nums)):
        count += 1 if 2 * nums[i] == (nums[i - 1] if i - 1 >= 0 else 0) + (
            nums[i + 1] if i + 1 < len(nums) else 0) else 0
    return count


# problem 25
def fill_rectangular(queries):
    n = len(queries)
    rectangular = set()
    results = []
    for idx, rect in enumerate(queries):
        indicator, length, width = rect
        if indicator == 0:
            rectangular.add((length, width))
        else:
            res = False
            for r in rectangular:
                if min(r) >= min(length, width) and max(r) >= max(length, width):
                    res = True
                    break
            results.append(res)
            if 0 <= idx + 1 < n and queries[idx + 1][0] == 0:
                rectangular = set()
    return results


assert fill_rectangular([[0, 1, 1], [0, 2, 3], [1, 2, 2], [1, 3, 4], [0, 3, 4]]) == [True, False]
assert fill_rectangular([[0, 1, 1], [0, 2, 3], [1, 2, 2], [1, 3, 4], [0, 3, 4], [1, 3, 4]]) == [True, False, True]


# problem 26:
def min_col_row(n, m, queries):
    row_sequence = [i for i in range(n, 0, -1)]
    col_sequence = [i for i in range(m, 0, -1)]
    results = []

    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > target:
                left = mid + 1
            elif arr[mid] < target:
                right = mid - 1
            else:
                return mid
        return -10

    for query in queries:
        if query[0] == 0:
            results.append(row_sequence[-1] * col_sequence[-1])
        elif query[0] == 1:
            r_idx = binary_search(row_sequence, query[1] + 1)
            if r_idx != -10:
                row_sequence = row_sequence[:r_idx] + row_sequence[r_idx + 1:]
        elif query[0] == 2:
            r_idx = binary_search(col_sequence, query[1] + 1)
            if r_idx != -10:
                col_sequence = col_sequence[:r_idx] + col_sequence[r_idx + 1:]
    return results


assert min_col_row(3, 4, [[0], [1, 0], [0], [2, 1], [0]]) == [1, 2, 2]


# problem 27:
def find_minimum_number(queries):
    if not queries:
        return 0
    count = 1
    queries.sort()
    left, right = queries[0]
    for i in range(1, len(queries)):
        if queries[i][0] < right:
            left, right = max(left, queries[i][0]), min(right, queries[i][1])
        elif queries[i][0] == right:
            left = right
        elif queries[i][0] > right:
            count += 1
            left, right = queries[i]
    return count


# problem 28
def add_two_strings(a, b):
    a, b = list(a), list(b)
    result = []
    while a and b:
        result.append(str(int(a.pop()) + int(b.pop())))
    while a:
        result.append(a.pop())
    while b:
        result.append(b.pop())
    result.reverse()
    return ''.join(result)


# problem 29 - 1
def find_even_nums(nums):
    count = 0
    for num in nums:
        count += 1 if len(str(num)) % 2 == 0 else 0
    return count


def find_specific_nums(n):
    if n < 10:
        return 3 if n >= 4 else 2 if 2 <= n < 4 else 1
    count = 0
    sn = [int(x) for x in str(n)]
    for i in range(len(sn)):
        if sn[i] > 4:
            multi = sn[i] - 2 if i == 0 or i == len(sn) - 1 else sn[i] - 3
        elif sn[i] == 3:
            multi = 2 if i == 0 or i == len(sn) - 1 else 1
        else:
            if i == 0:
                multi = 3 if sn[i] == 4 else 2
            else:
                multi = 2 if sn[i] == 4 else 1 if sn[i] == 2 else 0
            count += multi * 7 ** (len(sn) - i - 1)
            break
        count += multi * 7 ** (len(sn) - i - 1)
    return 1 + n - count


n = 36
count = 1
for i in range(1, n + 1):
    count += 1 if '0' in str(i) or '2' in str(i) or '4' in str(i) else 0
print(count)


# problem 15


# problem 18 prefix string
def prefix_string_pairs(s):
    tree = {}
    pairs = 0
    s = sorted(s, key=lambda x: -len(x))

    def insert_tree(tree, item):
        cur = tree
        for i in range(len(item)):
            if item[i] in cur:
                cur[item[i]][1] += 1
            else:
                cur[item[i]] = [{}, 1]
            cur = cur[item[i]][0]

    def is_in_tree(tree, item):
        cur = tree
        count = 0
        for i in range(len(item)):
            if item[i] in cur:
                count = cur[item[i]][1]
                cur = cur[item[i]][0]
            else:
                return 0
        return count

    for i in s:
        cur = is_in_tree(tree, i)
        if cur == 0:
            insert_tree(tree, i)
        else:
            pairs += cur
            insert_tree(tree, i)
    return pairs


# problem 17:
def equally_rearranging(s):
    result = ''
    record = {'W': 0, 'D': 0, 'L': 0}
    for i in s:
        record[i] += 1
    while True:
        cur = ''.join([x for x in 'WDL' if record[x] != 0])
        if not cur:
            break
        min_count = min(record[x] for x in cur)
        result += cur * min_count
        for i in cur:
            record[i] -= min_count
    return result


print(equally_rearranging('LDWDL'))


# max freq in sub array
def max_freq(nums, k):
    """

    :param nums:
    :param k:
    :return:
    """
    left, right = 0, 0
    rec = {}
    result = []
    while right < len(nums):
        if nums[right] in rec:
            rec[nums[right]] += 1
        else:
            rec[nums[right]] = 1
        if right - left + 1 == k:
            result.append(max(rec.values()))
            rec[nums[left]] -= 1
            left += 1
        right += 1
    return result


# problem 36 maximum sub matrix sum
def maximum_sub_matrix_sum(m, k):
    optimal = float('-inf')
    optimal_set = set()
    n, d = len(m), len(m[0])
    for i in range(n - k + 1):
        for j in range(d - k + 1):
            sub_m = sum([x[j:j + k] for x in m[i:i + k]], [])
            sub_m_sum = sum(sub_m)
            if sub_m_sum > optimal:
                optimal = sub_m_sum
                optimal_set = set(sub_m)
            elif sub_m_sum == optimal:
                for item in sub_m:
                    optimal_set.add(item)
    return sum(optimal_set)


# problem binary string addition
def add2strings(bs, queries):
    r = {'0': 0, '1': 0}
    result = []
    bs = [x for x in bs]
    for c in bs:
        r[c] += 1
    for query in queries:
        if query == "?":
            result.append(r['1'])
        else:
            if bs[-1] == '0':
                r['1'] += 1
                r['0'] -= 1
                bs[-1] = '1'
            elif r['0'] == 0 and bs[-1] == '1':
                r['1'] = 1
                bs = ['1'] + bs
                r['0'] = len(bs) - 1
            else:
                one_count = 0
                for i in range(len(bs) - 1, -1, -1):
                    if bs[i] == '1':
                        one_count += 1
                        bs[i] = '0'
                    else:
                        bs[i] = '1'
                        break
                r['1'] += 1 - one_count
                r['0'] += one_count - 1
    return result


add2strings('1101', ['?', '+', '?', '+', '?', '+', '?'])


# jumping integer
def jump_int(nums, k):
    record = {}
    for i in range(len(nums)):
        if i not in record:
            record[i] = 1
        if i + k < len(nums) and nums[i + k] == nums[i]:
            record[i + k] = record[i] + 1
    return max(record.values())


# number of subarrays with distinct appearance of numbers
def num_distinct_sub_arrays(nums, k):
    pass


# micro service
def micro_services(subscribe, event, end):
    event_fact = {}
    for query in event:
        event_fact[query[-1]] = {'start': query[0], 'freq': query[1]}
    subscribe_record = {x: [] for x in event_fact}
    signal_count = {x: 0 for x in event_fact}
    subscribe = sorted([s for s in subscribe if s[1] <= end], key=lambda x: x[1])
    for query in subscribe:
        start = event_fact[query[2]]['start']
        freq = event_fact[query[2]]['freq']
        if query[0] == 'subscrib':
            subscribe_record[query[2]].append(query[1])
        else:
            subscribe_time = max(subscribe_record[query[2]].pop(), start)
            signal_count[query[2]] += (query[1] - start) // freq - (subscribe_time - start) // freq
            signal_count[query[2]] += 1 if (subscribe_time - start) % freq == 0 and (
                    query[1] - start) % freq == 0 else 0
    for key in subscribe_record:
        if subscribe_record[key]:
            start = event_fact[key]['start']
            freq = event_fact[key]['freq']
            subscribe_time = max(subscribe_record[key].pop(), start)
            signal_count[key] += (end - start) // freq - (subscribe_time - start) // freq
            signal_count[key] += 1 if (subscribe_time - start) % freq == 0 and (end - start) % freq == 0 else 0
    return signal_count


subscribes = [["subscrib", 0, 1], ["unsubscrib", 15, 1], ["subscrib", 7, 2]]
events = [[0, 5, 1], [10, 2, 2]]
end = 20

print(micro_services(subscribes, events, end))


# draw star matrix
def star_matrix(n):
    return ['*' * n if i == 0 or i == n - 1 else '*' + ' ' * (n - 2) + '*' for i in range(n)]


# diagnoal sort
def diagnoal_sort(m):
    m = [[6, 3, 1, 4, -2],
         [1, 8, -2, 3, 4],
         [3, 7, 3, 1, 3]]
    n, d = len(m), len(m[0])
    max_len = min(n, d)
    rec = {}
    for i in range(n):
        for j in range(d):
            if m[i][j] in rec:
                rec[m[i][j]] += 1
            else:
                rec[m[i][j]] = 1
    rec = rec.items()
    rec = sorted(rec, key=lambda x: x[0])
    rec.sort(key=lambda x: x[1])
    rec = sum([[x[0]] * x[1] for x in rec], [])
    for j in range(d):
        cur_len = min(j + 1, max_len)
        sub_rec = [rec.pop() for _ in range(cur_len)]
        for idx, item in enumerate(sub_rec):
            m[idx][j - idx] = item
    for i in range(1, n):
        cur_len = min(n - i, max_len)
        sub_rec = [rec.pop() for _ in range(cur_len)]
        for idx, item in enumerate(sub_rec):
            m[i + idx][d - idx - 1] = item
    return m


# Quora round 2
# Serialize and Deserialize Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class CodecDFS:
    def serialize1(self, root):
        # version 1
        def dfs(node, s):
            if not node:
                s += 'None,'
            else:
                s += str(node.val) + ','
                s = dfs(node.left, s)
                s = dfs(node.right, s)
            return s

        return dfs(root, '')

    def serialize2(self, root):
        # version 2
        self.result = ''

        def dfs(node):
            if not node:
                self.result += 'None,'
            else:
                self.result += str(node.val) + ','
                dfs(node.left)
                dfs(node.right)
            return s

        dfs(root)
        return self.result

    def deserialize(self, data):
        node_values = data.split(',')
        node_values.reverse()

        def dfs(nodes):
            if nodes[-1] == 'None':
                nodes.pop()
                return None
            node = TreeNode(int(nodes.pop().val))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node

        return dfs(node_values)


from collections import deque


class CodecBFS:
    def serialize(self, root):
        if not root:
            return 'None,'
        queue, result = deque([root]), ''
        while queue:
            cur = queue.popleft()
            if not cur:
                result += 'None,'
                continue
            result += str(cur.val) + ','
            queue.append(cur.left)
            queue.append(cur.right)
        return result

    def deserialize(self, data):
        node_values = data.split(',')
        if node_values[0] == 'None':
            return None
        root = TreeNode(int(node_values[0]))
        queue = deque([root])
        i = 1
        while queue and i < len(node_values):
            cur = queue.popleft()
            if node_values[i] != 'None':
                left = TreeNode(int(node_values[i]))
                cur.left = left
                queue.append(left)
            i += 1
            if node_values[i] != 'None':
                cur.right = TreeNode(int(node_values[i]))
                queue.append(cur.right)
            i += 1
        return root


# find peak element
def binary_search(nums, left, right, value):
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


def find_peak(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left


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

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        from random import choice
        return choice(self.list)


# unique path
def unique_path(m, n):
    dp = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def unique_path_ii(grid):
    if grid[0][0] or grid[-1][-1]:
        return 0
    n, d = len(grid), len(grid[0])
    dp = [[0 if grid[i][j] else 1 for i in range(n)] for j in range(d)]
    for i in range(n):
        for j in range(d):
            if dp[i][j] == 0:
                continue
            elif i == 0:
                dp[i][j] = dp[i][j - 1]
            elif j == 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def unique_path_iii(grid):
    n, d = len(grid), len(grid[0])
    si, sj, num_spaces, num_paths = 0, 0, 0, 0
    for i in range(n):
        for j in range(d):
            if grid[i][j] != -1:
                num_spaces += 1
            if grid[i][j] == 1:
                si, sj = i, j

    def backtrack(ci, cj, num_remains):
        nonlocal num_paths
        if grid[ci][cj] == 2 and num_remains == 1:
            num_paths += 1
            return
        cc = grid[ci][cj]
        grid[ci][cj] = -2
        num_remains -= 1
        for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            if 0 <= ci + di < n and 0 <= cj + dj < d and grid[ci + di][cj + dj] >= 0:
                backtrack(ci + di, cj + dj, num_remains)
        grid[ci][cj] = cc

    backtrack(si, sj, num_spaces)
    return num_paths


# sum with 1: [1, 1, 1] -> [1, 1, 3] -> [1, 5, 3] -> [1, 5, 9], 3 times of transformation
def sum_with_one(queries):
    import heapq
    optimal = float('+Inf')
    for query in queries:
        count, valid = 0, True
        query = [-x for x in query[::-1]]
        cur_sum = sum(query)
        while cur_sum + len(query) != 0:
            cur_max = heapq.heappop(query)
            cur_gap = cur_sum - cur_max
            cur_max -= cur_gap
            cur_sum -= cur_gap
            heapq.heappush(query, cur_max)
            print(cur_gap, cur_max, cur_sum, query)
            if cur_max >= 0:
                valid = False
                break
            count += 1
        if valid:
            optimal = min(optimal, count)
    return optimal


def search_rotate(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[left] == target:
            return left
        elif nums[mid] < nums[right]:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid
        else:
            if nums[left] < target <= nums[mid]:
                right = mid
            else:
                left = mid + 1
    return left if nums[left] == target else -1


def search_rotation_with_duplicate(nums, target):
    if nums[0] == target:
        return True
    while nums[0] != nums[-1]:
        nums.pop()
    if not nums:
        return False
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return True
        if nums[mid] >= nums[left]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


from collections import deque
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

    def get_avg_in_past_hour(self, prev_timestamp, timestamp=None, ):
        if not self.sample:
            return 0.0
        if not timestamp:
            timestamp = time()
        # prev_timestamp = timestamp - time_span
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

    def get_avg_in_past_hour(self, ):
        if not self.sample:
            return 0.0
        while self.sample:
            diff = time() - self.sample[0][1]
            if diff >= 3600:
                es = self.sample.popleft()
                self.total -= es[0]
            else:
                break
        return self.total / len(self.sample)


# kmp algorithm - find substring in another string
class KMP:
    def partial(self, s):
        result = [0]
        for c in s[1:]:
            idx = result[-1]
            if s[idx] == c:
                result.append(idx + 1)
            else:
                result.append(0)
        return result

    def find(self, s1, s2):
        ps, j, result = self.partial(s2), 0, []
        for i in range(len(s1)):
            while j > 0 and s1[j] != s2[i]:
                j = ps[j - 1]
            if s1[j] == s1[i]:
                j += 1
            if j == len(s2):
                result.append(i - (j - 1))
                j = ps[j - 1]
        return result


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


# path sum
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def has_path_sum_dfs(root, target):
    flag = False

    def dfs(node, v):
        if node is None:
            return
        if node.left is None and node.right is None:
            if v + node.val == target:
                nonlocal flag
                flag = True
            return
        dfs(node.left, v + node.val)
        dfs(node.right, v + node.val)

    dfs(root, 0)
    return flag


def has_path_sum_bfs(root, targetSum):
    if not root:
        return False
    stack = [(root, 0)]
    while stack:
        cur_node, cur_sum = stack.pop()
        cur_sum += cur_node.val
        if cur_node.left is None and cur_node.right is None and cur_sum == targetSum:
            return True
        if cur_node.left:
            stack.append((cur_node.left, cur_sum))
        if cur_node.right:
            stack.append((cur_node.right, cur_sum))
    return False


def path_sum_ii_dfs(root, target):
    """
    :param root:
    :param target:
    :return:
    """
    results = []

    def dfs(node, v, path):
        if node is None:
            return
        if node.right is None and node.right is None and node.val + v == target:
            nonlocal results
            results.append(path + [node.val])
            return
        dfs(node.left, v + node.val, path + [node.val])
        dfs(node.right, v + node.val, path + [node.val])

    dfs(root, 0, [])
    return results


def sum_to_k(nums, k):
    count, cur_sum = 0, 0
    record = defaultdict(int)
    for num in nums:
        cur_sum += num
        if cur_sum == k:
            count += 1
        count += record[cur_sum - k]
        record[cur_sum] += 1
    return count


def path_sum_iii(root, target):
    count, record = 0, defaultdict(int)

    def dfs(node, v):
        nonlocal count
        if not node:
            return
        v += node.val
        if v == target:
            count += 1
        count += record[v - target]
        record[v] += 1
        dfs(node.left, v)
        dfs(node.right, v)
        record[v] -= 1

    dfs(root, 0)
    return count


# longest common prefix string
def longest_common_prefix(strs):
    cur_common = strs[0]
    for string in strs[1:]:
        new_common = ''
        for c1, c2, in zip(string, cur_common):
            if c1 == c2:
                new_common += c1
            else:
                break
        cur_common = new_common
    return cur_common


# meeting rooms i
def meeting_rooms_i(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False
    return True


import heapq


def meeting_rooms_ii(intervals):
    intervals.sort()
    queue, count = [intervals[0][1]], 0
    for interval in intervals[1:]:
        if queue[0] <= interval[0]:
            heapq.heappop(queue)
        heapq.heappush(queue, interval[1])
    return len(queue)


def subset_sum_recursive(nums, k):
    results = []

    def backtrack(idx, s, path):
        if s == 0 and path:
            results.append(path[:])
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, s - nums[i], path)
            path.pop()

    backtrack(0, k, [])
    return results


subset_sum_recursive([1, 2, 3, 4, 5, 6], 7)


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


# find max unconsecutive sum in an array with duplications
def max_unconsecutive_sum(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0 for _ in range(nums)]
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        if abs(nums[i] - nums[i - 1]) == 1:
            k = i - 1
            while abs(nums[i] - nums[k]) == 1 and k > 0:
                k -= 1
            if abs(nums[i] - nums[k]) == 1:
                dp[i] = max(nums[i], dp[i - 1])
            else:
                dp[i] = max(dp[k] + nums[i], dp[i - 1])
        else:
            dp[i] = dp[i - 1] + nums[i]
    return dp[-1]


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


# union 2 rectangle
def union_two_rectangles(rectangle1, rectangle2):
    rectangle1.sort()
    rectangle2.sort()
    if rectangle1[0][0] <= rectangle2[0][0]:
        (x1, y1), (x2, y2) = rectangle1
        (x3, y3), (x4, y4) = rectangle2
    else:
        (x1, y1), (x2, y2) = rectangle2
        (x3, y3), (x4, y4) = rectangle1
    if x3 < x2 and not y4 >= y1 and not y3 <= y2:
        mask = abs(max(x1, x3) - min(x2, x4)) * abs(max(y2, y4) - min(y1, y3))
    else:
        mask = 0
    return abs(x1 - x2) * abs(y1 - y2) + abs(x3 - x4) * abs(y3 - y4) - mask


assert union_two_rectangles([[1, 4], [3, 1]], [[2, 3], [4, 2]]) == 7


def union_three_rectangles(rectangles):
    x = set()
    queries = []
    for x1, y1, x2, y2 in rectangles:
        x.add(x1)
        x.add(x2)
        queries.append((y1, 1, x1, x2))
        queries.append((y2, -1, x1, x2))
    # array of x coordinates in left to right order
    i_to_x = list(sorted(x))
    # inverse dictionary maps x coordinate to its rank
    x_to_i = {xi: i for i, xi in enumerate(i_to_x)}
    # number of current rectangles intersected by the sweepline in interval [i_to_x[i], i_to_x[i+1]]
    num_current_rectangles = [0] * (len(i_to_x) - 1)
    area = 0
    length_union_intervals = 0
    previous_y = 0  # arbitrary initial value. because length is 0 at first iteration
    for y, offset, x1, x2 in sorted(queries):
        area += (y - previous_y) * length_union_intervals
        i1, i2 = x_to_i[x1], x_to_i[x2]  # update number of current rectangles that are intersected
        for j in range(i1, i2):
            length_interval = i_to_x[j + 1] - i_to_x[j]
            if num_current_rectangles[j] == 0:
                length_union_intervals += length_interval
            num_current_rectangles[j] += offset
            if num_current_rectangles[j] == 0:
                length_union_intervals -= length_interval
        previous_y = y
    return area


# stock price input and output
class StockPrice:
    def __init__(self, k):
        self.window_size = k
        self.size = 0
        self.price_flow = deque([])

    def put(self, price):
        if self.window_size <= len(self.price_flow):
            self.price_flow.popleft()
        while self.price_flow and price > self.price_flow[-1]:
            self.price_flow.pop()
        self.price_flow.append(price)

    def get_price(self):
        return self.price_flow[0]


# find the missing 2 values given range 1-N
def find_missing_gaussian(nums, n):
    s = n*(n+1)/2 - sum(nums)
    avg = s / 2
    first_half = 0
    second_half = 0
    for num in nums:
        if num <= avg:
            first_half += num
        else:
            second_half += num
    total_first_half = int(avg) * (int(avg) + 1) / 2
    missing_1 = total_first_half - first_half
    missing_2 = s - missing_1
    return missing_1, missing_2


def find_missing_xor(nums, n):
    xor = nums[0]
    for num in nums[1:]:
        xor ^= num
    for i in range(1, n+1):
        xor ^= i
    bit_set_pos = xor & ~(xor - 1)
    x, y = 0, 0
    for num in nums:
        if num & bit_set_pos:
            x = x ^ num
        else:
            y = y ^ num
    for i in range(1, n+1):
        if i & bit_set_pos:
            x ^= i
        else:
            y ^= i
    return x, y


# prime number multiplication
def prime_number_multiplication(nums):
    results = []

    def backtrack(idx, p):
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            p *= nums[i]
            results.append(p)
            backtrack(i + 1, p)
            p //= nums[i]

    backtrack(0, 1)
    return results


prime_number_multiplication([2,3,5])


def prime_number_multiplication_with_duplication(nums):
    results = []
    nums.sort()

    def backtrack(idx, path):
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            results.append(path[:])
            backtrack(i+1, path)
            path.pop()

    backtrack(0, [])
    return results


prime_number_multiplication_with_duplication([2,2,3])


# merging k sorted arrays
def merge_k_sorted_arrays(arrays):
    ans, heap = [], []
    for i, array in enumerate(arrays):
        heapq.heappush(heap, (array[0], i, 0))
    while heap:
        num, array_idx, item_idx = heapq.heappop(heap)
        ans.append(num)
        if item_idx < len(arrays[array_idx]) - 1:
            heapq.heappush(heap, (arrays[array_idx][item_idx + 1], array_idx, item_idx + 1))
    return ans


#  given a binary tree and K, count the number of paths starting from the root and summing to K.
def path_sum_to_k(root, k):
    if not root:
        return 0

    count = 0

    def dfs(node, s):
        nonlocal count
        if node is None:
            return
        s += node.val
        if s == k:
            count += 1
        dfs(node.left, s)
        dfs(node.right, s)

    dfs(root, 0)
    return count


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


# hit bricks
def hit_bricks(grid, hits):
    # if we do it reversely of hits, it will take O(MN) + O(K) since we can utilize the known information,
    # all points will be at most be visited twice (marked as unstable + mark as stable)
    results = []
    ny, nx = len(grid), len(grid[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(y, x):
        if grid[y][x] != 1:
            return 0
        grid[y][x], ans = 2, 1  # mark all the bricks that are directly or indirectly connected to the top wall as 2
        for dy, dx in directions:
            new_x, new_y = x + dx, y + dy
            # recursive DFS
            if 0 <= new_x < nx and 0 <= new_y < ny and grid[new_y][new_x] == 1:
                ans += dfs(new_y, new_x)
        return ans

    def is_stable(y, x):
        grid[y][x] += 1  # add the hit brick back to the graph
        if grid[y][x] <= 0:  # to avoid being hit too many times
            return False
        if y == 0 and grid[y][x] == 1 or grid[y][x] == 2:
            return True
        for dy, dx in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < nx and 0 <= new_y < ny and grid[new_y][new_x] == 2:
                return True
        return False

    # remove the hit bricks first, later add them back
    for y, x in hits:
        grid[y][x] -= 1

    # mark all the bricks as 2
    for x in range(nx):
        dfs(0, x)

    for y, x in hits[::-1]:
        if is_stable(y, x):
            results.append(dfs(y, x) - 1)
        else:
            results.append(0)

    return results[::-1]


# max possible number from two lists preserve order
def max_number(nums1, nums2, k):
    def merge(l1, l2):
        results = []
        while l1 and l2:
            if l1 < l2:
                results.append(l2[0])
                l2.pop(0)
            else:
                results.append(l1[0])
                l1.pop(0)
        if l1 or l2:
            results += l1 + l2
        return results

    def find_max(nums, length):
        # find the maximal number constrained by length and relative order
        stack = []
        max_out = len(nums) - length
        for i in range(len(nums)):
            while max_out and stack and nums[i] > stack[-1]:
                stack.pop()
                max_out -= 1  # to preserve the order as well as the length
            stack.append(nums[i])
        return stack[:length]  # return the required length

    optimal = [0 for _ in range(k)]
    for i in range(k + 1):
        j = k - i
        stack1 = find_max(nums1, i)
        stack2 = find_max(nums2, j)
        if len(stack1) + len(stack2) == k:
            optimal = max(optimal, merge(stack1, stack2))
    return optimal


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


# construct binary tree with in-order and pre-order traversal
def build_tree(preorder, inorder):
    # the pre-order sequence always starts with root node while the in-order sequence is always
    # split by the root node, i.e., the root node is always located in the middle of the tree
    p_idx = 0
    x_to_i = {xi: i for i, xi in enumerate(inorder)}  # build a mapping from value to index

    def recursion(left, right):
        nonlocal p_idx  # call the index pointer for pre-order array
        if left > right:  # reached a leaf node, no subtree anymore
            return None
        cur_val = preorder[p_idx]
        root = TreeNode(cur_val)
        p_idx += 1
        root.left = recursion(left, x_to_i[cur_val] - 1)  # fetch the corresponding index from in-order array with O(1)
        root.right = recursion(x_to_i[cur_val] + 1, right)
        return root

    return recursion(0, len(preorder) - 1)


# wildcard matching
def wildcard(s, p):
    dp = [[False for _ in range(len(s) + 1)] for _ in range(len(p) + 1)]
    dp[0][0] = True

    for i in range(1, len(p) + 1):
        dp[i][0] = dp[i-1][0] if p[i-1] == '*' else False

    for i in range(1, len(p) + 1):
        for j in range(1, len(s) + 1):
            if p[i-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            if p[i-1] == s[j-1] or p[j-1] == '?':
                dp[i][j] = dp[i-1][j-1]
    return dp[-1][-1]


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


# max values in the sliding windows
def max_sliding_window(nums, k):
    if not nums:
        return []
    if k == 1:
        return nums

    def clean_queue(queue, i):
        if queue and queue[0] == i - k:  # window moves forward
            queue.popleft()
        while queue and nums[i] > nums[queue[-1]]:
            queue.pop()
        return queue

    deq = deque([])

    for i in range(k):
        deq = clean_queue(deq, i)
        deq.append(i)

    max_values = [nums[deq[0]]]

    for i in range(k, len(nums)):
        deq = clean_queue(deq, i)
        deq.append(i)
        max_values.append(nums[deq[0]])

    return max_values