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