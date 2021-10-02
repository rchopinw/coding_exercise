# LC39: Combination Sum
# Given an array of distinct integers candidates and a target integer target,
# return a list of all unique combinations of candidates where the chosen numbers sum to target.
# You may return the combinations in any order.
#
# The same number may be chosen from candidates an unlimited number of times.
# Two combinations are unique if the frequency of at least one of the chosen numbers is different.
#
# It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
#
# Input: candidates = [2, 3, 5], target = 8
# Output: [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
from collections import Counter


# Approach 1: recursive backtrack
def subset_sum_recursive(nums, k):  # not allow picking one item multiple times
    results = []

    def backtrack(idx, s, path):
        if s == 0 and path:
            results.append(path[:])
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            backtrack(i + 1, s - nums[i], path + [nums[i]])

    backtrack(0, k, [])
    return results


def subset_sum_recursive_ii(nums, k):  # allow picking one item multiple times
    results = []

    def backtrack(idx, path, s):
        if s == k:
            results.append(path)
        if s > k:
            return
        for i in range(idx, len(nums)):
            backtrack(i, path + [nums[i]], s + nums[i])

    backtrack(0, [], 0)
    return results


def subset_sum_recursive_iii(candidates, target):
    results = []
    candidates.sort()
    def backtrack(idx, path, s):
        if s == target:
            results.append(path)
        for i in range(idx, len(candidates)):
            if i > idx and candidates[i-1] == candidates[i]:
                continue
            backtrack(i + 1, path + [candidates[i]], s + candidates[i])
    backtrack(0, [], 0)
    return results


def subset_sum_recursive_iiii(candidates, target):
    results = []
    candidates.sort()
    def backtrack(space, path, s):
        if s == 0:
            results.append(path)
        for i in range(len(space)):
            if i > 0 and space[i-1] == space[i]:
                continue
            backtrack(space[i+1:], path + [space[i]], s - space[i])
    backtrack(candidates, [], target)
    return results


# Approach 2: DP
def subset_sum_dp(nums, k):
    dp = {i: [] for i in range(k + 1)}
    dp[0] = [[]]
    for num in nums:
        for cur_sum in range(num, k + 1):
            for cur_combination in dp[cur_sum - num]:
                dp[cur_sum].append(cur_combination + [num])
    return dp[k]


def subset_sum_dp_ii(nums, k):
    dp = [[False for _ in range(k + 1)] for _ in range(len(nums) + 1)]
    for i in range(len(nums) + 1):
        dp[i][0] = True
    for i in range(1, len(nums) + 1):
        for j in range(1, k + 1):
            if nums[i - 1] > j:
                dp[i][j] = dp[i - 1][j]  # if last combination can sum to k, then disregarding the current item
            else:
                # if last combination can sum to k or add current one can sum to k
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
    return dp[-1][-1]


if __name__ == '__main__':
    # tests for duplication-allowed cases
    # dp:
    assert subset_sum_dp([1, 2], 8) == [[1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 2],
                                        [1, 1, 1, 1, 2, 2],
                                        [1, 1, 2, 2, 2],
                                        [2, 2, 2, 2]]
    assert subset_sum_dp([1], 3) == [[1, 1, 1]]
    assert subset_sum_dp([], 9) == []
    assert subset_sum_dp([1, 3, 10], 0) == [[]]

    # recursion
    assert subset_sum_recursive_ii([1, 2], 8) == [[1, 1, 1, 1, 1, 1, 1, 1],
                                                  [1, 1, 1, 1, 1, 1, 2],
                                                  [1, 1, 1, 1, 2, 2],
                                                  [1, 1, 2, 2, 2],
                                                  [2, 2, 2, 2]]
    assert subset_sum_recursive_ii([1], 3) == [[1, 1, 1]]
    assert subset_sum_recursive_ii([], 9) == []
    assert subset_sum_recursive_ii([1, 2, 12], 0) == [[]]

    # tests for duplication-prohibited cases
    # dp:
    assert subset_sum_dp_ii([1, 5, 2, 3], 200) is False
    assert subset_sum_dp_ii([1, 0, 8, 23, 2, 15], 23)
    assert subset_sum_dp_ii([], 12) is False
    assert subset_sum_dp_ii([1, 3, 2, 4], 0)

    # recursion
    assert subset_sum_recursive([1, 5, 8, 7, 6, 13, 2], 13) == [[1, 5, 7], [5, 8], [5, 6, 2], [7, 6], [13]]
    assert subset_sum_recursive([1, 1, 1, 2], 3)