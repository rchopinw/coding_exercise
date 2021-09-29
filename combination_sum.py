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


# Approach 2: DP
def subset_sum_dp(nums, k):
    dp = {i: [] for i in range(k+1)}
    dp[0] = [[]]
    for num in nums:
        for cur_sum in range(num, k+1):
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

