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

