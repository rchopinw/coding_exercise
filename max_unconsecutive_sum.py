from collections import Counter


# find max unconsecutive sum in an array with duplications
def max_unconsecutive_sum(nums):
    count = Counter(nums)
    chose, avoid = 0, 0
    for num in sorted(count):
        if num - 1 in count:
            chose, avoid = avoid + num * count[num], max(chose, avoid)
        else:
            chose, avoid = max(chose, avoid) + num * count[num], max(chose, avoid)
    return max(chose, avoid)
