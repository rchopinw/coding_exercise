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


if __name__ == '__main__':
    nums1 = [2, 2, 5, 5, 5, 5, 1, 1, 1, 1, 1, 20, 12, 8, 8, 8, 8, 8]
    assert max_unconsecutive_sum(nums1) == 97

    nums2 = [1, 1, 2, 2, 2, 2, 3, 3, 0]
    assert max_unconsecutive_sum(nums2) == 8



