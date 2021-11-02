def find_peak(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left


def binary_search(nums, left, right, value):
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left
