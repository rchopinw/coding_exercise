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
