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
