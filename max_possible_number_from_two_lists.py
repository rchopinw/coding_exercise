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


if __name__ == '__main__':
    assert max_number([4, 8, 6, 1], [7, 7, 9, 1], 5) == [9, 8, 6, 1, 1]

