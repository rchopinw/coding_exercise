# prime number multiplication
def prime_number_multiplication(nums):
    results = []

    def backtrack(idx, p):
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            p *= nums[i]
            results.append(p)
            backtrack(i + 1, p)
            p //= nums[i]

    backtrack(0, 1)
    return results


prime_number_multiplication([2,3,5])


def prime_number_multiplication_with_duplication(nums):
    results = []
    nums.sort()

    def backtrack(idx, path):
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            results.append(path[:])
            backtrack(i+1, path)
            path.pop()

    backtrack(0, [])
    return results


prime_number_multiplication_with_duplication([2,2,3])
