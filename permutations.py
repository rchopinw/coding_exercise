from collections import Counter


# generate all the anagrams from given string
def generate_anagrams_i(s):
    result = []

    def backtrack(cur_s, rec):
        if len(cur_s) == len(s):
            result.append(cur_s)
            return
        for i in range(len(s)):
            if rec[s[i]] > 0:
                rec[s[i]] -= 1
                backtrack(cur_s + s[i], rec)
                rec[s[i]] += 1

    backtrack('', Counter(s))
    return result


def generate_anagrams_ii(s):  # generate anagrams with duplication
    result = []

    def backtrack(cur_s, rec):
        if len(cur_s) == len(s):
            result.append(cur_s)
            return
        for c in rec:
            if rec[c] > 0:
                rec[c] -= 1
                backtrack(cur_s + c, rec)
                rec[c] += 1
    backtrack('', Counter(s))
    return result


# prime number multiplication
def prime_number_multiplication(nums):
    results = []

    def backtrack(idx, p):
        if idx == len(nums):
            return
        for i in range(idx, len(nums)):
            results.append(p)
            backtrack(i + 1, p * nums[i])
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


def prime_number_multiplication_ii(nums):
    result = [1]
    for num in nums:
        result += [x * num for x in result]
    return result[1:]
