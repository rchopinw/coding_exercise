# partition of palindromes
def partition_of_palindromes(s):
    results = []

    def backtrack(cur, path):
        if not cur:
            results.append(path)
            return
        for i in range(1, len(cur) + 1):
            if cur[:i] == cur[:i][::-1]:
                backtrack(cur[i:], path + [cur[:i]])

    backtrack(s, [])

    return results


def partition_of_palindromes_ii(s):
    dp = {i: [] for i in range(len(s) + 1)}
    dp[0] = [[]]
    for i in range(1, len(s) + 1):
        for j in range(i):
            for combination in dp[j]:
                if s[j:i] == s[j:i][::-1]:
                    dp[i].append(combination + [s[j:i]])
    return dp[len(s)]


# shortest palindrome
def shortest_palindrome(s):
    s_dual = s + '*' + s[::-1]
    dp = [0] * len(s_dual)
    for i in range(1, len(dp)):
        j = dp[i - 1]
        while j > 0 and s_dual[i] != s_dual[j]:
            j = dp[j - 1]
        if s_dual[i] == s_dual[j]:
            dp[i] = j + 1
    return s[::-1][:len(s) - dp[-1]] + s
