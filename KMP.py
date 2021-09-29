# kmp algorithm - find substring in another string
class KMP:
    def _partial(self, s):
        dp = [0 for _ in range(len(s))]
        for i in range(1, len(s)):
            j = dp[i - 1]
            while j > 0 and s[i] != s[j]:
                j = dp[j - 1]
            if s[i] == s[j]:
                dp[i] = j + 1
        return dp

    def find(self, s1, s2):
        partial = self._partial(s2)
        j = 0
        for i in range(len(s1)):
            while j > 0 and s1[i] != s2[j]:
                j = partial[j - 1]
            if s1[i] == s2[j]:
                j += 1
                if j == len(s2):
                    return i - j + 1
        return -1

