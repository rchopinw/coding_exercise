# kmp algorithm - find substring in another string
class KMP:
    def partial(self, s):
        result = [0]
        for c in s[1:]:
            idx = result[-1]
            if s[idx] == c:
                result.append(idx + 1)
            else:
                result.append(0)
        return result

    def find(self, s1, s2):
        ps, j, result = self.partial(s2), 0, []
        for i in range(len(s1)):
            while j > 0 and s1[j] != s2[i]:
                j = ps[j - 1]
            if s1[j] == s1[i]:
                j += 1
            if j == len(s2):
                result.append(i - (j - 1))
                j = ps[j - 1]
        return result
