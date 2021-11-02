# wildcard matching
def wildcard(s, p):
    dp = [[False for _ in range(len(s) + 1)] for _ in range(len(p) + 1)]
    dp[0][0] = True

    for i in range(1, len(p) + 1):
        dp[i][0] = dp[i-1][0] if p[i-1] == '*' else False

    for i in range(1, len(p) + 1):
        for j in range(1, len(s) + 1):
            if p[i-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            if p[i-1] == s[j-1] or p[j-1] == '?':
                dp[i][j] = dp[i-1][j-1]
    return dp[-1][-1]
