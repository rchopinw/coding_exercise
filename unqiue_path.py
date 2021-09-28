# unique path
def unique_path(m, n):
    dp = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def unique_path_ii(grid):
    if grid[0][0] or grid[-1][-1]:
        return 0
    n, d = len(grid), len(grid[0])
    dp = [[0 if grid[i][j] else 1 for i in range(n)] for j in range(d)]
    for i in range(n):
        for j in range(d):
            if dp[i][j] == 0:
                continue
            elif i == 0:
                dp[i][j] = dp[i][j - 1]
            elif j == 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def unique_path_iii(grid):
    n, d = len(grid), len(grid[0])
    si, sj, num_spaces, num_paths = 0, 0, 0, 0
    for i in range(n):
        for j in range(d):
            if grid[i][j] != -1:
                num_spaces += 1
            if grid[i][j] == 1:
                si, sj = i, j

    def backtrack(ci, cj, num_remains):
        nonlocal num_paths
        if grid[ci][cj] == 2 and num_remains == 1:
            num_paths += 1
            return
        cc = grid[ci][cj]
        grid[ci][cj] = -2
        num_remains -= 1
        for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            if 0 <= ci + di < n and 0 <= cj + dj < d and grid[ci + di][cj + dj] >= 0:
                backtrack(ci + di, cj + dj, num_remains)
        grid[ci][cj] = cc

    backtrack(si, sj, num_spaces)
    return num_paths

