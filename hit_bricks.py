# hit bricks
def hit_bricks(grid, hits):
    # if we do it reversely of hits, it will take O(MN) + O(K) since we can utilize the known information,
    # all points will be at most be visited twice (marked as unstable + mark as stable)
    results = []
    ny, nx = len(grid), len(grid[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(y, x):
        if grid[y][x] != 1:
            return 0
        grid[y][x], ans = 2, 1  # mark all the bricks that are directly or indirectly connected to the top wall as 2
        for dy, dx in directions:
            new_x, new_y = x + dx, y + dy
            # recursive DFS
            if 0 <= new_x < nx and 0 <= new_y < ny and grid[new_y][new_x] == 1:
                ans += dfs(new_y, new_x)
        return ans

    def is_stable(y, x):
        grid[y][x] += 1  # add the hit brick back to the graph
        if grid[y][x] <= 0:  # to avoid being hit too many times
            return False
        if y == 0 and grid[y][x] == 1 or grid[y][x] == 2:
            return True
        for dy, dx in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < nx and 0 <= new_y < ny and grid[new_y][new_x] == 2:
                return True
        return False

    # remove the hit bricks first, later add them back
    for y, x in hits:
        grid[y][x] -= 1

    # mark all the bricks as 2
    for x in range(nx):
        dfs(0, x)

    for y, x in hits[::-1]:
        if is_stable(y, x):
            results.append(dfs(y, x) - 1)
        else:
            results.append(0)

    return results[::-1]


if __name__ == '__main__':
    grid1 = [[0, 1, 0, 1, 0], [0, 1, 0, 1, 1], [1, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
    hits1 = [[2,1], [1,4], [1,3]]

    grid2 = [[1,0,0,0],[1,1,1,0]]
    hits2 = [[1,0]]

    grid3 = [[1]]
    hits3 = [[0, 0]]

    grid5 = [[1, 0, 1]]
    hits5 = [[0, 0]]

    assert hit_bricks(grid1, hits1) == [3, 0, 1]
    assert hit_bricks(grid2, hits2) == [2]
    assert hit_bricks(grid3, hits3) == [0]
    assert hit_bricks(grid5, hits5) == [0]