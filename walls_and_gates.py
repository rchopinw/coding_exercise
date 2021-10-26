from collections import deque


def wallsAndGates(rooms) -> None:
    """
    Do not return anything, modify rooms in-place instead.
    """
    doors = deque([])
    ny, nx = len(rooms), len(rooms[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(ny):
        for j in range(nx):
            if rooms[i][j] == 0:
                doors.append((i, j, 0))
    while doors:
        cur_y, cur_x, cur_distance = doors.popleft()
        rooms[cur_y][cur_x] = cur_distance
        for dx, dy in directions:
            new_x, new_y = cur_x + dx, cur_y + dy
            if 0 <= new_x < nx and 0 <= new_y < ny and rooms[new_y][new_x] > rooms[cur_y][cur_x]:
                rooms[new_y][new_x] = cur_distance + 1
                doors.append((new_y, new_x, cur_distance + 1))










































