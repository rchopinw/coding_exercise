from collections import deque


# king walking on a chess board
def king_walking(board):
    # king is indicated as 1, target is indicated as 2, obstacles as -1, empty block as 0
    # only one king and one target on the board
    king_y, king_x, target_y, target_x = 0, 0, 0, 0
    ny, nx = len(board), len(board[0])
    directions = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not dx == dy == 0]
    for y in range(ny):
        for x in range(nx):
            if board[y][x] == 1:
                king_y, king_x = y, x
            elif board[y][x] == 2:
                target_y, target_x = y, x
    queue1 = deque([(king_y, king_x, 0)])
    queue2 = deque([(target_y, target_x, 0)])
    q1_visited = {(king_y, king_x): 0}
    q2_visited = {(target_y, target_x): 0}
    while queue1 and queue2:
        for _ in range(len(queue1)):
            y, x, d = queue1.popleft()
            for dy, dx in directions:
                new_y, new_x, new_d = y + dy, x + dx, d + 1
                if 0 <= new_y < ny and \
                        0 <= new_x < nx and \
                        (new_y, new_x) not in q1_visited and \
                        board[new_y][new_x] != -1:
                    q1_visited[(new_y, new_x)] = new_d
                    queue1.append((new_y, new_x, new_d))

        for y, x in q1_visited:
            if (y, x) in q2_visited:
                return q1_visited[y, x] + q2_visited[y, x]

        for _ in range(len(queue2)):
            y, x, d = queue2.popleft()
            for dy, dx in directions:
                new_y, new_x, new_d = y + dy, x + dx, d + 1
                if 0 <= new_y < ny and 0 <= new_x < nx and (new_y, new_x) not in q2_visited and board[new_y][
                    new_x] != -1:
                    q2_visited[(new_y, new_x)] = new_d
                    queue2.append((new_y, new_x, new_d))

        for y, x in q2_visited:
            if (y, x) in q1_visited:
                return q1_visited[y, x] + q2_visited[y, x]
    return None

