# rotate output
def spiral_matrix_o_to_i(matrix):
    result = []
    ny, nx = len(matrix), len(matrix[0])
    left, right, up, down = 0, nx - 1, 0, ny - 1
    while len(result) < ny * nx:
        for x in range(left, right + 1):
            result.append(matrix[up][x])

        for y in range(up + 1, down + 1):
            result.append(matrix[y][right])

        if up != down:
            for x in range(right - 1, left - 1, -1):
                result.append(matrix[down][x])

        if left != right:
            for y in range(down - 1, up, -1):
                result.append(matrix[y][left])

        up += 1
        left += 1
        right -= 1
        down -= 1
    return result


def spiral_matrix_i_to_o(matrix):
    result = []
    ny, nx = len(matrix), len(matrix[0])
    up, down, left, right = 0, ny - 1, 0, nx - 1
    while len(result) < nx * ny:
        for x in range(right, left - 1, -1):
            result.append(matrix[down][x])

        for y in range(down - 1, up - 1, -1):
            result.append(matrix[y][left])

        if up != down:
            for x in range(left + 1, right + 1):
                result.append(matrix[up][x])

        if left != right:
            for y in range(up + 1, down):
                result.append(matrix[y][right])

        left += 1
        right -= 1
        up += 1
        down -= 1
    return result[::-1]

