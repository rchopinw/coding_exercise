# solve for the binary representation of given number n - 1
def binary_representation_loop(n):
    s = ''
    n = n - 1
    if n == 0:
        return '0'
    while n > 0:
        s += str(n%2)
        n //= 2
    return s[::-1]


def binary_representation_recursion(n, flag=True):
    if flag:
        n = n - 1
    if n == 0:
        return '0'
    else:
        return binary_representation_recursion(n // 2, False) + str(n % 2)


if __name__ == "__main__":
    assert binary_representation_loop(9) == '1000', 'Wrong with test case {}'.format(9)
    assert binary_representation_loop(10) == '1001', 'Wrong with test case {}'.format(10)
    assert binary_representation_loop(1) == '0', 'Wrong with test case {}'.format(1)
    assert binary_representation_loop(2) == '1', 'Wrong with test case {}'.format(2)

    assert binary_representation_recursion(1)