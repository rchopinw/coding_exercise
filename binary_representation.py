# solve for the binary representation of given number n - 1
def binary_representation_loop(n):
    s = ''
    n = n - 1
    while n > 0:
        s += str(n%2)
        n //= 2
    return str(int(s))[::-1]


def binary_representation_recursion(n, flag=True):
    if flag:
        n = n - 1
    if n == 0:
        return '0'
    else:
        return binary_representation_recursion(n // 2, False) + str(n % 2)