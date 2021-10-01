# excel column name
def excel_column_number(column):
    return sum((ord(column[i]) - 64) * 26 ** (len(column) - i - 1) for i in range(len(column)))


def excel_column_name(num):
    ans = ''
    while num > 0:
        num -= 1
        ans += chr(num % 26 + 65)
        num //= 26
    return ans[::-1]