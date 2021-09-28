# longest common prefix string
def longest_common_prefix(strs):
    cur_common = strs[0]
    for string in strs[1:]:
        new_common = ''
        for c1, c2, in zip(string, cur_common):
            if c1 == c2:
                new_common += c1
            else:
                break
        cur_common = new_common
    return cur_common
