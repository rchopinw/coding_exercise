# Quora round 2
class QueryForEventCount:
    def __init__(self, intervals):
        self.intervals = intervals
        self.size = len(intervals)

    def request(self, query):
        count = 0
        for interval in self.intervals:
            if query < interval[0] or query > interval[1]:
                count += 1
        return self.size - count


def parse_string(s):
    result, tmp_s, i = [], '', 0
    while i < len(s):
        if s[i] == ' ':
            result.append(tmp_s)
            tmp_s = ''
        elif s[i] == '"':
            k = i + 1
            while k < len(s) and s[k] != '"':
                k += 1
            result.append(s[i+1:k])
            i = k + 1
            continue
        else:
            tmp_s += s[i]
        i += 1

