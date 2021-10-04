class UnionFind:
    def __init__(self, parent):
        self.parent = parent

    def union(self, x, y):
        x_parent = self.find(x)
        y_parent = self.find(y)
        if x_parent != y_parent:
            self.parent[y_parent] = x_parent

    def find(self, x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


def is_overlap(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    return (x1 - x2)**2 + (y1 - y2)**2 <= (r1 + r2)**2


def radar_car(radars):
    n_radars = len(radars)
    uf = UnionFind(parent=[i for i in range(n_radars)])
    for i in range(n_radars):
        for j in range(i, n_radars):
            if is_overlap(radars[i], radars[j]):
                uf.union(i, j)
    for i in range(n_radars):
        uf.parent[i] = uf.find(uf.parent[i])
    upper_bound, lower_bound = 1.0, 0.0
    prev_group = -1
    cur_upper, cur_lower = float('-inf'), float('inf')
    for i, cur_group in enumerate(uf.parent):
        if cur_group == prev_group:
            cur_upper = max(cur_upper, radars[i][1] + radars[i][2])
            cur_lower = min(cur_lower, radars[i][1] - radars[i][2])
        else:
            if cur_upper >= upper_bound and cur_lower <= lower_bound:
                return False
            cur_upper, cur_lower = float('-inf'), float('inf')
        prev_group = cur_group
    return True
