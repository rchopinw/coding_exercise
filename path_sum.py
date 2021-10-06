from collections import defaultdict, deque


# path sum
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def has_path_sum_dfs(root, target):
    flag = False

    def dfs(node, v):
        if node is None:
            return
        if node.left is None and node.right is None:
            if v + node.val == target:
                nonlocal flag
                flag = True
            return
        dfs(node.left, v + node.val)
        dfs(node.right, v + node.val)

    dfs(root, 0)
    return flag


def has_path_sum_bfs(root, targetSum):
    if not root:
        return False
    stack = deque([(root, 0)])
    while stack:
        cur_node, cur_sum = stack.popleft()
        cur_sum += cur_node.val
        if cur_node.left is None and cur_node.right is None and cur_sum == targetSum:
            return True
        if cur_node.left:
            stack.append((cur_node.left, cur_sum))
        if cur_node.right:
            stack.append((cur_node.right, cur_sum))
    return False


def path_sum_ii_dfs(root, target):
    """
    :param root:
    :param target:
    :return:
    """
    results = []

    def dfs(node, v, path):
        if node is None:
            return
        if node.right is None and node.right is None and node.val + v == target:
            results.append(path + [node.val])
            return
        dfs(node.left, v + node.val, path + [node.val])
        dfs(node.right, v + node.val, path + [node.val])

    dfs(root, 0, [])
    return results


def sum_to_k(nums, k):
    count, cur_sum = 0, 0
    record = defaultdict(int)
    for num in nums:
        cur_sum += num
        if cur_sum == k:
            count += 1
        count += record[cur_sum - k]
        record[cur_sum] += 1
    return count


def path_sum_iii(root, target):
    count, record = 0, defaultdict(int)

    def dfs(node, v):
        nonlocal count
        if not node:
            return
        v += node.val
        if v == target:
            count += 1
        count += record[v - target]
        record[v] += 1
        dfs(node.left, v)
        dfs(node.right, v)
        record[v] -= 1

    dfs(root, 0)
    return count


#  given a binary tree and K, count the number of paths starting from the root and summing to K.
def path_sum_to_k(root, k):
    if not root:
        return 0

    count = 0

    def dfs(node, s):
        nonlocal count
        if node is None:
            return
        s += node.val
        if s == k:
            count += 1
        dfs(node.left, s)
        dfs(node.right, s)

    dfs(root, 0)
    return count

















def path_sum_iiii(root, k):
    count = 0
    prefix = defaultdict(int)
    def dfs(node, cur_sum):
        nonlocal count
        if node is None:
            return
        cur_sum += node.val
        count += (cur_sum == k) + prefix[cur_sum - k]
        prefix[cur_sum] += 1
        dfs(node.left, cur_sum)
        dfs(node.right, cur_sum)
        prefix[cur_sum] -= 1
    dfs(root, 0)
    return count






























