class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# construct binary tree with in-order and pre-order traversal
def build_tree(preorder, inorder):
    # the pre-order sequence always starts with root node while the in-order sequence is always
    # split by the root node, i.e., the root node is always located in the middle of the two children
    p_idx = 0
    x_to_i = {xi: i for i, xi in enumerate(inorder)}  # build a mapping from value to index

    def recursion(left, right):
        nonlocal p_idx  # call the index pointer for pre-order array
        if left > right:  # reached a leaf node, no subtree anymore
            return None
        cur_val = preorder[p_idx]
        root = TreeNode(cur_val)
        p_idx += 1
        root.left = recursion(left, x_to_i[cur_val] - 1)  # fetch the corresponding index from in-order array with O(1)
        root.right = recursion(x_to_i[cur_val] + 1, right)
        return root
    return recursion(0, len(preorder) - 1)


def build_tree_ii(inorder, postorder):
    x_to_i = {x: i for i, x in enumerate(inorder)}

    def recursion(left, right):
        if left <= right:
            cur_val = postorder.pop()
            root = TreeNode(cur_val)
            mid = x_to_i[cur_val]
            root.right = recursion(mid + 1, right)
            root.left = recursion(left, mid - 1)
            return root
        return

    return recursion(0, len(inorder) - 1)


def in_order_traversal(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)
    dfs(root)
    return result


def pre_order_traversal(root):
    result = []
    def dfs(node):
        if not node:
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return result


if __name__ == '__main__':
    root1 = TreeNode(5)
    root1.left = TreeNode(1)
    root1.left.left = TreeNode(4)
    root1.left.right = TreeNode(7)
    root1.left.right.left = TreeNode(8)
    root1.right = TreeNode(2)
    root1_test = build_tree(pre_order_traversal(root1), in_order_traversal(root1))
    assert CodecBFS().serialize(root1_test) == CodecBFS().serialize(root1)

    root2 = TreeNode(3)
    root2.left = TreeNode(1)
    root2.right = TreeNode(2)
    root2_test = build_tree(pre_order_traversal(root2), in_order_traversal(root2))
    assert CodecBFS().serialize(root2_test) == CodecBFS().serialize(root2)

    root3 = None
    assert None == build_tree([], [])

    root4 = TreeNode(10)
    root4_test = build_tree([10], [10])
    assert  root4_test.val == 10 and not root4_test.left and not root4_test.right
