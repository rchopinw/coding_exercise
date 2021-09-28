class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# construct binary tree with in-order and pre-order traversal
def build_tree(preorder, inorder):
    # the pre-order sequence always starts with root node while the in-order sequence is always
    # split by the root node, i.e., the root node is always located in the middle of the tree
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

