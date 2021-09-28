# Serialize and Deserialize Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class CodecDFS:
    def serialize1(self, root):
        # version 1
        def dfs(node, s):
            if not node:
                s += 'None,'
            else:
                s += str(node.val) + ','
                s = dfs(node.left, s)
                s = dfs(node.right, s)
            return s

        return dfs(root, '')

    def serialize2(self, root):
        # version 2
        self.result = ''

        def dfs(node):
            if not node:
                self.result += 'None,'
            else:
                self.result += str(node.val) + ','
                dfs(node.left)
                dfs(node.right)
            return s

        dfs(root)
        return self.result

    def deserialize(self, data):
        node_values = data.split(',')
        node_values.reverse()

        def dfs(nodes):
            if nodes[-1] == 'None':
                nodes.pop()
                return None
            node = TreeNode(int(nodes.pop().val))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node

        return dfs(node_values)


from collections import deque


class CodecBFS:
    def serialize(self, root):
        if not root:
            return 'None,'
        queue, result = deque([root]), ''
        while queue:
            cur = queue.popleft()
            if not cur:
                result += 'None,'
                continue
            result += str(cur.val) + ','
            queue.append(cur.left)
            queue.append(cur.right)
        return result

    def deserialize(self, data):
        node_values = data.split(',')
        if node_values[0] == 'None':
            return None
        root = TreeNode(int(node_values[0]))
        queue = deque([root])
        i = 1
        while queue and i < len(node_values):
            cur = queue.popleft()
            if node_values[i] != 'None':
                left = TreeNode(int(node_values[i]))
                cur.left = left
                queue.append(left)
            i += 1
            if node_values[i] != 'None':
                cur.right = TreeNode(int(node_values[i]))
                queue.append(cur.right)
            i += 1
        return root
