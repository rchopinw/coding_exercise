from collections import deque


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class CodecBinaryTreeDFS:
    def __init__(self):
        pass

    def serialize(self, root):
        s = ''
        def dfs_encode(node):
            nonlocal s
            if not node:
                s += 'None,'
                return
            s += '{},'.format(node.val)
            dfs_encode(node.left)
            dfs_encode(node.right)

        dfs_encode(root)
        return s

    def deserialize(self, data):
        data = data.split(',')
        data.reverse()

        def dfs_decode(sequence):
            if sequence[-1] == 'None':
                sequence.pop()
                return None
            root = TreeNode(int(sequence.pop()))
            root.left = dfs_decode(sequence)
            root.right = dfs_decode(sequence)
            return root

        return dfs_decode(data)


class CodecBinaryTreeBFS:
    def __init__(self):
        pass

    def serialize(self, root):
        queue, s = deque([root]), ''
        while queue:
            cur = queue.popleft()
            if cur is None:
                s += 'None,'
                continue
            s += str(cur.val) + ','
            queue.append(cur.left)
            queue.append(cur.right)
        return s

    def deserialize(self, data):
        data = data.split(',')
        if data[0] == 'None':
            return None
        root = TreeNode(int(data[0]))
        queue = deque([root])
        i = 1
        while queue and i < len(data):
            cur = queue.popleft()
            if data[i] != 'None':
                cur.left = TreeNode(int(data[i]))
                queue.append(cur.left)
            i += 1
            if data[i] != 'None':
                cur.right = TreeNode(int(data[i]))
                queue.append(cur.right)
            i += 1
        return root


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

        dfs(root)
        return self.result

    def deserialize(self, data):
        node_values = data.split(',')
        node_values.reverse()

        def dfs(nodes):
            if nodes[-1] == 'None':
                nodes.pop()
                return None
            node = TreeNode(int(nodes.pop()))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node

        return dfs(node_values)
