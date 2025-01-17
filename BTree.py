import array

class BTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = array.array('i')  # Array of keys
        self.children = []  # List of child pointers

class BTree:
    def __init__(self, order):
        self.root = BTreeNode()
        self.order = order

    def insert(self, key):
        root = self.root
        if len(root.keys) == self.order - 1:  # If root is full
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        if node.is_leaf:
            node.keys.append(key)
            node.keys = array.array('i', sorted(node.keys))
        else:
            idx = len(node.keys) - 1
            while idx >= 0 and key < node.keys[idx]:
                idx -= 1
            idx += 1
            if len(node.children[idx].keys) == self.order - 1:
                self._split_child(node, idx)
                if key > node.keys[idx]:
                    idx += 1
            self._insert_non_full(node.children[idx], key)

    def _split_child(self, parent, idx):
        order = self.order
        node_to_split = parent.children[idx]
        new_node = BTreeNode(is_leaf=node_to_split.is_leaf)
        mid_idx = order // 2

        parent.keys.insert(idx, node_to_split.keys[mid_idx])
        parent.children.insert(idx + 1, new_node)

        new_node.keys = array.array('i', node_to_split.keys[mid_idx + 1:])
        node_to_split.keys = array.array('i', node_to_split.keys[:mid_idx])

        if not node_to_split.is_leaf:
            new_node.children = node_to_split.children[mid_idx + 1:]
            node_to_split.children = node_to_split.children[:mid_idx + 1]

    def search(self, key, node=None):
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return True

        if node.is_leaf:
            return False

        return self.search(key, node.children[i])

    def delete(self, key):
        self._delete(self.root, key)
        if not self.root.keys and not self.root.is_leaf:
            self.root = self.root.children[0]

    def _delete(self, node, key):
        if key in node.keys:
            idx = node.keys.index(key)
            if node.is_leaf:
                node.keys.pop(idx)
            else:
                predecessor = self._get_predecessor(node, idx)
                node.keys[idx] = predecessor
                self._delete(node.children[idx], predecessor)
        else:
            if node.is_leaf:
                return

            idx = 0
            while idx < len(node.keys) and key > node.keys[idx]:
                idx += 1

            if len(node.children[idx].keys) < (self.order - 1) // 2:
                self._fill(node, idx)

            if idx < len(node.keys) and key == node.keys[idx]:
                self._delete(node.children[idx + 1], key)
            else:
                self._delete(node.children[idx], key)

    def _fill(self, parent, idx):
        if idx > 0 and len(parent.children[idx - 1].keys) >= (self.order - 1) // 2:
            self._borrow_from_prev(parent, idx)
        elif idx < len(parent.children) - 1 and len(parent.children[idx + 1].keys) >= (self.order - 1) // 2:
            self._borrow_from_next(parent, idx)
        else:
            if idx == len(parent.children) - 1:
                self._merge(parent, idx - 1)
            else:
                self._merge(parent, idx)

    def _borrow_from_prev(self, parent, idx):
        child = parent.children[idx]
        sibling = parent.children[idx - 1]

        child.keys.insert(0, parent.keys[idx - 1])
        parent.keys[idx - 1] = sibling.keys.pop()

        if not child.is_leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, parent, idx):
        child = parent.children[idx]
        sibling = parent.children[idx + 1]

        child.keys.append(parent.keys[idx])
        parent.keys[idx] = sibling.keys.pop(0)

        if not child.is_leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, parent, idx):
        child = parent.children[idx]
        sibling = parent.children[idx + 1]

        child.keys.append(parent.keys.pop(idx))
        child.keys.extend(sibling.keys)

        if not child.is_leaf:
            child.children.extend(sibling.children)

        parent.children.pop(idx + 1)

    def _get_predecessor(self, node, idx):
        current = node.children[idx]
        while not current.is_leaf:
            current = current.children[-1]
        return current.keys[-1]



from collections import deque


def visualize_btree(tree):
    if not tree.root.keys:
        print("The BTree is empty.")
        return

    queue = deque([(tree.root, 0)])  # Queue of (node, level)
    current_level = 0
    result = []

    while queue:
        node, level = queue.popleft()

        # Start a new level
        if level > current_level:
            print("Level", current_level, ":", result)
            result = []
            current_level = level

        # Add keys of the current node to the result
        result.append(node.keys)

        # Add children to the queue
        if not node.is_leaf:
            for child in node.children:
                queue.append((child, level + 1))

    # Print the last level
    if result:
        print("Level", current_level, ":", result)
