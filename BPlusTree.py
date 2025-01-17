class BPlusTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
        self.next = None

class BPlusTree:
    def __init__(self, order):
        self.root = BPlusTreeNode()
        self.order = order

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
        if node.is_leaf:
            if key in node.keys:
                node.keys.remove(key)
            return

        idx = 0
        while idx < len(node.keys) and key > node.keys[idx]:
            idx += 1

        if idx < len(node.keys) and key == node.keys[idx]:
            if node.children[idx].is_leaf:
                predecessor = node.children[idx].keys[-1]
                node.keys[idx] = predecessor
                self._delete(node.children[idx], predecessor)
            else:
                successor = node.children[idx + 1].keys[0]
                node.keys[idx] = successor
                self._delete(node.children[idx + 1], successor)
        else:
            if len(node.children[idx].keys) < (self.order - 1) // 2:
                self._fill(node, idx)

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

    def range_search(self, start, end):
        result = []
        current = self._find_leaf(start)

        while current:
            for key in current.keys:
                if start <= key <= end:
                    result.append(key)
            if current.keys[-1] > end:
                break
            current = current.next

        return result

    def _find_leaf(self, key, node=None):
        if node is None:
            node = self.root

        if node.is_leaf:
            return node

        idx = 0
        while idx < len(node.keys) and key > node.keys[idx]:
            idx += 1

        return self._find_leaf(key, node.children[idx])