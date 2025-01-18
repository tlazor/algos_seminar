import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(mo):
    mo.md(r"""## CProfile Wrapper""")
    return


@app.cell
def _():
    import cProfile
    import pstats
    import pandas as pd
    from io import StringIO
    import os

    from pympler.asizeof import asizeof

    # Initialize the DataFrame
    profile_data = pd.DataFrame(columns=["tree_type", "order", "operation", "cumulative_time", "tree_size"])

    def profile(tree_type=None, order=None, operation=None, tree_size=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')

                # Extract cumulative time for the specific function
                cumulative_time = None
                for key, value in ps.stats.items():
                    if func.__name__ == key[2]:  # Check if the function name matches
                        cumulative_time = value[3]  # Extract cumulative time
                        break

                # Add a new row to the DataFrame
                if cumulative_time is not None:
                    global profile_data
                    profile_data = pd.concat([
                        profile_data,
                        pd.DataFrame([{
                            "tree_type": tree_type.__name__,
                            "order": order,
                            "operation": operation,
                            "cumulative_time": cumulative_time,
                            "tree_size": tree_size
                        }])
                    ], ignore_index=True)

                return result
            return wrapper
        return decorator
    return (
        StringIO,
        asizeof,
        cProfile,
        os,
        pd,
        profile,
        profile_data,
        pstats,
    )


@app.cell
def _():
    # import cProfile
    # import pstats
    # from io import StringIO
    # import os

    # from pympler.asizeof import asizeof

    # import pandas as pd

    # results_df = pd.Dataframe()

    # def profile(func):
    #     def wrapper(*args, **kwargs):
    #         profiler = cProfile.Profile()
    #         profiler.enable()
    #         result = func(*args, **kwargs)
    #         profiler.disable()
    #         s = StringIO()
    #         ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')

    #         # Filter stats to include only functions from your script
    #         script_name = os.path.basename(__file__)  # Get the script name
    #         filtered_stats = [
    #             print(f"Cumulative time for {func.__name__}: {value[3]:.6f} seconds")  # Extract the function name
    #             for key, value in ps.stats.items() if func.__name__ == key[2]
    #         ]

    #         return result
    #     return wrapper
    return


@app.cell
def _(mo):
    mo.md(r"""## B Tree Implementation""")
    return


@app.cell
def _(math):
    def validate_btree(node, t, depth=0, leaf_depths=None):
        if leaf_depths is None:
            leaf_depths = []

        # Check keys and children consistency
        if len(node.children) != len(node.keys) + 1 and not node.leaf:
            print(f"Invalid node at depth {depth}: keys={node.keys}, children={len(node.children)}")
            return False

        # Check for minimum and maximum keys in non-root nodes
        if depth > 0 and (len(node.keys) < math.ceil(t / 2) - 1 or len(node.keys) > t - 1):
            print(f"Invalid key count in node at depth {depth}: {node.keys}")
            return False

        # Check for leaf depths consistency
        if node.leaf:
            leaf_depths.append(depth)
        else:
            for child in node.children:
                if not validate_btree(child, t, depth + 1, leaf_depths):
                    return False

        # Ensure all leaves are at the same depth
        if depth == 0 and len(set(leaf_depths)) > 1:
            print(f"Leaves are at inconsistent depths: {leaf_depths}")
            return False

        return True

    def val_btree(tree):
        return validate_btree(tree.root, tree.t)

    return val_btree, validate_btree


@app.cell
def _(mo):
    mo.md(r"""### BTree Node""")
    return


@app.cell
def _():
    from array import array

    class BTreeNode:
        def __init__(self, leaf=False):
            self.leaf = leaf
            self.keys = array('i')
            self.children = []

        def __repr__(self):
            if self.leaf:
                return f"Leaf(keys={list(self.keys)})"
            else:
                children_repr = ", ".join(repr(child) if child else "None" for child in self.children)
                return f"Internal(keys={list(self.keys)}, children=[{children_repr}])"
    return BTreeNode, array


@app.cell
def _(mo):
    mo.md(r"""### BTree""")
    return


@app.cell
def _(BTreeNode, array, math):
    class BTree:
        def __init__(self, t):
            self.root = BTreeNode(True)
            self.t = t

        def insert(self, k):
            root = self.root
            if len(root.keys) == 2 * self.t - 1:  # Maximum number of keys
                temp = BTreeNode(leaf=False)  # New root is not a leaf
                temp.children.append(self.root)
                self.root = temp
                self.split_child(temp, 0)
                self.insert_non_full(temp, k)
            else:
                self.insert_non_full(root, k)

        def insert_non_full(self, x, k):
            i = len(x.keys) - 1
            if x.leaf:
                # Insert the key in the correct position
                x.keys = array('i', list(x.keys) + [0]) # Add a placeholder for the new key
                while i >= 0 and k < x.keys[i]:
                    x.keys[i + 1] = x.keys[i]
                    i -= 1
                x.keys[i + 1] = k
            else:
                # Find the child to recurse into
                while i >= 0 and k < x.keys[i]:
                    i -= 1
                i += 1
                # Split the child if it's full
                if len(x.children[i].keys) == 2 * self.t - 1:
                    self.split_child(x, i)
                    if k > x.keys[i]:
                        i += 1
                self.insert_non_full(x.children[i], k)

        def split_child(self, x, i):
            t = self.t
            y = x.children[i]
            z = BTreeNode(y.leaf)  # New node has the same leaf status
            mid = (t - 1) // 2

            # Adjust keys
            x.keys.insert(i, y.keys[mid])
            z.keys = y.keys[mid + 1:]
            y.keys = y.keys[:mid]

            # Adjust children if not leaf
            if not y.leaf:
                z.children = y.children[mid + 1:]
                y.children = y.children[:mid + 1]

            x.children.insert(i + 1, z)

        def delete(self, k):
            self._delete(self.root, k)
            if len(self.root.keys) == 0 and not self.root.leaf:
                self.root = self.root.children[0]

        def _delete(self, x, k):
            t_min = math.ceil(self.t / 2) - 1  # Minimum keys
            i = 0
            while i < len(x.keys) and k > x.keys[i]:
                i += 1
        
            if x.leaf:
                if i < len(x.keys) and x.keys[i] == k:
                    x.keys.pop(i)  # Delete key directly from leaf
                return
        
            # Key found in internal node
            if i < len(x.keys) and x.keys[i] == k:
                self.delete_internal_node(x, k, i)
            else:
                # Ensure child `i` exists before accessing
                if i >= len(x.children):
                    raise ValueError(f"Invalid child index {i}. Node: {x.keys}, Children: {len(x.children)}")
                
                # If the child is deficient, fix it
                if len(x.children[i].keys) <= t_min:
                    self._fix_deficiency(x, i)
        
                    # Recalculate `i` after fixing deficiency
                    if i < len(x.keys) and k > x.keys[i]:
                        i += 1
                    if i >= len(x.children):  # Revalidate index
                        i = len(x.children) - 1
        
                # Continue deletion in the adjusted child
                self._delete(x.children[i], k)

        def delete_internal_node(self, x, k, i):
            t_min = math.ceil(self.t / 2) - 1
            if len(x.children[i].keys) > t_min:
                predecessor = self.get_predecessor(x.children[i])
                x.keys[i] = predecessor
                self._delete(x.children[i], predecessor)
            elif len(x.children[i + 1].keys) > t_min:
                successor = self.get_successor(x.children[i + 1])
                x.keys[i] = successor
                self._delete(x.children[i + 1], successor)
            else:
                self.merge_nodes(x, i)
                self._delete(x.children[i], k)

        def get_predecessor(self, x):
            while not x.leaf:
                x = x.children[-1]
            return x.keys[-1]

        def get_successor(self, x):
            while not x.leaf:
                x = x.children[0]
            return x.keys[0]

        def _fix_deficiency(self, x, i):
            t_min = math.ceil(self.t / 2) - 1
            if i > 0 and len(x.children[i - 1].keys) > t_min:
                self.borrow_from_prev(x, i)
            elif i < len(x.children) - 1 and len(x.children[i + 1].keys) > t_min:
                self.borrow_from_next(x, i)
            else:
                if i < len(x.children) - 1:
                    self.merge_nodes(x, i)
                else:
                    self.merge_nodes(x, i - 1)

        def borrow_from_prev(self, x, i):
            child = x.children[i]
            sibling = x.children[i - 1]

            child.keys.insert(0, x.keys[i - 1])
            if not sibling.leaf:
                child.children.insert(0, sibling.children.pop())
            x.keys[i - 1] = sibling.keys.pop()

        def borrow_from_next(self, x, i):
            child = x.children[i]
            sibling = x.children[i + 1]

            child.keys.append(x.keys[i])
            if not sibling.leaf:
                child.children.append(sibling.children.pop(0))
            x.keys[i] = sibling.keys.pop(0)

        def merge_nodes(self, x, i):
            child = x.children[i]
            sibling = x.children[i + 1]

            child.keys.append(x.keys[i])
            child.keys.extend(sibling.keys)
            if not sibling.leaf:
                child.children.extend(sibling.children)

            x.keys.pop(i)
            x.children.pop(i + 1)

        def print_tree(self, x=None, level=0):
            if x is None:
                x = self.root
            print("Level", level, ":", len(x.keys), "keys:", x.keys)
            level += 1
            for child in x.children:
                self.print_tree(child, level)

        def search(self, key, x=None):
            if x is None:
                x = self.root  # Start from the root if no node is provided

            # Find the first key greater than or equal to the given key
            i = 0
            while i < len(x.keys) and key > x.keys[i]:
                i += 1

            # If the key is found in the current node
            if i < len(x.keys) and key == x.keys[i]:
                return x, i

            # If the node is a leaf, the key is not present
            if x.leaf:
                return None

            # Otherwise, search the appropriate child
            return self.search(key, x.children[i])

        def range_search(self, low, high, x=None, results=None):
            if x is None:
                x = self.root  # Start from the root if no node is provided
            if results is None:
                results = []  # Initialize the results list

            # Traverse the current node's keys and children
            i = 0
            while i < len(x.keys) and x.keys[i] < low:
                i += 1

            while i < len(x.keys) and x.keys[i] <= high:
                # If the node is not a leaf, explore the left child
                if not x.leaf:
                    self.range_search(low, high, x.children[i], results)
                # Add the key if it is within the range
                results.append(x.keys[i])
                i += 1

            # Explore the last child if it exists
            if not x.leaf and (i < len(x.children)):
                self.range_search(low, high, x.children[i], results)

            return results
    return (BTree,)


@app.cell
def _(BTree, master_dataset, val_btree):
    btree = BTree(3)

    temp = 6

    # Insert keys
    keys_to_insert = master_dataset[:temp]
    for key in keys_to_insert:
        btree.insert(key)

    print(val_btree(btree))

    btree.print_tree()

    # Test deletion cases
    keys_to_delete = master_dataset[:temp]
    for key in keys_to_delete:
        print(f"\n")
        btree.print_tree()
        print(f"\nDeleting key {key}:")
        btree.delete(key)
    btree.print_tree()
    return btree, key, keys_to_delete, keys_to_insert, temp


@app.cell
def _(mo):
    mo.md(r"""## B+ Tree Implementation""")
    return


@app.cell
def _(mo):
    mo.md("""### BPlusTree Node""")
    return


@app.cell
def _(array):
    class BPlusNode:
        def __init__(self, order):
            self.order = order
            self.values = array('i')
            self.keys = []
            self.next_key = None
            self.parent = None
            self.is_leaf = False

        # Insert at the leaf
        def insert_at_leaf(self, leaf, value, key):
            if self.values:
                for i in range(len(self.values)):
                    if value == self.values[i]:
                        self.keys[i].append(key)  # Append key to the existing array at index i
                        break
                    elif value < self.values[i]:
                        # Insert value and key at the correct position
                        self.values = array('i', self.values[:i]) + array('i', [value]) + array('i', self.values[i:])
                        self.keys = self.keys[:i] + [[key]] + self.keys[i:]
                        break
                    elif i + 1 == len(self.values):
                        self.values.append(value)
                        self.keys.append([key])
                        break
            else:
                # Initialize with the first value and key
                self.values = array('i', [value])
                self.keys = [[key]]
    return (BPlusNode,)


@app.cell
def _(mo):
    mo.md(r"""### BPlusTree""")
    return


@app.cell
def _(BPlusNode, math):
    class BPlusTree:
        def __init__(self, order):
            self.root = BPlusNode(order)
            self.root.is_leaf = True

        # Insert operation
        def insert(self, value, key):
            # value = str(value)
            old_node = self.search(value)
            old_node.insert_at_leaf(old_node, value, key)

            if (len(old_node.values) == old_node.order):
                node1 = BPlusNode(old_node.order)
                node1.is_leaf = True
                node1.parent = old_node.parent
                mid = int(math.ceil(old_node.order / 2)) - 1
                node1.values = old_node.values[mid + 1:]
                node1.keys = old_node.keys[mid + 1:]
                node1.next_key = old_node.next_key
                old_node.values = old_node.values[:mid + 1]
                old_node.keys = old_node.keys[:mid + 1]
                old_node.next_key = node1
                self.insert_in_parent(old_node, node1.values[0], node1)

        # Search operation for different operations
        def search(self, value):
            current_node = self.root
            while(current_node.is_leaf == False):
                temp2 = current_node.values
                for i in range(len(temp2)):
                    if (value == temp2[i]):
                        current_node = current_node.keys[i + 1]
                        break
                    elif (value < temp2[i]):
                        current_node = current_node.keys[i]
                        break
                    elif (i + 1 == len(current_node.values)):
                        current_node = current_node.keys[i + 1]
                        break
            return current_node

        def range_search(self, start_value, end_value):
            results = []
            # Find the starting point for the range
            current_node = self.search(start_value)

            while current_node:
                for value in current_node.values:
                    if start_value <= value <= end_value:
                        results.append(value)
                    elif value > end_value:
                        # We've gone past the range, so stop searching
                        return results

                # Move to the next leaf node using the linked list
                current_node = current_node.next_key

            return results

        # Find the node
        def find(self, value, key):
            l = self.search(value)
            for i, item in enumerate(l.values):
                if item == value:
                    if key in l.keys[i]:
                        return True
                    else:
                        return False
            return False

        # Inserting at the parent
        def insert_in_parent(self, node, new_value, new_child):
            # If the node is the root, create a new root
            if self.root == node:
                new_root = BPlusNode(node.order)
                new_root.values = [new_value]
                new_root.keys = [node, new_child]
                self.root = new_root
                node.parent = new_root
                new_child.parent = new_root
                return

            # Insert the new value and child into the parent node
            parent_node = node.parent
            parent_keys = parent_node.keys
            for i in range(len(parent_keys)):
                if parent_keys[i] == node:
                    # Insert the new value and new child at the appropriate position
                    parent_node.values = parent_node.values[:i] + [new_value] + parent_node.values[i:]
                    parent_node.keys = parent_node.keys[:i + 1] + [new_child] + parent_node.keys[i + 1:]

                    # If the parent node is overfull, split it
                    if len(parent_node.keys) > parent_node.order:
                        new_parent = BPlusNode(parent_node.order)
                        new_parent.parent = parent_node.parent
                        mid_index = int(math.ceil(parent_node.order / 2)) - 1

                        # Distribute values and keys
                        new_parent.values = parent_node.values[mid_index + 1:]
                        new_parent.keys = parent_node.keys[mid_index + 1:]
                        middle_value = parent_node.values[mid_index]

                        # Trim the original parent node
                        if mid_index == 0:
                            parent_node.values = parent_node.values[:mid_index + 1]
                        else:
                            parent_node.values = parent_node.values[:mid_index]
                        parent_node.keys = parent_node.keys[:mid_index + 1]

                        # Update parent references
                        for child in parent_node.keys:
                            child.parent = parent_node
                        for child in new_parent.keys:
                            child.parent = new_parent

                        # Recursively insert the middle value into the parent
                        self.insert_in_parent(parent_node, middle_value, new_parent)
                    return


        # Delete a node
        def delete(self, key):
            # Find the node containing the key
            node_ = self.search(key)

            if not node_:
                print("Key not found")
                return

            # Locate the key within the node
            for i, key_list in enumerate(node_.keys):
                if key in key_list:
                    # Remove the key from the key list
                    key_list.remove(key)

                    # If the key list becomes empty, handle node adjustment
                    if not key_list:
                        del node_.keys[i]
                        node_.values.pop(i)

                        if node_ == self.root:
                            # Special case: if the root becomes empty
                            if not node_.keys:
                                self.root = None
                        else:
                            # Propagate changes to maintain B+ tree properties
                            self.delete_entry(node_, key, key)
                    return
            print("Key not found")

        @staticmethod
        def move_kv(neighbor_node, i):
            moved_key = neighbor_node.keys.pop(i)
            moved_value = neighbor_node.values.pop(i)
            return moved_key, moved_value

        @staticmethod
        def temp(parent_node, separator_value, val):
            for i, value in enumerate(parent_node.values):
                if value == separator_value:
                    parent_node.values[i] = val
                    break

        # Delete an entry
        def delete_entry(self, current_node, target_value, target_key):

            # Remove the key and value if the current node is not a leaf
            if not current_node.is_leaf:
                for i, key in enumerate(current_node.keys):
                    if key == target_key:
                        current_node.keys.pop(i)
                        break
                for i, value in enumerate(current_node.values):
                    if value == target_value:
                        current_node.values.pop(i)
                        break

            # Adjust the root if it becomes a single key
            if self.root == current_node and len(current_node.keys) == 1:
                self.root = current_node.keys[0]
                current_node.keys[0].parent = None
                del current_node
                return

            # Check if the node is underfull
            is_underfull = (
                (len(current_node.keys) < int(math.ceil(current_node.order / 2)) and not current_node.is_leaf)
                or (len(current_node.values) < int(math.ceil((current_node.order - 1) / 2)) and current_node.is_leaf)
            )

            if is_underfull:
                is_predecessor = False
                parent_node = current_node.parent
                previous_node = None
                next_node = None
                previous_key = None
                next_key = None

                for i, child in enumerate(parent_node.keys):
                    if child == current_node:
                        if i > 0:
                            previous_node = parent_node.keys[i - 1]
                            previous_key = parent_node.values[i - 1]
                        if i < len(parent_node.keys) - 1:
                            next_node = parent_node.keys[i + 1]
                            next_key = parent_node.values[i]

                # Determine the neighbor node and value
                if not previous_node:
                    neighbor_node = next_node
                    separator_value = next_key
                elif not next_node:
                    is_predecessor = True
                    neighbor_node = previous_node
                    separator_value = previous_key
                else:
                    if len(current_node.values) + len(next_node.values) < current_node.order:
                        neighbor_node = next_node
                        separator_value = next_key
                    else:
                        is_predecessor = True
                        neighbor_node = previous_node
                        separator_value = previous_key

                # Merge or redistribute
                if len(current_node.values) + len(neighbor_node.values) < current_node.order:
                    if not is_predecessor:
                        current_node, neighbor_node = neighbor_node, current_node

                    neighbor_node.keys += current_node.keys
                    if not current_node.is_leaf:
                        neighbor_node.values.append(separator_value)
                    else:
                        neighbor_node.next_key = current_node.next_key
                    neighbor_node.values += current_node.values

                    if not neighbor_node.is_leaf:
                        for child in neighbor_node.keys:
                            child.parent = neighbor_node

                    self.delete_entry(current_node.parent, separator_value, current_node)
                    del current_node
                else:
                    if is_predecessor:
                        if not current_node.is_leaf:
                            moved_key, moved_value = BPlusTree.move_kv(neighbor_node, i=-1)
                            current_node.keys.insert(0, moved_key)
                            current_node.values.insert(0, separator_value)
                            BPlusTree.temp(parent_node, separator_value, moved_value)
                        else:
                            moved_key, moved_value = BPlusTree.move_kv(neighbor_node, i=-1)
                            current_node.keys.insert(0, moved_key)
                            current_node.values.insert(0, moved_value)
                            BPlusTree.temp(parent_node, separator_value, moved_value)
                    else:
                        if not current_node.is_leaf:
                            moved_key, moved_value = BPlusTree.move_kv(neighbor_node, i=0)
                            current_node.keys.append(moved_key)
                            current_node.values.append(separator_value)
                            BPlusTree.temp(parent_node, separator_value, moved_value)
                        else:
                            moved_key, moved_value = BPlusTree.move_kv(neighbor_node, i=0)
                            current_node.keys.append(moved_key)
                            current_node.values.append(moved_value)
                            BPlusTree.temp(parent_node, separator_value, neighbor_node.values[0])

                    if not neighbor_node.is_leaf:
                        for child in neighbor_node.keys:
                            child.parent = neighbor_node
                    if not current_node.is_leaf:
                        for child in current_node.keys:
                            child.parent = current_node
                    if not parent_node.is_leaf:
                        for child in parent_node.keys:
                            child.parent = parent_node
    return (BPlusTree,)


@app.cell
def _(mo):
    mo.md(r"""## Tests""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Test Methods""")
    return


@app.cell
def _(BPlusTree):
    import random
    random.seed(0)

    def generate_dataset(num_elements):
        """
        Generate a dataset of keys for testing with random distribution.

        Parameters:
            num_elements (int): Number of elements in the dataset.

        Returns:
            list: Generated dataset as a list of keys.
        """
        # Generate random integers within a specified range
        dataset = random.sample(range(num_elements * 10), num_elements)
        return dataset

    def create_tree(dataset, tree):
        for i in dataset:
            if isinstance(tree, BPlusTree):
                tree.insert(i, i)   
            else:
                tree.insert(i)

    def delete_tree(dataset, tree):
        for i in dataset:
            tree.delete(i)

    def search_tree(searchables, tree):
        for i in searchables:
            tree.search(i)

    def range_search_tree(pairs, tree):
        for low, high in pairs:
            tree.range_search(low, high)
    return (
        create_tree,
        delete_tree,
        generate_dataset,
        random,
        range_search_tree,
        search_tree,
    )


@app.cell
def _():
    def split_into_pairs(lst, n):
        """
        Split every n elements from a list into a list of pairs.

        Parameters:
            lst (list): The input list.
            n (int): The step size to get elements.

        Returns:
            list: A list of pairs.
        """
        # Get every n-th element
        nth_elements = lst[::n]

        # Pair consecutive elements
        pairs = list(zip(nth_elements, nth_elements[1:]))

        pairs = [sorted(pair) for pair in pairs]

        return pairs
    return (split_into_pairs,)


@app.cell
def _(
    asizeof,
    create_tree,
    delete_tree,
    profile,
    range_search_tree,
    search_tree,
    split_into_pairs,
):
    def all_tests(master_dataset, dataset_sizes, orders, tree_type):
        for order in orders:
            for d_size in dataset_sizes:
                profiled_create = profile(tree_type=tree_type, order=order, operation="Insertion", tree_size=d_size)(create_tree)
                profiled_delete = profile(tree_type=tree_type, order=order, operation="Deletion", tree_size=d_size)(delete_tree)
                profiled_search = profile(tree_type=tree_type, order=order, operation="Search", tree_size=d_size)(search_tree)
                profiled_range_search = profile(tree_type=tree_type, order=order, operation="Range Search", tree_size=d_size)(range_search_tree)

                dataset = master_dataset[:d_size]

                # Creation experiment
                tree = tree_type(order)
                profiled_create(dataset, tree)
                print(f"\tsize of tree: {asizeof(tree)}")

                # Search experiment
                profiled_search(dataset, tree)

                # Range Search experiment
                pairs = split_into_pairs(dataset, d_size // 10)
                profiled_range_search(pairs, tree)

                # Deletion experiment
                profiled_delete(dataset, tree)

                del tree
    return (all_tests,)


@app.cell
def _(mo):
    mo.md("""### Data Collection""")
    return


@app.cell
def _(generate_dataset, profile_data, random):
    random.seed(0)

    profile_data.drop(profile_data.index, inplace=True)

    dataset_sizes = [10**i for i in range(2, 6)]
    orders = [3,4,5]

    master_dataset = generate_dataset(dataset_sizes[-1])
    return dataset_sizes, master_dataset, orders


@app.cell
def _(BTree, all_tests, dataset_sizes, master_dataset, orders):
    all_tests(master_dataset, dataset_sizes, orders, BTree)
    return


@app.cell
def _(BPlusTree, all_tests, dataset_sizes, master_dataset, orders):
    all_tests(master_dataset, dataset_sizes, orders, BPlusTree)
    return


@app.cell
def _(profile_data):
    profile_data
    return


@app.cell
def _(mo):
    mo.md("""## Visualization""")
    return


@app.cell
def _(pd):
    # Data extracted from the input
    data = [
        {"size": 100, "order": 4, "operation": "Insertion", "cumulative_time": 0.000413, "tree_size": 20656},
        {"size": 100, "order": 4, "operation": "Search", "cumulative_time": 0.000287, "tree_size": None},
        {"size": 100, "order": 4, "operation": "Range Search", "cumulative_time": 0.000331, "tree_size": None},
        {"size": 100, "order": 4, "operation": "Deletion", "cumulative_time": 0.000752, "tree_size": 768},
        {"size": 1000, "order": 4, "operation": "Insertion", "cumulative_time": 0.006043, "tree_size": 217744},
        {"size": 1000, "order": 4, "operation": "Search", "cumulative_time": 0.004638, "tree_size": None},
        {"size": 1000, "order": 4, "operation": "Range Search", "cumulative_time": 0.002864, "tree_size": None},
        {"size": 1000, "order": 4, "operation": "Deletion", "cumulative_time": 0.011363, "tree_size": 768},
        {"size": 10000, "order": 4, "operation": "Insertion", "cumulative_time": 0.083737, "tree_size": 2147752},
        {"size": 10000, "order": 4, "operation": "Search", "cumulative_time": 0.071617, "tree_size": None},
        {"size": 10000, "order": 4, "operation": "Range Search", "cumulative_time": 0.036210, "tree_size": None},
        {"size": 10000, "order": 4, "operation": "Deletion", "cumulative_time": 0.146482, "tree_size": 768},
        {"size": 100000, "order": 4, "operation": "Insertion", "cumulative_time": 1.672859, "tree_size": 21372968},
        {"size": 100000, "order": 4, "operation": "Search", "cumulative_time": 0.986769, "tree_size": None},
        {"size": 100000, "order": 4, "operation": "Range Search", "cumulative_time": 0.274352, "tree_size": None},
        {"size": 100000, "order": 4, "operation": "Deletion", "cumulative_time": 1.862700, "tree_size": 768},
    ]

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    return data, df


@app.cell
def _(df):
    import matplotlib.pyplot as plt

    # Filter data for each operation
    operations = df['operation'].unique()

    # Plot cumulative time vs size for each operation
    for operation in operations:
        operation_data = df[df['operation'] == operation]
        plt.figure(figsize=(8, 6))
        if operation == "Range Search":
            print(operation_data['cumulative_time'], operation_data['size'] // 10)
            plt.plot(operation_data['size'], operation_data['cumulative_time'] / (operation_data['size'] // 10) * 1000, marker='o')
        else:
            plt.plot(operation_data['size'], operation_data['cumulative_time'] / operation_data['size'] * 1000, marker='o')
        plt.title(f"Effect of Size on Operation Time: {operation}")
        plt.xlabel("Size")
        plt.ylabel("Time per Operation (milliseconds)")
        # plt.xscale("log")  # Optional: Log scale for better visualization of size effects
        plt.grid(True)
        plt.show()

    # Plot tree size vs size for each operation where tree size is available
    for operation in operations:
        operation_data = df[df['operation'] == "Insertion"]
        if operation_data['tree_size'].notna().any():  # Only plot if tree size data exists
            plt.figure(figsize=(8, 6))
            plt.plot(operation_data['size'], operation_data['tree_size'], marker='o')
            plt.title(f"Effect of Dataset Size on Tree Size: {operation}")
            plt.xlabel("Dataset Size")
            plt.ylabel("Tree Size (bytes)")
            # plt.xscale("log")  # Optional: Log scale for better visualization of size effects
            plt.grid(True)
            plt.show()
    return operation, operation_data, operations, plt


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
