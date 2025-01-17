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
def _(__file__):
    import cProfile
    import pstats
    from io import StringIO
    import os

    from pympler.asizeof import asizeof

    def profile(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')

            # Filter stats to include only functions from your script
            script_name = os.path.basename(__file__)  # Get the script name
            filtered_stats = [
                print(f"Cumulative time for {func.__name__}: {value[3]:.6f} seconds")  # Extract the function name
                for key, value in ps.stats.items() if func.__name__ == key[2]
            ]

            return result
        return wrapper
    return StringIO, asizeof, cProfile, os, profile, pstats


@app.cell
def _(mo):
    mo.md(r"""## B Tree Implementation""")
    return


@app.cell
def _(mo):
    mo.md(r"""### BTree Node""")
    return


@app.cell
def _():
    # Btree node
    class BTreeNode:
        def __init__(self, leaf=False):
            self.leaf = leaf
            self.keys = []
            self.children = []

        def __repr__(self):
            if self.leaf:
                return f"Leaf(keys={list(self.keys)})"
            else:
                children_repr = ", ".join(repr(child) if child else "None" for child in self.children)
                return f"Internal(keys={list(self.keys)}, children=[{children_repr}])"
    return (BTreeNode,)


@app.cell
def _(mo):
    mo.md(r"""### BTree""")
    return


@app.cell
def _(BTreeNode, math):
    class BTree:
        def __init__(self, t):
            self.root = BTreeNode(True)
            self.t = t

        def insert(self, k):
            root = self.root
            if len(root.keys) == self.t - 1:  # Maximum number of keys
                temp = BTreeNode(False)  # New root must not be a leaf
                temp.children.append(self.root)
                self.root = temp
                self.split_child(temp, 0)
                self.insert_non_full(temp, k)
            else:
                self.insert_non_full(root, k)

        def insert_non_full(self, x, k):
            i = len(x.keys) - 1
            if x.leaf:
                x.keys.append(None)  # Placeholder
                while i >= 0 and k < x.keys[i]:
                    x.keys[i + 1] = x.keys[i]
                    i -= 1
                x.keys[i + 1] = k
            else:
                while i >= 0 and k < x.keys[i]:
                    i -= 1
                i += 1
                if len(x.children[i].keys) == self.t - 1:
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

            # print(f"{i=}")
            # print(x.leaf, len(x.children))
            if x.leaf:
                if i < len(x.keys) and x.keys[i] == k:
                    x.keys.pop(i)
                return
            if i < len(x.keys) and x.keys[i] == k:
                self.delete_internal_node(x, k, i)
            elif len(x.children[i].keys) > t_min:
                self._delete(x.children[i], k)
            else:
                # Fix deficiency in the child
                self._fix_deficiency(x, i)

                # Recalculate `i` after fixing deficiency
                if i < len(x.keys) and k > x.keys[i]:
                    i += 1
                if i >= len(x.children):
                    i = len(x.children) - 1  # Clamp `i` to the last valid child index
                # print(f"{i=} {len(x.children)=} {x.keys=}")
                # Continue the deletion process
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
            """
            Search for a key in the B-tree.

            Parameters:
                key: The key to search for.
                x: The node to start searching from (defaults to the root).

            Returns:
                A tuple (node, index) if the key is found.
                None if the key is not found.
            """
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
            """
            Perform a range search in the B-tree to find all keys between low and high (inclusive).

            Parameters:
                low: The lower bound of the range.
                high: The upper bound of the range.
                x: The current node (defaults to the root if None).
                results: A list to collect the keys (used during recursion).

            Returns:
                A list of keys within the specified range.
            """
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
def _(mo):
    mo.md(r"""## B+ Tree Implementation""")
    return


@app.cell
def _(mo):
    mo.md("""### BPlusTree Node""")
    return


@app.cell
def _():
    class BPlusNode:
        def __init__(self, order):
            self.order = order
            self.values = []
            self.keys = []
            self.next_key = None
            self.parent = None
            self.is_leaf = False

        # Insert at the leaf
        def insert_at_leaf(self, leaf, value, key):
            if (self.values):
                temp1 = self.values
                for i in range(len(temp1)):
                    if (value == temp1[i]):
                        self.keys[i].append(key)
                        break
                    elif (value < temp1[i]):
                        self.values = self.values[:i] + [value] + self.values[i:]
                        self.keys = self.keys[:i] + [[key]] + self.keys[i:]
                        break
                    elif (i + 1 == len(temp1)):
                        self.values.append(value)
                        self.keys.append([key])
                        break
            else:
                self.values = [value]
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
        def insert_in_parent(self, n, value, ndash):
            if (self.root == n):
                rootNode = BPlusNode(n.order)
                rootNode.values = [value]
                rootNode.keys = [n, ndash]
                self.root = rootNode
                n.parent = rootNode
                ndash.parent = rootNode
                return

            parentNode = n.parent
            temp3 = parentNode.keys
            for i in range(len(temp3)):
                if (temp3[i] == n):
                    parentNode.values = parentNode.values[:i] + \
                        [value] + parentNode.values[i:]
                    parentNode.keys = parentNode.keys[:i +
                                                      1] + [ndash] + parentNode.keys[i + 1:]
                    if (len(parentNode.keys) > parentNode.order):
                        parentdash = BPlusNode(parentNode.order)
                        parentdash.parent = parentNode.parent
                        mid = int(math.ceil(parentNode.order / 2)) - 1
                        parentdash.values = parentNode.values[mid + 1:]
                        parentdash.keys = parentNode.keys[mid + 1:]
                        value_ = parentNode.values[mid]
                        if (mid == 0):
                            parentNode.values = parentNode.values[:mid + 1]
                        else:
                            parentNode.values = parentNode.values[:mid]
                        parentNode.keys = parentNode.keys[:mid + 1]
                        for j in parentNode.keys:
                            j.parent = parentNode
                        for j in parentdash.keys:
                            j.parent = parentdash
                        self.insert_in_parent(parentNode, value_, parentdash)

        # Delete a node
        def delete(self, key):
            # Find the node containing the key
            node_ = self.search(key)
        
            if not node_:
                print("Key not found in Tree")
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
        
            # If we reach here, the key was not found in the node
            print("Key not found in Tree")

        def move_kv(self, neighbor_node, i):
            moved_key = neighbor_node.keys.pop(i)
            moved_value = neighbor_node.values.pop(i)
            return moved_key, moved_value

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
                            moved_key, moved_value = self.move_kv(neighbor_node, i=-1)
                            # moved_key = neighbor_node.keys.pop(-1)
                            # moved_value = neighbor_node.values.pop(-1)
                            current_node.keys.insert(0, moved_key)
                            current_node.values.insert(0, separator_value)
                            for i, value in enumerate(parent_node.values):
                                if value == separator_value:
                                    parent_node.values[i] = moved_value
                                    break
                        else:
                            moved_key, moved_value = self.move_kv(neighbor_node, i=-1)
                            # moved_key = neighbor_node.keys.pop(-1)
                            # moved_value = neighbor_node.values.pop(-1)
                            current_node.keys.insert(0, moved_key)
                            current_node.values.insert(0, moved_value)
                            for i, value in enumerate(parent_node.values):
                                if value == separator_value:
                                    parent_node.values[i] = moved_value
                                    break
                    else:
                        if not current_node.is_leaf:
                            moved_key, moved_value = self.move_kv(neighbor_node, i=0)
                            # moved_key = neighbor_node.keys.pop(0)
                            # moved_value = neighbor_node.values.pop(0)
                            current_node.keys.append(moved_key)
                            current_node.values.append(separator_value)
                            for i, value in enumerate(parent_node.values):
                                if value == separator_value:
                                    parent_node.values[i] = moved_value
                                    break
                        else:
                            moved_key, moved_value = self.move_kv(neighbor_node, i=0)
                            # moved_key = neighbor_node.keys.pop(0)
                            # moved_value = neighbor_node.values.pop(0)
                            current_node.keys.append(moved_key)
                            current_node.values.append(moved_value)
                            for i, value in enumerate(parent_node.values):
                                if value == separator_value:
                                    parent_node.values[i] = neighbor_node.values[0]
                                    break
        
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
def _(BPlusTree, profile):
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

    @profile
    def create_tree(dataset, tree):
        for i in dataset:
            if isinstance(tree, BPlusTree):
                tree.insert(i, i)   
            else:
                tree.insert(i)

        return tree

    @profile
    def delete_tree(dataset, tree):
        for i in dataset:
            tree.delete(i)
        return tree

    @profile
    def search_tree(searchables, tree):
        for i in searchables:
            tree.search(i)

    @profile
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
    range_search_tree,
    search_tree,
    split_into_pairs,
):
    def all_tests(master_dataset, dataset_sizes, order, tree_type):
        for d_size in dataset_sizes:
            dataset = master_dataset[:d_size]

            print(f"\nsize: {d_size} and order: {order}")

            # Creation experiment
            print(f"\tInsertion")
            tree = tree_type(order)
            print("\t", end="")
            tree = create_tree(dataset, tree)
            print(f"\tsize of tree: {asizeof(tree)}")

            # Search experiment
            print(f"\tSearch")
            print("\t", end="")
            search_tree(dataset, tree)

            # Range Search experiment
            print(f"\tRange Search")
            pairs = split_into_pairs(dataset, d_size // 10)
            print("\t", end="")
            range_search_tree(pairs, tree)

            # Deletion experiment
            print(f"\tDeletion")
            print("\t", end="")
            tree = delete_tree(dataset, tree)
            print(f"\tsize of tree: {asizeof(tree)}")

            del tree
    return (all_tests,)


@app.cell
def _(mo):
    mo.md("""### Data Collection""")
    return


@app.cell
def _(generate_dataset, random):
    random.seed(0)

    dataset_sizes = [10**i for i in range(2, 6)]
    order = 4

    master_dataset = generate_dataset(dataset_sizes[-1])
    return dataset_sizes, master_dataset, order


@app.cell
def _():
    #all_tests(master_dataset, dataset_sizes, order, BTree)
    return


@app.cell
def _(BPlusTree, all_tests, dataset_sizes, master_dataset, order):
    all_tests(master_dataset, dataset_sizes, order, BPlusTree)
    return


@app.cell
def _(mo):
    mo.md("""## Visualization""")
    return


@app.cell
def _():
    import pandas as pd

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
    return data, df, pd


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
