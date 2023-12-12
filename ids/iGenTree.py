import numpy as np
import itertools


def gen_random_partition(array, flatten: bool = True):
    """
    Generate a random partition of the given array.
    """
    if len(array) <= 2:
        return array
    partition = []
    for element in array:
        random_index = np.random.randint(0, len(partition) + 1)
        if random_index == len(partition):
            partition.append([element])
        else:
            partition[random_index].append(element)

    if len(partition) == 1:
        partition = partition[0]

    if flatten:
        for i, subset in enumerate(partition):
            if len(subset) == 1:
                partition[i] = partition[i][0]

    return partition


def gen_and_or_tree(task_num, max_depth=None):
    """
    Generate a random AND/OR tree with `task_num` tasks.
    Returns a list of lists or integers, where each element/subelement is a node.
    Each leaf node is an integer from 0 to `task_num-1`, and each non-leaf node is a list.
    """
    if max_depth is None:
        max_depth = np.random.randint(1, task_num // 2)
    # Fix max_depth to be at least 1 and at most task_num
    max_depth = max(1, min(max_depth, task_num))

    # Generate a random AND/OR tree with task_num tasks
    tree = list(range(task_num))
    for depth in range(max_depth - 1):
        tree = gen_random_partition(tree, flatten=True)

    # Randomly choose the root node type
    root_node_type = np.random.choice(['AND', 'OR'])
    
    return tree, root_node_type


def normal_form(tree, root_node_type, form):
    """
    Convert a tree to a normal form. (CNF or DNF)
    Worst case Complexity: At least O(e^(n/e)) where n is the number of leafs in the tree.
    """
    reverse_node_type = {'AND': 'OR', 'OR': 'AND'}

    while isinstance(tree, list) and len(tree) <= 1:
        tree = tree[0]
        root_node_type = reverse_node_type[root_node_type]
    if not isinstance(tree, list):
        return tree
    new_tree = [normal_form(subtree, root_node_type=reverse_node_type[root_node_type], form=form) for subtree in tree]
    trees_only = [subtree for subtree in new_tree if isinstance(subtree, list)]
    leafs_only = [subtree for subtree in new_tree if not isinstance(subtree, list)]
    target_node_type = 'AND' if form == 'CNF' else 'OR'
    if root_node_type == target_node_type:
        return leafs_only + [leaf for subtree in trees_only for leaf in subtree]
    else:
        new_normal_tree = []
        for combination in itertools.product(*trees_only):
            new_combination = []
            for item in combination:
                if isinstance(item, list):
                    new_combination.extend(item)
                else:
                    new_combination.append(item)
            new_combination.extend(leafs_only)
            new_normal_tree.append(new_combination)
        return new_normal_tree


def dfs(tree, root_node_type, num_tasks):
    """
    Depth-first search of the tree.
    
    Returns a generator that yields the node id, node type, parent node id, and depth of each node.
    
    To get the node id, node type, parent node id, and depth of the root node, use `next(generator)`.
    
    To iterate through the generator, use `for node_id, node_type, parent, depth in generator:`.

    For each `num_tasks`, the total number of non-leaf nodes is at most `num_tasks - 1`.
    
    For each node, `node_id` is iterated in ascending order starting from `num_tasks + 1` (unless it is a leaf, then node_id is the node). We reserve `num_tasks` for the id of the dummy task coalition.

    For each node, `node_type` is either 'AND', 'OR', or 'LEAF'.
    """
    global_node_id = num_tasks + 1
    frontier = [[tree, global_node_id, root_node_type, num_tasks, 0]]
    while len(frontier) > 0:
        node, node_id, node_type, parent, depth = frontier.pop()
        yield node_id, node_type, parent, depth
        if isinstance(node, list):
            for child in node:
                if isinstance(child, list):
                    global_node_id += 1
                    frontier.append([child, global_node_id, 'AND' if node_type == 'OR' else 'OR', node_id, depth + 1])
                else:
                    frontier.append([child, child, 'LEAF', node_id, depth + 1])


def bfs(tree, root_node_type, num_tasks):
    """
    Breadth-first search of the tree.
    
    Returns a generator that yields the node id, node type, parent node id, and depth of each node.
    
    To get the node id, node type, parent node id, and depth of the root node, use `next(generator)`.
    
    To iterate through the generator, use `for node_id, node_type, parent, depth in generator:`.

    For each `num_tasks`, the total number of non-leaf nodes is at most `num_tasks - 1`.
    
    For each node, `node_id` is iterated in ascending order starting from `num_tasks + 1` (unless it is a leaf, then node_id is the node). We reserve `num_tasks` for the id of the dummy task coalition.

    For each node, `node_type` is either 'AND', 'OR', or 'LEAF'.
    """
    global_node_id = num_tasks + 1
    frontier = [[tree, global_node_id, root_node_type, num_tasks, 0]]
    while len(frontier) > 0:
        node, node_id, node_type, parent, depth = frontier.pop(0)
        yield node_id, node_type, parent, depth
        if isinstance(node, list):
            for child in node:
                if isinstance(child, list):
                    global_node_id += 1
                    frontier.append([child, global_node_id, 'AND' if node_type == 'OR' else 'OR', node_id, depth + 1])
                else:
                    frontier.append([child, child, 'LEAF', node_id, depth + 1])