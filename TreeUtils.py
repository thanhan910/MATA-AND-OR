from dataclasses import dataclass
from enum import Enum
import itertools
import numpy as np


class NodeType(Enum):
    AND = "AND"
    OR = "OR"
    LEAF = "LEAF"
    DUMMY = "DUMMY"

def reverse_node_type(node_type):
    if node_type == NodeType.AND:
        return NodeType.OR
    elif node_type == NodeType.OR:
        return NodeType.AND
    else:
        return node_type
    

def is_not_leaf(node_type):
    return node_type == NodeType.AND or node_type == NodeType.OR


@dataclass
class Node:
    node_id: int
    node_type: NodeType = None
    parent_id: int = None
    depth: int = None
    children_ids: list[int] = None
    


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

    if flatten:
        for i, subset in enumerate(partition):
            if len(subset) == 1:
                partition[i] = partition[i][0]

    if len(partition) == 1:
        partition = partition[0]

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


def traverse_tree(tree, root_node_type, num_tasks, order='bfs'):
    """
    Traverse tree using depth-first search or breadth-first search.
    
    Returns a generator that yields the node id, node type, parent node id, and depth of each node.
    
    To get the node id, node type, parent node id, and depth of the root node, use `next(generator)`.
    
    To iterate through the generator, use `for node_id, node_type, parent, depth in generator:`.

    For each `num_tasks`, the total number of non-leaf nodes is at most `num_tasks - 1`.
    
    For each node, `node_id` is iterated in ascending order starting from `num_tasks + 1` (unless it is a leaf, then node_id is the node). We reserve `num_tasks` for the id of the dummy task coalition.

    For each node, `node_type` is either 'AND', 'OR', or 'LEAF'.
    """
    frontier_pop_index = 0 if order.lower() == 'bfs' else -1
    global_node_id = num_tasks + 1
    frontier = [[tree, global_node_id, root_node_type, None, 0]]
    while len(frontier) > 0:
        node, node_id, node_type, parent_id, depth = frontier.pop(frontier_pop_index)
        yield node_id, node_type, parent_id, depth
        if isinstance(node, list):
            for child in node:
                if isinstance(child, list):
                    global_node_id += 1
                    frontier.append([child, global_node_id, 'AND' if node_type == 'OR' else 'OR', node_id, depth + 1])
                else:
                    frontier.append([child, child, 'LEAF', node_id, depth + 1])


def calc_tree_info(tree, root_node_type, num_tasks, order='bfs'):
    tree_traversed = list(traverse_tree(tree, root_node_type, num_tasks, order=order))
    tree_info = [
        Node(
            node_id=i,
            node_type=NodeType.DUMMY,
            parent_id=None,
            depth=None,
            children_ids=[]
        ) for i in range(len(tree_traversed))
    ]
    for node_id, node_type, parent_id, depth in tree_traversed:
        node_id: int
        parent_id: int
        tree_info[node_id].node_id = node_id
        tree_info[node_id].node_type = NodeType(node_type)
        tree_info[node_id].parent_id = parent_id
        tree_info[node_id].depth = depth
        tree_info[parent_id].children_ids.append(node_id)


def gen_tree_advanced(task_num, max_depth=None, strict_AndOr_alternating = True):
    """
    Generate a random AND/OR tree with `task_num` tasks.
    Returns a list of lists or integers, where each element/subelement is a node.
    Each leaf node is an integer from 0 to `task_num-1`, and each non-leaf node is a list.
    """
    if max_depth is None:
        max_depth = np.random.randint(1, task_num // 2)
    # Fix max_depth to be at least 1 and at most task_num
    max_depth = max(1, min(max_depth, task_num))

    # We reserve task_num for the id of the dummy task coalition.
    tree_info = [
        Node(
            node_id=i,
            node_type=NodeType.LEAF,
            parent_id=None,
            depth=0,
            children_ids=[]
        ) for i in range(task_num)
    ] + [
        Node(
            node_id=task_num,
            node_type=NodeType.DUMMY,
            parent_id=None,
            depth=None,
            children_ids=[]
        )
    ]
    
    global_node_id = task_num
    node_ids_list = list(range(task_num))
    non_leaf_node_type = np.random.choice([NodeType.AND, NodeType.OR])
    depth = 0
    for _ in range(max_depth - 1):
        partition = gen_random_partition(node_ids_list, flatten=True)
        node_ids_list = []
        depth += 1
        non_leaf_node_type = reverse_node_type(non_leaf_node_type)
        for subset in partition:
            if isinstance(subset, list):
                global_node_id += 1
                tree_info.append(Node(
                    node_id=global_node_id,
                    node_type=np.random.choice([NodeType.AND, NodeType.OR]),
                    parent_id=None,
                    depth=depth,
                    children_ids=subset
                ))
                for leaf in subset:
                    tree_info[leaf].parent_id = global_node_id
                node_ids_list.append(global_node_id)
            else:
                tree_info[subset].parent_id = global_node_id
                tree_info[subset].depth = depth
                node_ids_list.append(subset)
    
    # Root node
    depth += 1
    global_node_id += 1
    tree_info.append(Node(
        node_id=global_node_id,
        node_type=np.random.choice([NodeType.AND, NodeType.OR]),
        parent_id=None,
        depth=depth,
        children_ids=node_ids_list
    ))
    for subset in node_ids_list:
        tree_info[subset].parent_id = global_node_id

    # Fix depth values, since we have been using the reversed depth
    for node in tree_info:
        if node.depth is not None:
            node.depth = depth - node.depth

    return tree_info


def traverse_tree_advanced(tree_info : list[Node], order='dfs', root_node_index=-1, leaf_only=False):
    """
    Traverse tree using depth-first search.
    
    Returns a generator that yields the node id, node type, parent node id, and depth of each node.
    
    To get the node id, node type, parent node id, and depth of the root node, use `next(generator)`.
    
    To iterate through the generator, use `for node_id, node_type, parent, depth in generator:`.

    For each `num_tasks`, the total number of non-leaf nodes is at most `num_tasks - 1`.
    
    For each node, `node_id` is iterated in ascending order starting from `num_tasks + 1` (unless it is a leaf, then node_id is the node). We reserve `num_tasks` for the id of the dummy task coalition.

    For each node, `node_type` is either 'AND', 'OR', or 'LEAF'.
    """
    frontier_pop_index = 0 if order.lower() == 'bfs' else -1
    frontier = [tree_info[root_node_index]]
    while len(frontier) > 0:
        node = frontier.pop(frontier_pop_index)
        if leaf_only and node.node_type == NodeType.LEAF:
            yield node
        if node.children_ids is not None:
            for child_id in node.children_ids:
                frontier.append(tree_info[child_id])


def convert_tree_to_list(tree_info : list[Node]):
    """
    Convert tree_info to tree_list, a list of lists or integers, where each element/subelement is a node.
    """
    def _helper(node : Node):
        if node.children_ids is None or node.children_ids == []:
            return node.node_id
        return [_helper(tree_info[i]) for i in node.children_ids]
    return _helper(tree_info[-1])


def normal_form_advanced(tree_info : list[Node], form : str = 'DNF'):
    """
    Convert a tree to a normal form. (CNF or DNF)
    Worst case Complexity: At least O(e^(n/e)) where n is the number of leafs in the tree.
    """

    def _normalized(node : Node):
        print(node.node_id)
        if not is_not_leaf(node.node_type):
            return node.node_id
        # Flatten the tree
        new_children_ids = []
        for child_id in node.children_ids:
            if tree_info[child_id].node_type == node.node_type:
                new_children_ids.extend(tree_info[child_id].children_ids)
            else:
                new_children_ids.append(child_id)

        new_tree = [_normalized(tree_info[i]) for i in new_children_ids]

        list_clauses = [clause for clause in new_tree if isinstance(clause, list)]
        literal_clauses = [clause for clause in new_tree if not isinstance(clause, list)]
        target_node_type = NodeType.AND if form.upper() == 'CNF' else NodeType.OR
        
        if node.node_type == target_node_type:
            return literal_clauses + [leaf for subtree in list_clauses for leaf in subtree]
        else:
            new_normal_tree = []
            for combination in itertools.product(*list_clauses):
                new_combination = []
                # Flatten the combination
                for item in combination:
                    if isinstance(item, list):
                        new_combination.extend(item)
                    else:
                        new_combination.append(item)
                # Add the literal clauses
                new_combination.extend(literal_clauses)
                # Add the new combination to the new tree
                new_normal_tree.append(new_combination)
            return new_normal_tree
        
    return _normalized(tree_info[-1])
