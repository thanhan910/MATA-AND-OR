from .tree_types import Node, NodeType, reverse_node_type
import random


def gen_random_composition(n, lower_bound, upper_bound, numbers_count):
    """
    Splits the integer 'n' into a composition, i.e. a sum of 'numbers_count' random integers, where each integer is >= lower_bound and <= upper_bound.

    :param a: The integer to be split.
    :param lower_bound: The minimum value of each split integer.
    :param upper_bound: The maximum value of each split integer.
    :param min_numbers_count: The minimum number of integers to split 'a' into.
    :return: A list of integers that sum up to 'a'.
    """

    # Check if it's possible to split 'a' under given constraints
    if n < lower_bound * numbers_count or n > upper_bound * numbers_count:
        raise ValueError("Impossible to split the integer under given constraints.")
    
    if lower_bound > upper_bound:
        raise ValueError("Lower bound cannot be greater than upper bound.")
    
    if lower_bound == upper_bound:
        if n != lower_bound * numbers_count:
            raise ValueError("Impossible to split the integer under given constraints.")
        else:
            return [lower_bound] * numbers_count

    # Start with an empty list to store the split integers
    composition = []
    remaining_numbers_count = numbers_count
    remaining_sum = n
    while remaining_numbers_count > 0:
        remaining_numbers_count -= 1
        # Choose a random value for the current piece
        current_value_lower_bound = max(lower_bound, remaining_sum - upper_bound * remaining_numbers_count)
        current_value_upper_bound = min(upper_bound, remaining_sum - lower_bound * remaining_numbers_count)
        current_piece = random.randint(current_value_lower_bound, current_value_upper_bound)
        composition.append(current_piece)
        remaining_sum -= current_piece


    # shuffle the list
    random.shuffle(composition)

    return composition


def assign_depth(num_children_list : list[int], min_depth: int = 1, max_depth: int = None):
    """
    Assigns a depth to each children of each non-leaf node in the tree, based on the node's children count, and accounting for min depth and max depth constraints. 

    The target of this function is to make sure the depth of the entire tree is >= min_depth and <= max_depth.
    """
    if max_depth is None:
        max_depth = len(num_children_list)
    depth_list : list[list[int]] = [] # depth -> list of nodes with children of depth = depth + 1, and children count in that list
    slots_list : list[int] = [1]
    for num_children in num_children_list:
        # choose a random element and its index from the slots list, where the element value is > 0
        chosen_slot_index, chosen_slot_value = random.choice([(i, x) for i, x in enumerate(slots_list) if x > 0])
        # add num_children to the depth list
        if len(depth_list) <= chosen_slot_index:
            depth_list.append([])
        depth_list[chosen_slot_index].append(num_children)
        # update the slots list
        slots_list[chosen_slot_index] -= 1
        if chosen_slot_index + 1 >= len(slots_list):
            slots_list.append(0)
        slots_list[chosen_slot_index + 1] += num_children
    return depth_list


def gen_random_tree(num_leafs : int, min_depth : int = 1, num_non_leafs : int = None, min_breath : int = 2, max_breath : int = None):
    """
    Generates a random tree with the given number of leafs, minimum depth, non_leafs_count, and the given maximum and minimum breath per non-leaf node.

    Priority is given to the minimum depth, then to the number of non-leaf nodes, then to the minimum children count, then max children count.
    """
    depth_info : dict[int, int] = {}
    parent_info : dict[int, int] = {}
    children_info : dict[int, list[int]] = {}

    depth_info[0] = 0

    assert min_breath >= 2
    assert max_breath is None or max_breath >= min_breath
    assert min_depth >= 1

    if max_breath is None:
        max_breath = num_leafs

    if num_non_leafs is None:
        n = num_leafs - 1
        lower_bound = min_breath - 1
        upper_bound = max_breath - 1
        min_num_non_leafs = min_depth

        non_leafs_count_lower_bound = max(min_num_non_leafs, n // upper_bound + (1 if n % upper_bound else 0))
        non_leafs_count_upper_bound = max(min_num_non_leafs, n // lower_bound)

        num_non_leafs = random.randint(non_leafs_count_lower_bound, non_leafs_count_upper_bound)

    num_children_list = gen_random_composition(
        n=num_leafs - 1,
        lower_bound = min_breath - 1,
        upper_bound = max_breath - 1,
        numbers_count=num_non_leafs,
    )
    num_children_list = [num_children + 1 for num_children in num_children_list]

    tree_depth = 0 # initial depth of the tree
    frontier = [0] # initialize frontier, i.e. nodes that have no children yet
    leaf_nodes = [] # initialize leaf_nodes, used to keep track of the leaf nodes (when a node has depth >= max_depth, it is considered a leaf, and will be added to this list instead of the frontier)
    global_id_iterator = 0 # used to assign unique ids to nodes
    deepest_leafs = [0] # used to keep track of the deepest leafs in the tree
    for num_children in num_children_list:
        
        # Check if the tree is not deep enough, i.e., if the current depth is less than the minimum depth and there are still leafs to add
        tree_is_not_deep_enough = tree_depth < min_depth
        
        # If the tree is not deep enough, greedily add nodes at the bottom, else explore the nodes uniformly
        node_id = random.choice(deepest_leafs) if tree_is_not_deep_enough else random.choice(frontier)

        frontier.remove(node_id)
        
        depth = depth_info[node_id]

        children_info[node_id] = []

        new_child_depth = depth + 1

        adding_deepest_leaf = (new_child_depth >= tree_depth) and tree_is_not_deep_enough

        if new_child_depth > tree_depth:
            tree_depth = new_child_depth
            deepest_leafs = []

        # Add new 'num_children' children to the current node
        for _ in range(num_children):
            global_id_iterator += 1

            # if max_depth is not None and new_child_depth >= max_depth:
            #     leaf_nodes.append(global_id_iterator)
            # else:
            frontier.append(global_id_iterator)

            children_info[node_id].append(global_id_iterator)

            depth_info[global_id_iterator] = new_child_depth

            if adding_deepest_leaf:
                deepest_leafs.append(global_id_iterator)
            parent_info[global_id_iterator] = node_id

    leaf_nodes += frontier
    
    assert num_non_leafs == len(num_children_list)
    assert len(parent_info) == num_leafs + num_non_leafs - 1
    assert len(depth_info) == num_leafs + num_non_leafs
    assert len(leaf_nodes) == num_leafs

    avg_branching_factor = (num_leafs + num_non_leafs - 1) / num_non_leafs
    
    return depth_info, parent_info, children_info, leaf_nodes, num_non_leafs, tree_depth, avg_branching_factor


def gen_conventional_node_ids(leaf_nodes : list[int], children_info : dict[int, list[int]], root_node_id : int = 0):
    """
    Generate new node ids, based on this convention:
    - leaf nodes are numbered first, from 0 to len(leaf_nodes) - 1 (inclusive)
    - then, a dummy node is added. Its id is len(leaf_nodes).
    - then non-leaf nodes, from len(leaf_nodes) + 1 to len(leaf_nodes) + len(children_info) - 1 (inclusive)
    - then finally, the root node
    """
    middle_nodes = [node_id for node_id in children_info.keys() if node_id != root_node_id and len(children_info[node_id]) > 0]
    random.shuffle(middle_nodes)
    random.shuffle(leaf_nodes)
    # old_dummy_node_id = -1 # stub
    node_ids_sorted = leaf_nodes + [-1] + middle_nodes + [root_node_id]
    new_node_ids_map = { old_node_id : new_node_id for new_node_id, old_node_id in enumerate(node_ids_sorted) }
    new_root_node_id = len(node_ids_sorted) - 1
    new_leaf_nodes = list(range(len(leaf_nodes)))
    return new_node_ids_map, new_leaf_nodes, new_root_node_id


def reevaluate_tree(depth_info : dict[int, int], parent_info : dict[int, int], children_info : dict[int, int], new_node_ids_map : dict[int, int]):
    """
    Reevaluate tree information based on the new node_id mapping.
    """
    new_parent_info = {}
    new_children_info = {}
    new_depth_info = {}
    for old_node_id, new_node_id in new_node_ids_map.items():
        if old_node_id in parent_info:
            new_parent_info[new_node_id] = new_node_ids_map[parent_info[old_node_id]]
        if old_node_id in children_info:
            new_children_info[new_node_id] = [new_node_ids_map[child_id] for child_id in children_info[old_node_id]]
        if old_node_id in depth_info:
            new_depth_info[new_node_id] = depth_info[old_node_id]
    return new_depth_info, new_parent_info, new_children_info


def assign_node_type(depth_info: dict[int, int], leaf_nodes : list[int], children_info: dict[int, int], root_node_id : int, strict_and_or: bool = True, root_node_type : NodeType = None):
    """
    Randomly assigns a node type to each node in the tree, where the node type is either "AND", "OR", or "LEAF".
    """
    node_type_info = {}
    if root_node_type is None:
        root_node_type = random.choice([NodeType.AND, NodeType.OR])
    reversed_root_node_type = reverse_node_type(root_node_type)
    node_type_info[root_node_id] = root_node_type
    for node_id in leaf_nodes:
        node_type_info[node_id] = NodeType.LEAF
    for node_id, node_children_ids in children_info.items():
        if node_children_ids is None or len(node_children_ids) == 0:
            node_type_info[node_id] = NodeType.LEAF
            continue

        if node_id == root_node_id:
            continue
        
        if strict_and_or:
            node_depth = depth_info[node_id]
            node_type_info[node_id] = root_node_type if node_depth % 2 == 0 else reversed_root_node_type
        else:
            node_type_info[node_id] = random.choice([NodeType.AND, NodeType.OR])

    return node_type_info


def get_tree_info(tree_info_size : int, depth_info : dict[int, int], parent_info : dict[int, int], children_info : dict[int, list[int]], node_type_info : dict[int, NodeType]):
    return [Node(
        node_id=node_id,
        node_type=NodeType(node_type_info[node_id]),
        parent_id=parent_info[node_id] if node_id in parent_info else None,
        children_ids=children_info[node_id] if node_id in children_info else None,
        depth=depth_info[node_id] if node_id in depth_info else None,
    ) for node_id in range(tree_info_size)]


def gen_tree_info_full(num_leafs : int, min_depth : int = 1, num_non_leafs : int = None, min_num_children : int = 2, max_num_children : int = None, root_node_type : NodeType = None, strict_and_or : bool = True):
    """
    Generate a random tree, with minimum depth, number of non-leaf nodes, and the maximum and minimum number of children per non-leaf node.

    Return a list of Node objects, where each Node object contains information about the node's id, type, parent id, children ids, and depth.

    The node ids are assigned according to the following convention:
    - leaf nodes are numbered first, from 0 to num_leafs - 1 (inclusive)
    - then, a dummy node is added. Its id is num_leafs.
    - then non-leaf nodes, from num_leafs + 1 to num_leafs + num_non_leafs - 1 (inclusive)
    - then finally, the root node
    """
    depth_info, parent_info, children_info, leaf_nodes, num_non_leafs, tree_depth, avg_branching_factor = gen_random_tree(num_leafs=num_leafs, min_depth=min_depth, num_non_leafs=num_non_leafs, min_breath=min_num_children, max_breath=max_num_children)
    new_node_ids_map, leaf_nodes, root_node_id = gen_conventional_node_ids(leaf_nodes=leaf_nodes, children_info=children_info, root_node_id=0)
    depth_info, parent_info, children_info = reevaluate_tree(depth_info, parent_info, children_info, new_node_ids_map)
    node_type_info = assign_node_type(depth_info=depth_info, leaf_nodes=leaf_nodes, children_info=children_info, root_node_id=root_node_id,strict_and_or=strict_and_or, root_node_type=root_node_type)
    node_type_info[num_leafs] = NodeType.DUMMY
    tree_info_size = num_leafs + num_non_leafs + 1
    tree_info = get_tree_info(tree_info_size, depth_info, parent_info, children_info, node_type_info)
    return tree_info, depth_info, parent_info, children_info, leaf_nodes, root_node_id, num_non_leafs, tree_depth, avg_branching_factor 

