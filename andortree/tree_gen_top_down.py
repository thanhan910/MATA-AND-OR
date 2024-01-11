
import random


def gen_num_non_leafs(min_depth: int, min_degree: int, max_degree: int, num_leafs: int, min_num_non_leafs: int = 1, max_num_non_leafs: int = None):
    n = num_leafs - 1
    lower_bound = min_degree - 1
    upper_bound = max_degree - 1

    non_leafs_count_lower_bound = max(min_depth, min_num_non_leafs, n // upper_bound + (1 if n % upper_bound else 0))
    non_leafs_count_upper_bound = max(min_depth, n // lower_bound)
    if max_num_non_leafs is not None:
        non_leafs_count_upper_bound = min(non_leafs_count_upper_bound, max_num_non_leafs)

    return random.randint(non_leafs_count_lower_bound, non_leafs_count_upper_bound)


def gen_random_composition(n, lower_bound, upper_bound, size):
    """
    Splits the integer 'n' into a composition, i.e. a sum of 'size' random integers, where each integer is >= lower_bound and <= upper_bound.

    :param a: The integer to be split.
    :param lower_bound: The minimum value of each split integer.
    :param upper_bound: The maximum value of each split integer.
    :param min_numbers_count: The minimum number of integers to split 'n' into.
    :return: A list of integers that sum up to 'n'.
    """

    # Check if it's possible to split 'a' under given constraints
    if n < lower_bound * size or n > upper_bound * size:
        raise ValueError("Impossible to split the integer under given constraints.")
    
    if lower_bound > upper_bound:
        raise ValueError("Lower bound cannot be greater than upper bound.")
    
    if lower_bound == upper_bound:
        if n != lower_bound * size:
            raise ValueError("Impossible to split the integer under given constraints.")
        else:
            return [lower_bound] * size

    # Start with an empty list to store the split integers
    composition = []
    remaining_numbers_count = size
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


def swap_degree_values_outer(depth_degree_allocation : list[list[int]], leafs_counts : list[int], current_it : int, degree_values_list : list[int]):

    tree_depth = len(depth_degree_allocation)
    
    arg_max_it = max(range(current_it, len(degree_values_list)), key=lambda i: degree_values_list[i])
    
    depths_options = [d for d, v in enumerate(depth_degree_allocation) if degree_values_list[arg_max_it] > v[-1] and d < tree_depth-1]
    
    if len(depths_options) == 0:
        return None
    
    selected_depth = random.choice(depths_options)
    
    added_leafs_count = degree_values_list[arg_max_it] - depth_degree_allocation[selected_depth][-1]
    
    degree_values_list[arg_max_it], depth_degree_allocation[selected_depth][-1] = depth_degree_allocation[selected_depth][-1], degree_values_list[arg_max_it]
    
    depth_degree_allocation[selected_depth].sort(reverse=True)
    
    chosen_leaf_depth = selected_depth + 1
    
    leafs_counts[chosen_leaf_depth] += added_leafs_count
    
    return chosen_leaf_depth


def swap_degree_values_inner(depth_degree_allocation : list[list[int]], leafs_counts : list[int]):

    tree_depth = len(depth_degree_allocation)
    
    chosen_i, chosen_j = None, None
    
    for i in range(0, tree_depth - 1):
        for j in range(tree_depth - 1, i, -1):
            if depth_degree_allocation[i][-1] < depth_degree_allocation[j][0]:
                chosen_i, chosen_j = i, j
                break
        if chosen_j is not None:
            break
    
    if chosen_j is None:
        return None
    
    added_leafs_count = depth_degree_allocation[chosen_j][0] - depth_degree_allocation[chosen_i][-1]
    
    depth_degree_allocation[chosen_i][-1], depth_degree_allocation[chosen_j][0] = depth_degree_allocation[chosen_j][0], depth_degree_allocation[chosen_i][-1]
    
    depth_degree_allocation[chosen_i].sort(reverse=True)
    
    depth_degree_allocation[chosen_j].sort(reverse=True)
    
    # incrementally update the depth list from chosen_j onwards
    for k in range(1, tree_depth - chosen_j):
        depth_degree_allocation[chosen_i + k] += depth_degree_allocation[chosen_j + k][:added_leafs_count]
        new_added_leafs_count = sum(depth_degree_allocation[chosen_j + k][:added_leafs_count])
        depth_degree_allocation[chosen_j + k] = depth_degree_allocation[chosen_j + k][added_leafs_count:]
        depth_degree_allocation[chosen_i + k].sort(reverse=True)
        added_leafs_count = new_added_leafs_count

    chosen_leaf_depth = chosen_i + tree_depth - chosen_j

    # update the slots list
    leafs_counts[chosen_leaf_depth] += added_leafs_count

    return chosen_leaf_depth


def add_value_cut_outer(depth_degree_allocation : list[list[int]], leafs_counts : list[int], current_it : int, degree_values_list : list[int], min_degree: int, max_degree: int):

    tree_depth = len(depth_degree_allocation)

    arg_max_it = max(range(current_it, len(degree_values_list)), key=lambda i: degree_values_list[i])
    
    if degree_values_list[arg_max_it] - 1 < min_degree:
        return None
    
    depths_options = [d for d, v in enumerate(depth_degree_allocation) if (max_degree is None or v[-1] + 1 <= max_degree) and d < tree_depth - 1]
    
    if len(depths_options) == 0:
        return None
    
    selected_depth = random.choice(depths_options)
    
    added_leafs_count = random.randint(1, min(max_degree - depth_degree_allocation[selected_depth][-1], degree_values_list[arg_max_it] - min_degree))
    
    degree_values_list[arg_max_it] -= added_leafs_count
    
    depth_degree_allocation[selected_depth][-1] += added_leafs_count
    
    chosen_leaf_depth = selected_depth + 1
    
    leafs_counts[chosen_leaf_depth] += added_leafs_count
    
    return chosen_leaf_depth


def add_value_cut_inner(depth_degree_allocation : list[list[int]], leaves_counts : list[int], min_degree: int, max_degree: int):
    
    tree_depth = len(depth_degree_allocation)

    chosen_i, chosen_j = None, None
    for i in range(0, tree_depth - 1):
        if max_degree is not None and depth_degree_allocation[i][-1] + 1 > max_degree:
            continue
        for j in range(tree_depth - 1, i, -1):
            if depth_degree_allocation[j][0] - 1 >= min_degree:
                chosen_i, chosen_j = i, j
                break
        if chosen_j is not None:
            break

    if chosen_j is None:
        return None
     
    added_leafs_count = random.randint(1, min(max_degree - depth_degree_allocation[chosen_i][-1], depth_degree_allocation[chosen_j][0] - min_degree))
    
    depth_degree_allocation[chosen_i][-1] += added_leafs_count
    depth_degree_allocation[chosen_i].sort(reverse=True)

    depth_degree_allocation[chosen_j][0] -= added_leafs_count
    depth_degree_allocation[chosen_j].sort(reverse=True)

    
    # incrementally update the depth list from chosen_j onwards
    for k in range(1, tree_depth - chosen_j):
        # move the end added_slots elements from depth_list[k + 1] to depth_list[k]
        depth_degree_allocation[chosen_i + k] += depth_degree_allocation[chosen_j + k][:added_leafs_count]
        new_added_leafs_count = sum(depth_degree_allocation[chosen_j + k][:added_leafs_count])
        depth_degree_allocation[chosen_j + k] = depth_degree_allocation[chosen_j + k][added_leafs_count:]
        depth_degree_allocation[chosen_i + k].sort(reverse=True)
        added_leafs_count = new_added_leafs_count

    chosen_leaf_depth = chosen_i + tree_depth - chosen_j

    leaves_counts[chosen_leaf_depth] += added_leafs_count

    return chosen_leaf_depth


def alter_depth_degree_allocation(depth_degree_allocation : list[list[int]], leafs_counts : list[int], current_it : int, degree_values_list : list[int], min_degree: int, max_degree: int):
    chosen_depth = swap_degree_values_outer(depth_degree_allocation, leafs_counts, current_it, degree_values_list)
    if chosen_depth is not None:
        return chosen_depth
    chosen_depth = swap_degree_values_inner(depth_degree_allocation, leafs_counts)
    if chosen_depth is not None:
        return chosen_depth
    chosen_depth = add_value_cut_outer(depth_degree_allocation, leafs_counts, current_it, degree_values_list, min_degree, max_degree)
    if chosen_depth is not None:
        return chosen_depth
    chosen_depth = add_value_cut_inner(depth_degree_allocation, leafs_counts, min_degree, max_degree)
    return chosen_depth


def accumulate_degree_value(depth_degree_allocation : list[list[int]], leafs_counts : list[int], max_degree: int, current_it : int, degree_values_list : list[int]):
    """
    Exploit remaining degree values to increase the degree value of existing tree (existing depth_degree_allocation).
    """
    tree_depth = len(depth_degree_allocation)
    outer_degree_value_remaining = degree_values_list[current_it]
    d = 0
    i = 0
    while outer_degree_value_remaining > 0 and d < tree_depth:
        new_degree_value = min(max_degree, depth_degree_allocation[d][i] + outer_degree_value_remaining)
        added_leafs_count = new_degree_value - depth_degree_allocation[d][i]
        depth_degree_allocation[d][i] = new_degree_value
        if d + 1 < tree_depth:
            leafs_counts[d + 1] += added_leafs_count
        outer_degree_value_remaining -= added_leafs_count
        i += 1
        if i >= len(depth_degree_allocation[d]):
            d += 1
            i = 0
    return outer_degree_value_remaining


def assign_depth_degree(degree_values_list : list[int], min_depth: int = 1, max_depth: int = None, min_degree : int = 2, max_degree : int = None, strict_num_internal_nodes : bool = False, min_num_internals : int = 1):
    """
    Assigns depth and degree to each non-leaf node (internal node) in the tree, based on the node's children count, and accounting for constraints. 

    The target of this function is to make sure the depth of the entire tree is >= min_depth and <= max_depth.
    """
    assert min_depth >= 1
    assert max_depth is None or min_depth is None or max_depth >= min_depth
    assert min_degree >= 2
    assert max_degree is None or max_degree >= min_degree
    if max_degree is None:
        max_degree = float("inf")
    
    depth_degree_allocation : list[list[int]] = [] # depth -> list of nodes's degree value. The node's have depth = depth.
    leafs_counts : list[int] = [1] # depth -> number of leafs of that depth

    internal_nodes_count = len(degree_values_list)

    for it in range(len(degree_values_list)):
        # choose a random element and its index from the slots list, where the element value is > 0
        if len(depth_degree_allocation) < min_depth:
            chosen_leaf_depth = len(depth_degree_allocation)
        else:
            available_depths = [d for d, v in enumerate(leafs_counts) if v > 0]

            if len(available_depths) > 0:
                chosen_leaf_depth = random.choice(available_depths)
            else:
                chosen_leaf_depth = alter_depth_degree_allocation(depth_degree_allocation, leafs_counts, it, degree_values_list, min_degree, max_degree)
                if chosen_leaf_depth is None:
                    if strict_num_internal_nodes or internal_nodes_count <= min_num_internals:
                        return None
                    else:
                        outer_degree_value_remaining = accumulate_degree_value(depth_degree_allocation, leafs_counts, max_degree, it, degree_values_list)
                        if outer_degree_value_remaining > 0:
                            return None
                        else:
                            internal_nodes_count -= 1
                            continue


        if len(depth_degree_allocation) <= chosen_leaf_depth:
            depth_degree_allocation.append([])
        
        degree_value = degree_values_list[it]

        depth_degree_allocation[chosen_leaf_depth].append(degree_value)

        depth_degree_allocation[chosen_leaf_depth].sort(reverse=True)
        
        leafs_counts[chosen_leaf_depth] -= 1
        
        if max_depth is None or chosen_leaf_depth <= max_depth:
            if chosen_leaf_depth + 1 >= len(leafs_counts):
                leafs_counts.append(degree_value)
            else:
                leafs_counts[chosen_leaf_depth + 1] += degree_value

    return depth_degree_allocation, internal_nodes_count


def form_tree(depth_degree_allocation: list[list[int]]):
    """
    Form a tree from a depth -> degree allocation.
    """
    parent_info : dict[int, int] = {}
    depth_info : dict[int, int] = {}
    children_info : dict[int, list[int]] = {}
    global_id_iterator = 0
    root_id = global_id_iterator
    children_info[root_id] = []
    frontiers: list[list[int]] = [[] for d in range(len(depth_degree_allocation) + 1)]
    frontiers[0] = [root_id]
    for depth, degree_values in enumerate(depth_degree_allocation):
        for degree in degree_values:
            parent_id_index = random.randint(0, len(frontiers[depth]) - 1)
            parent_id = frontiers[depth].pop(parent_id_index)
            if parent_id not in children_info:
                children_info[parent_id] = []
            for _ in range(degree):
                global_id_iterator += 1
                frontiers[depth + 1].append(global_id_iterator)
                parent_info[global_id_iterator] = parent_id
                depth_info[global_id_iterator] = depth
                children_info[parent_id].append(global_id_iterator)
    return root_id, parent_info, depth_info, children_info, frontiers



def gen_random_tree(num_leafs : int, min_depth : int = 1, max_depth : int = None, num_non_leafs : int = None, min_degree : int = 2, max_degree : int = None, min_num_non_leafs : int = 1, max_num_non_leafs : int = None):
    """
    Generates a random tree with the given number of leafs, minimum depth, maximum depth, non_leafs_count, and the given maximum and minimum number of children possible per non-leaf node.

    Priority is given to the minimum depth, then to the number of non-leaf nodes, then to the minimum children count, then max children coun, then finally max depth.
    """
    depth_info : dict[int, int] = {}
    parent_info : dict[int, int] = {}
    children_info : dict[int, list[int]] = {}

    depth_info[0] = 0

    assert min_degree >= 2
    assert max_degree is None or max_degree >= min_degree
    assert min_depth >= 1
    assert max_depth is None or max_depth >= min_depth

    if max_degree is None:
        max_degree = num_leafs

    strict_num_non_leafs = num_non_leafs is not None

    if num_non_leafs is None:
        num_non_leafs = gen_num_non_leafs(min_depth, min_degree, max_degree, num_leafs, min_num_non_leafs, max_num_non_leafs)

    degree_values_list = gen_random_composition(
        n=num_leafs + num_non_leafs - 1,
        lower_bound = min_degree,
        upper_bound = max_degree,
        size=num_non_leafs,
    )

    depth_degree_allocation, internal_nodes_count = assign_depth_degree(
        degree_values_list=degree_values_list, 
        min_depth=min_depth, 
        max_depth=max_depth, 
        min_degree=min_degree, 
        max_degree=max_degree, 
        strict_num_internal_nodes=strict_num_non_leafs, 
        min_num_internals=min_num_non_leafs
    )

    root_id, parent_info, depth_info, children_info, frontiers = form_tree(depth_degree_allocation)

    tree_depth = max(depth_info.values())
    
    leaf_nodes = [node_id for node_id in depth_info if children_info.get(node_id, None) is None or len(children_info[node_id]) == 0]

    avg_branching_factor = (num_leafs + num_non_leafs - 1) / num_non_leafs
    
    return depth_info, parent_info, children_info, leaf_nodes, frontiers, num_non_leafs, tree_depth, avg_branching_factor

