import random
from .node_type import NodeType, reverse_node_type


def gen_tree(
        num_leaves: int, 
        min_num_internals : int = 1,  
        max_num_internals : int = None, 
        min_depth : int = 1, 
        max_depth : int = None, 
        min_degree : int = 2, 
        max_degree : int = None,
        min_leaf_depth: int = 0,
    ):
    """
    Generates a random tree with the given number of leafs, minimum depth, maximum depth, maximum and minimum depth (number of children possible per internal (non-leaf) node), and maximum and minimum number of internal nodes.

    If you need exact number of internal nodes, set min_num_internals = max_num_internals = exact number of internal nodes.

    If you need exact depth, set min_depth = max_depth = exact depth.

    If you need exact degree, set min_degree = max_degree = exact degree.
    """

    if max_degree is None:
        max_degree = num_leaves

    if max_num_internals is None:
        max_num_internals = num_leaves - 1

    if max_depth is None:
        max_depth = float("inf")

    depth_info : dict[int, int] = {}
    parent_info : dict[int, int] = {}
    children_info : dict[int, list[int]] = {}
    
    global_id_iterator = 0
    root_id = global_id_iterator
    children_info[root_id] = []
    depth_info[root_id] = 0
    
    leaves_by_depth : list[list[int]] = [[root_id]] # depth -> list of leaves of that depth

    current_num_leaves = 1
    current_num_internals = 0

    while current_num_leaves < num_leaves:

        shallow_leaves = {d: l for d, l in enumerate(leaves_by_depth) if len(l) > 0 and d < min_leaf_depth}
        if len(shallow_leaves) > 0:
            chosen_depth = random.choice(list(shallow_leaves.keys()))

            # Choose a random value for the degree of the parent
            
            current_num_internals += 1
            
            min_tba_num_leaves = sum([len(l) * (min_degree ** (min_leaf_depth - d) - 1) for d, l in shallow_leaves.items()])

            tba_degree_upper_bound_2 = min_degree + (num_leaves - current_num_leaves - min_tba_num_leaves) // (min_degree ** (min_leaf_depth - chosen_depth - 1))

            current_num_leaves -= 1

            tba_degree_lower_bound = max(min_degree, (num_leaves - current_num_leaves) - (max_degree - 1) * max(0, max_num_internals - current_num_internals))

            tba_degree_upper_bound_1 = (num_leaves - current_num_leaves) - (min_degree - 1) * max(0, min_num_internals - current_num_internals)

            tba_degree_upper_bound = min(max_degree, tba_degree_upper_bound_1, tba_degree_upper_bound_2)
            
            if tba_degree_lower_bound > tba_degree_upper_bound:
                raise Exception("No valid tree exists with the given parameters.")
            
            tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)

            parent_id_index = random.randint(0, len(leaves_by_depth[chosen_depth]) - 1)
            parent_id = leaves_by_depth[chosen_depth].pop(parent_id_index)

        elif len(leaves_by_depth) - 1 < min_depth:
            chosen_depth = len(leaves_by_depth) - 1
            parent_id_index = random.randint(0, len(leaves_by_depth[chosen_depth]) - 1)
            parent_id = leaves_by_depth[chosen_depth].pop(parent_id_index)

            current_num_internals += 1
            current_num_leaves -= 1
            
            # Choose a random value for the degree of the parent
            
            tba_degree_lower_bound = max(min_degree, (num_leaves - current_num_leaves) - (max_degree - 1) * max(0, max_num_internals - current_num_internals))
            
            tba_degree_upper_bound = min(max_degree, (num_leaves - current_num_leaves) - (min_degree - 1) * max(0, min_num_internals - current_num_internals))
            
            tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)

        else:
            # Choose a random depth for the parent (the leaf to add children to)
            depths_options = [d for d, v in enumerate(leaves_by_depth) if len(v) > 0 and d < max_depth]

            if len(depths_options) > 0:
                
                # Choose a random depth for the parent (the leaf to add children to)
                chosen_depth = random.choice(depths_options)
                
                # Choose a random leaf from the current depth
                parent_id_index = random.randint(0, len(leaves_by_depth[chosen_depth]) - 1)
                parent_id = leaves_by_depth[chosen_depth].pop(parent_id_index)

                current_num_internals += 1
                current_num_leaves -= 1
                
                # Choose a random value for the degree of the parent
                
                tba_degree_lower_bound = max(min_degree, (num_leaves - current_num_leaves) - (max_degree - 1) * max(0, max_num_internals - current_num_internals))
                
                tba_degree_upper_bound = min(max_degree, (num_leaves - current_num_leaves) - (min_degree - 1) * max(0, min_num_internals - current_num_internals))
                
                tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)

            else:
                # Choose a random internal non-full node to add children to.
                non_max_degree_node_options = [_node for _node, _children in children_info.items() if len(_children) < max_degree]
                
                if len(non_max_degree_node_options) == 0:
                    raise Exception("No valid tree exists with the given parameters.")
                    
                # Select a pair of nodes, where the first one is not maximum degree and the second one is not minimum degree, such that the first one is higher (closer to the root) than the second one

                highest_node = min(non_max_degree_node_options, key=lambda _node: depth_info[_node], default=None)

                non_min_degree_node_options = [_node for _node, _children in children_info.items() if len(_children) > min_degree]
                
                deepest_node = max(non_min_degree_node_options, key=lambda _node: depth_info[_node], default=None)

                # If no such pair exists, then the method is not valid. We move to another method: add leafs to existing internal nodes.
                if highest_node is not None and deepest_node is not None and depth_info[highest_node] < depth_info[deepest_node]:
                
                    # Move an edge under the deepest_non_min_degree_node to under the highest_non_max_degree_node
                    
                    # Choose a random child of the deepest_non_min_degree_node
                    child_id = random.choice(children_info[deepest_node])
                    # Remove the child from the children list of the deepest_non_min_degree_node
                    children_info[deepest_node].remove(child_id)
                    # Add the child to the children list of the highest_non_max_degree_node
                    children_info[highest_node].append(child_id)
                    # Update the parent_info
                    parent_info[child_id] = highest_node
                    # Update the depth_info
                    depth_info[child_id] = depth_info[highest_node] + 1
                    # Update depth_info of all descendants of the child, using BFS
                    queue = [child_id]
                    while len(queue) > 0:
                        current_node = queue.pop(0)
                        old_depth = depth_info[current_node]
                        new_depth = depth_info[parent_info[current_node]] + 1
                        if current_node not in children_info:
                            # Update leaves
                            leaves_by_depth[old_depth].remove(current_node)
                            leaves_by_depth[new_depth].append(current_node)
                        depth_info[current_node] = new_depth
                        queue += children_info[current_node]
                    continue
                
                # else:
                
                # Add leafs to existing internal nodes.

                parent_id = random.choice(non_max_degree_node_options)

                # Get the depth of the parent
                chosen_depth = depth_info[parent_id]

                # Choose a random value for the to-be-added degree of the parent

                tba_degree_lower_bound = max(1, (num_leaves - current_num_leaves) - (max_degree - 1) * max(0, max_num_internals - current_num_internals))
                
                tba_degree_upper_bound = min(
                    max_degree - len(children_info[parent_id]),
                    (num_leaves - current_num_leaves) - (min_degree - 1) * max(0, min_num_internals - current_num_internals)
                )

                if tba_degree_lower_bound > tba_degree_upper_bound:
                    raise Exception("No valid tree exists with the given parameters.")
                
                tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)
            
        if parent_id not in children_info:
            children_info[parent_id] = []
        
        if chosen_depth + 1 >= len(leaves_by_depth):
            leaves_by_depth.append([])

        for _ in range(tba_degree):
            global_id_iterator += 1
            leaves_by_depth[chosen_depth + 1].append(global_id_iterator)
            parent_info[global_id_iterator] = parent_id
            depth_info[global_id_iterator] = chosen_depth + 1
            children_info[parent_id].append(global_id_iterator)
            current_num_leaves += 1

    return depth_info, parent_info, children_info, leaves_by_depth



def assign_node_type(
        depth_info: dict[int, int], 
        children_info: dict[int, int], 
        leaf_nodes : list[int], 
        root_node_id : int = 0, 
        strict_and_or: bool = True, 
        root_node_type : NodeType = None
    ):
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


def gen_tree_simple(
        min_depth : int = 1, 
        max_depth : int = 1000, 
        min_degree : int = 2, 
        max_degree : int = 1000,
        min_leaf_depth: int = 0,
        eps: float = 0.1,
    ):
    """
    Generates a random tree with the given minimum depth, maximum depth, maximum and minimum depth (number of children possible per internal (non-leaf) node)

    If you need exact depth, set min_depth = max_depth = exact depth.

    If you need exact degree, set min_degree = max_degree = exact degree.

    eps is the probability of continuing the tree generation process after the minimum depth and minimum leaf depth conditions are satisfied.
    """

    depth_info : dict[int, int] = {}
    parent_info : dict[int, int] = {}
    children_info : dict[int, list[int]] = {}
    
    global_id_iterator = 0
    root_id = global_id_iterator
    children_info[root_id] = []
    depth_info[root_id] = 0
    
    leaves_by_depth : list[list[int]] = [[root_id]] # depth -> list of leaves of that depth

    must_continue = True


    while must_continue:

        shallow_leaves = {d: l for d, l in enumerate(leaves_by_depth) if len(l) > 0 and d < min_leaf_depth}
        
        if len(shallow_leaves) > 0:
            chosen_depth = random.choice(list(shallow_leaves.keys()))

        elif len(leaves_by_depth) - 1 < min_depth:
            chosen_depth = len(leaves_by_depth) - 1

        else:
            # Choose a random depth for the parent (the leaf to add children to)
            depths_options = [d for d, v in enumerate(leaves_by_depth) if len(v) > 0 and d < max_depth]

            if len(depths_options) > 0:
                
                # Choose a random depth for the parent (the leaf to add children to)
                chosen_depth = random.choice(depths_options)

                must_continue = random.random() < eps

            else:
                break

        # Choose a random leaf from the current depth as the parent
        parent_id_index = random.randint(0, len(leaves_by_depth[chosen_depth]) - 1)
        parent_id = leaves_by_depth[chosen_depth].pop(parent_id_index)
        
        # Choose a random value for the to-be-added degree of the parent
        tba_degree = random.randint(min_degree, max_degree)

        if parent_id not in children_info:
            children_info[parent_id] = []
        
        if chosen_depth + 1 >= len(leaves_by_depth):
            leaves_by_depth.append([])

        for _ in range(tba_degree):
            global_id_iterator += 1
            leaves_by_depth[chosen_depth + 1].append(global_id_iterator)
            parent_info[global_id_iterator] = parent_id
            depth_info[global_id_iterator] = chosen_depth + 1
            children_info[parent_id].append(global_id_iterator)

    return depth_info, parent_info, children_info, leaves_by_depth


