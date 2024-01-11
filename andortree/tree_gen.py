import random

def gen_tree(
        num_leaves: int, 
        min_num_internals : int = 1,  
        max_num_internals : int = None, 
        min_depth : int = 1, 
        max_depth : int = None, 
        min_degree : int = 2, 
        max_degree : int = None
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
    
    leaves : list[list[int]] = [[root_id]] # depth -> list of leaves of that depth

    current_num_leaves = 1
    current_num_internals = 0

    while current_num_leaves < num_leaves:

        if len(leaves) - 1 < min_depth:
            chosen_depth = len(leaves) - 1
            parent_id_index = random.randint(0, len(leaves[chosen_depth]) - 1)
            parent_id = leaves[chosen_depth].pop(parent_id_index)

            current_num_internals += 1
            current_num_leaves -= 1
            
            # Choose a random value for the degree of the parent
            
            tba_degree_lower_bound = max(min_degree, (num_leaves - current_num_leaves) - (max_degree - 1) * max(0, max_num_internals - current_num_internals))
            
            tba_degree_upper_bound = min(max_degree, (num_leaves - current_num_leaves) - (min_degree - 1) * max(0, min_num_internals - current_num_internals))
            
            tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)

        else:
            # Choose a random depth for the parent (the leaf to add children to)
            depths_options = [d for d, v in enumerate(leaves) if len(v) > 0 and d < max_depth]

            if len(depths_options) > 0:
                
                # Choose a random depth for the parent (the leaf to add children to)
                chosen_depth = random.choice(depths_options)
                
                # Choose a random leaf from the current depth
                parent_id_index = random.randint(0, len(leaves[chosen_depth]) - 1)
                parent_id = leaves[chosen_depth].pop(parent_id_index)

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
                    return None
                    
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
                            leaves[old_depth].remove(current_node)
                            leaves[new_depth].append(current_node)
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
                    return None
                
                tba_degree = random.randint(tba_degree_lower_bound, tba_degree_upper_bound)
            
        if parent_id not in children_info:
            children_info[parent_id] = []
        
        if chosen_depth + 1 >= len(leaves):
            leaves.append([])

        for _ in range(tba_degree):
            global_id_iterator += 1
            leaves[chosen_depth + 1].append(global_id_iterator)
            parent_info[global_id_iterator] = parent_id
            depth_info[global_id_iterator] = chosen_depth + 1
            children_info[parent_id].append(global_id_iterator)
            current_num_leaves += 1

    return depth_info, parent_info, children_info, leaves