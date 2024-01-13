import itertools
from .node_type import NodeType


def get_leaves_list(parent_info : dict[int, int], leaf_nodes : list[int]):
    """
    For each node in the tree, get the list of leaves that are descendants of that node.

    For each leaf node, get the path from that leaf node to the root node.
    """
    leaves_list_info : dict[int, list[int]] = { }
    for leaf_id in leaf_nodes:
        # leaves_list_info[leaf_id].append(leaf_id)
        current_node_id = leaf_id
        leaves_list_info[leaf_id] = [leaf_id]
        while current_node_id in parent_info:
            parent_id = parent_info[current_node_id]
            if parent_id not in leaves_list_info:
                leaves_list_info[parent_id] = [leaf_id]
            else:
                leaves_list_info[parent_id].append(leaf_id)
            current_node_id = parent_id
    
    return leaves_list_info


def get_path_to_root(parent_info : dict[int, int], leaf_nodes : list[int]):
    """
    For each leaf node, get the path from that leaf node to the root node.
    """
    path_to_root_info = {node_id : [] for node_id in leaf_nodes}
    for node_id in leaf_nodes:
        current_node_id = node_id
        while node_id in parent_info:
            path_to_root_info[node_id].append(current_node_id)
            current_node_id = parent_info[node_id]
    return path_to_root_info



def traverse_tree(children_info : dict[int, list[int]], order='dfs', root_node_id=0):
    """
    Traverse tree using depth-first search or breath-first search.
    
    Returns a generator that yields the node ids.
    """
    frontier_pop_index = 0 if order.lower() == 'bfs' else -1
    frontier = [root_node_id]
    while len(frontier) > 0:
        node_id = frontier.pop(frontier_pop_index)
        yield node_id
        if node_id in children_info:
            for child_id in children_info[node_id]:
                frontier.append(child_id)



def get_nodes_constraints(node_type_info : dict[int, int], leaves_list_info : dict[int, list[int]], leaf2task: dict[int, int], constraints):
    """
    For each node in the tree, get the list of agents that can perform under that node, and the list of nodes that each agent can perform on.
    """
    nodes_agents_info = { node : [] for node in node_type_info }
    for node in node_type_info:
        if node_type_info[node] == NodeType.LEAF:
            nodes_agents_info[node] = constraints[1][leaf2task[node]]
        else:
            nodes_agents_info[node] = list(set(itertools.chain(
                *[constraints[1][leaf2task[leaf]] for leaf in leaves_list_info[node]]
            ))) # Concat lists and remove duplicates

    a_nodes = [[] for a in constraints[0]]
    for n_id, n_agents in nodes_agents_info.items():
        for a in n_agents:
            a_nodes[a].append(n_id)

    return a_nodes, nodes_agents_info


def traverse_and_or_tree(node_type_info: dict, children_info: dict, depth_info: dict, root_node_id: int = 0):
    """
    Traverses an AND-OR goal tree and yields all possible solutions (subsets of leaves that can be completed to fulfill an AND-OR goal tree).
    """

    # skipped_nodes = set()

    def traverse_helper(node_id: int) -> list:
        
        # print("NODE: ", node_id)

        # if node_type_info[node_id] != NodeType.OR:
        #     if random.random() < 0.1 and depth_info[node_id] > 2:
        #         skipped_nodes.add(node_id)
        #         return

        # If the node is a leaf node (no children)
        if node_type_info[node_id] == NodeType.LEAF:
            # print("YIELD: ", node_id)
            yield [node_id], [node_id]
            return

        # For AND nodes, need to combine children subsets
        if node_type_info[node_id] == NodeType.AND:
            
            # leaves_subsets = [list(traverse_and_or_tree(child)) for child in children_info[node_id]]

            stack = [([], [node_id], 0)]
            
            while stack:
                combination, combination_path, index = stack.pop()
                if index >= len(children_info[node_id]):
                    yield combination, combination_path
                else:
                    # for item in leaves_subsets[index]:
                    for item, path in traverse_helper(children_info[node_id][index]):
                        stack.append((combination + item, combination_path + path, index + 1))

        
        # For OR nodes, simply yield from each child
        elif node_type_info[node_id] == NodeType.OR:

            # num_child = len(children_info[node_id])
            
            for child in children_info[node_id]:
                # if random.random() < 0.1 and depth_info[child] > 2 and num_child > 1:
                #     num_child -= 1
                #     skipped_nodes.add(child)
                #     continue
                for item, path in traverse_helper(child):
                    yield item, [node_id] + path

    yield from traverse_helper(root_node_id)

    # return list(traverse_helper(root_node_id)), skipped_nodes


def AO_star(node_id: int, children_info: dict[int, list[int]], node_type_info: dict[int, NodeType], reward_function : dict[int, float]):
    """
    Performs AO* search on a tree with the given root node id.
    """
    node_type = node_type_info[node_id]
    
    if node_type == NodeType.LEAF:
        return reward_function[node_id], [node_id]

    if node_type == NodeType.AND:
        total_reward = 0
    else: # OR node
        total_reward = float('-inf')

    best_solution = []

    for child_id in children_info[node_id]:

        child_reward, child_solution = AO_star(child_id, children_info, node_type_info, reward_function)

        if node_type == NodeType.AND:
            # child_reward, child_solution = AO_star(child_id, children_info, node_type_info, reward_function)
            total_reward += child_reward
            best_solution += child_solution

        elif child_reward > total_reward:
            # child_reward, child_solution = AO_star(child_id, children_info, node_type_info, reward_function)
            total_reward = child_reward
            best_solution = child_solution

    # return total_reward, [node_id] + best_solution
    return total_reward, best_solution