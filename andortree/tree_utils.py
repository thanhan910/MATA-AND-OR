import itertools
from .node_type import NodeType


def get_leaves_list(depth_info : dict[int, int], parent_info : dict[int, int], leaf_nodes : list[int]):
    """
    For each node in the tree, get the list of leaves that are descendants of that node.

    For each leaf node, get the path from that leaf node to the root node.
    """
    leaves_list_info = {node_id : [] for node_id in depth_info}
    for leaf_id in leaf_nodes:
        leaves_list_info[leaf_id].append(leaf_id)
        current_node_id = leaf_id
        while current_node_id in parent_info:
            leaves_list_info[parent_info[current_node_id]].append(current_node_id)
            current_node_id = parent_info[leaf_id]
    
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



def get_nodes_constraints(node_type_info : dict[int, int], leaves_list : dict[int, list[int]], leaf2task: dict[int, int], constraints):
    """
    For each node in the tree, get the list of agents that can perform under that node, and the list of nodes that each agent can perform on.
    """
    nodes_agents = { node : [] for node in node_type_info }
    for node in node_type_info:
        if node_type_info[node] == NodeType.LEAF:
            nodes_agents[node] = constraints[1][node]
        else:
            nodes_agents[node] = list(set(itertools.chain(
                *[constraints[1][leaf2task[leaf]] for leaf in leaves_list[node]]
            ))) # Concat lists and remove duplicates

    a_nodes = [[] for a in constraints[0]]
    for node_id, node_As in enumerate(nodes_agents):
        for a in node_As:
            a_nodes[a].append(node_id)

    return a_nodes, nodes_agents