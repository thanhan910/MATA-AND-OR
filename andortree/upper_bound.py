import numpy as np

from .node_type import NodeType
from .tree_utils import traverse_tree


def get_cap_vector(capabilities : list[int], tasks : list[list[int]], query_taskId : int):
    """
    Get the capability requirement vector of a task.
    
    Also called a "ubc vector".
    """
    cap_vec = np.zeros(len(capabilities))
    for c in tasks[query_taskId]:
        cap_vec[c] = 1
    return cap_vec


def get_cap_vector_all(capabilities : list[int], tasks : list[list[int]]):
    """
    Get the capability requirement vectors of all tasks.

    Also called a "ubc vector".
    """
    return [get_cap_vector(capabilities, tasks, j) for j in range(0, len(tasks))]


def calculate_ubc_vectors(
        node_type_info : dict[int, NodeType],
        parent_info : dict[int, int], 
        leaves_list_info : dict[int, list[int]],
        leaf2task : dict[int, int],
        tasks_capVecs : list[np.ndarray],
        capabilities : list[int], 
        query_nodeId : int
    ):
    """
    For each descendant of 'query_nodeId', generates a "ubc vector", a vector of values.

    Each value in the vector represents each capability and the upper bound of the number of agents that needed to perform that capability under that node. "ubc" stands for "upper bound capability".
    """
    if node_type_info[query_nodeId] == NodeType.LEAF:
        return {query_nodeId : tasks_capVecs[leaf2task[query_nodeId]]}
    
    leaf_nodes = leaves_list_info[query_nodeId] if query_nodeId in leaves_list_info else []
    
    ubcv_info = { n_id : np.zeros(len(capabilities)) for n_id in leaf_nodes }
    
    for leaf_id in leaf_nodes:
        ubcv_info[leaf_id] = tasks_capVecs[leaf2task[leaf_id]]
        current_node_id = leaf_id
        while current_node_id in parent_info and current_node_id != query_nodeId:
            prev_node_id = current_node_id
            current_node_id = parent_info[current_node_id]
            node_type = node_type_info[current_node_id]
            if node_type == NodeType.OR:
                ubcv_info[current_node_id] = np.max([ubcv_info.get(current_node_id, np.zeros(len(capabilities))), ubcv_info[prev_node_id]], axis=0)
            elif node_type == NodeType.AND:
                ubcv_info[current_node_id] = np.sum([ubcv_info.get(current_node_id, np.zeros(len(capabilities))), ubcv_info[prev_node_id]], axis=0)

    return ubcv_info
    

def upperbound_node(
        ubcv_info : dict[int, np.ndarray],
        capabilities : list[int], 
        agents : list[list[float]],
        nodes_constraints : tuple[list[list[int]], dict[int, list[int]]],
        query_nodeId=0 
    ):
    """
    Calculate the upper bound of the system reward, i.e. at the root of the AND-OR goal tree.
    """
    nodes_agents = nodes_constraints[1]

    caps_ranked = [sorted([agents[i][c] for i in nodes_agents[query_nodeId]], reverse=True) for c in capabilities]

    cap_req_num = ubcv_info[query_nodeId]
    
    return sum([sum(caps_ranked[c][:int(cap_req_num[c])]) for c in capabilities])


def upperbound_node_all(
        children_info : dict[int, list[int]],
        ubcv_info : dict[int, np.ndarray],
        capabilities : list[int], 
        agents : list[list[float]],
        nodes_constraints : tuple[list[list[int]], dict[int, list[int]]],
        query_nodeId=0 
    ):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.
    """

    descendant_nodes = list(traverse_tree(children_info, root_node_id=query_nodeId))
    
    nodes_caps_ranked = { 
        node_id : upperbound_node(
            ubcv_info,
            capabilities,
            agents,
            nodes_constraints,
            query_nodeId=node_id
        )
        for node_id in descendant_nodes
    }
    
    return {
        node_id : sum([sum(nodes_caps_ranked[node_id][c][:int(ubcv_info[node_id][c])]) for c in capabilities]) 
        for node_id in descendant_nodes
    }



def upperbound_node_all_min(
        nodes_upper_bound : dict[int, float],
        node_type_info : dict[int, NodeType],
        children_info : dict[int, list[int]],
        query_nodeId=0
    ):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.

    Refine the upper bound by taking the minimum of the upper bound calculated from the children nodes, and the upper bound calculated from the current node.
    """

    nodes_upper_bound_min = { node_id : 0 for node_id in nodes_upper_bound }

    def _min_upper_bound(node_id : int):
        node_type = node_type_info[node_id]

        if node_type == NodeType.LEAF:
            nodes_upper_bound_min[node_id] = nodes_upper_bound[node_id]

        elif node_type == NodeType.OR:
            nodes_upper_bound_min[node_id] = max(_min_upper_bound(child_id) for child_id in children_info[node_id])

        elif node_type == NodeType.AND:
            nodes_upper_bound_min[node_id] = sum(_min_upper_bound(child_id) for child_id in children_info[node_id])

        else:
            raise Exception("Unsupported node type")

        nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
        return nodes_upper_bound_min[node_id]
        
    _min_upper_bound(query_nodeId)

    return nodes_upper_bound_min