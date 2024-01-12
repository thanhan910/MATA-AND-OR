import numpy as np

from .tree_types import NodeType

def calc_upper_bound_vectors_tree(depth_info, node_type_info, children_info, capabilities, tasks, root_node_id=0):
    nodes_upperBound_capVector = { _node_id : np.zeros(len(capabilities)) for _node_id in depth_info }
    tasks_capVector = [np.array([1 if c in tasks[j] else 0 for c in capabilities]) for j in range(0, len(tasks))]

    def _calc_upper_bound_vector_node(node_id : int):
        node_type = node_type_info[node_id]
        if node_type == NodeType.LEAF:
            nodes_upperBound_capVector[node_id] = tasks_capVector[node_id]
            return nodes_upperBound_capVector[node_id]
        if node_type == NodeType.OR:
            for child_id in children_info[node_id]:
                _calc_upper_bound_vector_node(child_id)
            nodes_upperBound_capVector[node_id] = np.max([nodes_upperBound_capVector[child_index] for child_index in children_info[node_id]], axis=0)
            return nodes_upperBound_capVector[node_id]
        if node_type == NodeType.AND:
            for child_id in children_info[node_id]:
                _calc_upper_bound_vector_node(child_id)
            nodes_upperBound_capVector[node_id] = np.sum([nodes_upperBound_capVector[child_index] for child_index in children_info[node_id]], axis=0)
            return nodes_upperBound_capVector[node_id]
    
    _calc_upper_bound_vector_node(root_node_id)
    return nodes_upperBound_capVector
    

def upperBoundTree_root(depth_info, node_type_info, children_info, capabilities, tasks, agents, root_node_id=0):
    """
    Calculate the upper bound of the system reward, i.e. at the root of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(depth_info, node_type_info, children_info, capabilities, tasks, root_node_id)
    caps_ranked = [sorted([a[c] for a in agents], reverse=True) for c in capabilities]
    
    return sum([sum(caps_ranked[c][:int(nodes_upperBound_capVector[root_node_id][c])]) for c in capabilities])


def upperBoundTree_allNodes_v1(depth_info, node_type_info, children_info, capabilities, tasks, agents, nodes_constraints, root_node_id=0):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(depth_info, node_type_info, children_info, capabilities, tasks, root_node_id)
    
    nodes_agents = nodes_constraints[1]
    
    nodes_caps_ranked = { 
        node_id : [sorted([agents[i][c] for i in nodes_agents[node_id]], reverse=True) for c in capabilities] 
        for node_id in depth_info
    }
    
    return {
        node_id : sum([sum(nodes_caps_ranked[node_id][c][:int(nodes_upperBound_capVector[node_id][c])]) for c in capabilities]) 
        for node_id in depth_info
    }



def upperBoundTree_allNodes_v2(depth_info, node_type_info, children_info, capabilities, tasks, agents, nodes_constraints, root_node_id=0):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(depth_info, node_type_info, children_info, capabilities, tasks, root_node_id)
    
    nodes_agents = nodes_constraints[1]
    
    nodes_caps_ranked = {
        node_id : [sorted([agents[i][c] for i in nodes_agents[node_id]], reverse=True) for c in capabilities] 
        for node_id in depth_info
    }
    
    nodes_upper_bound = {
        node_id : sum([sum(nodes_caps_ranked[node_id][c][:int(nodes_upperBound_capVector[node_id][c])]) for c in capabilities]) 
        for node_id in depth_info
    }

    nodes_upper_bound_min = {node_id : 0 for node_id in depth_info}

    def _min_upper_bound(node_id : int):
        node_type = node_type_info[node_id]
        if node_type == NodeType.LEAF:
            nodes_upper_bound_min[node_id] = nodes_upper_bound[node_id]
            return nodes_upper_bound_min[node_id]
        if node_type == NodeType.OR:
            nodes_upper_bound_min[node_id] = max(_min_upper_bound(child_id) for child_id in children_info[node_id])
            nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
            return nodes_upper_bound_min[node_id]
        if node_type == NodeType.AND:
            nodes_upper_bound_min[node_id] = sum(_min_upper_bound(child_id) for child_id in children_info[node_id])
            nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
            return nodes_upper_bound_min[node_id]
        if node_type == NodeType.DUMMY:
            nodes_upper_bound_min[node_id] = 0
            return nodes_upper_bound_min[node_id]
        else:
            raise Exception("Unknown node type")
        
    _min_upper_bound(root_node_id)

    return nodes_upper_bound_min