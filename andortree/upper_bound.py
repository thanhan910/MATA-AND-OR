import numpy as np

from .tree_types import Node, NodeType

def calc_upper_bound_vectors_tree(tree_info: list[Node], capabilities, tasks, root_node_index=-1):
    nodes_upperBound_capVector = [np.zeros(len(capabilities)) for _ in range(0, len(tree_info))]
    tasks_capVector = [np.array([1 if c in tasks[j] else 0 for c in capabilities]) for j in range(0, len(tasks))]

    def _calc_upper_bound_vector_node(node_index : int):
        node = tree_info[node_index]
        if node.node_type == NodeType.LEAF:
            nodes_upperBound_capVector[node_index] = tasks_capVector[node.node_id]
            return nodes_upperBound_capVector[node_index]
        if node.node_type == NodeType.OR:
            for child_index in node.children_ids:
                _calc_upper_bound_vector_node(child_index)
            nodes_upperBound_capVector[node_index] = np.max([nodes_upperBound_capVector[child_index] for child_index in node.children_ids], axis=0)
            return nodes_upperBound_capVector[node_index]
        if node.node_type == NodeType.AND:
            for child_index in node.children_ids:
                _calc_upper_bound_vector_node(child_index)
            nodes_upperBound_capVector[node_index] = np.sum([nodes_upperBound_capVector[child_index] for child_index in node.children_ids], axis=0)
            return nodes_upperBound_capVector[node_index]
    
    _calc_upper_bound_vector_node(root_node_index)
    return nodes_upperBound_capVector
    

def upperBoundTree_root(tree_info: list[Node], capabilities, tasks, agents, root_node_index=-1):
    """
    Calculate the upper bound of the system reward, i.e. at the root of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(tree_info, capabilities, tasks, root_node_index)
    caps_ranked = [sorted([a[c] for a in agents], reverse=True) for c in capabilities]
    
    return sum([sum(caps_ranked[c][:int(nodes_upperBound_capVector[root_node_index][c])]) for c in capabilities])


def upperBoundTree_allNodes_v1(tree_info: list[Node], capabilities, tasks, agents, nodes_constraints, root_node_index=-1):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(tree_info, capabilities, tasks, root_node_index)
    
    nodes_agents = nodes_constraints[1]
    
    nodes_caps_ranked = [[sorted([agents[i][c] for i in nodes_agents[node_index]], reverse=True) for c in capabilities] for node_index in range(0, len(tree_info))]
    
    return [sum([sum(nodes_caps_ranked[node_index][c][:int(nodes_upperBound_capVector[node_index][c])]) for c in capabilities]) for node_index in range(0, len(tree_info))]



def upperBoundTree_allNodes_v2(tree_info: list[Node], capabilities, tasks, agents, nodes_constraints, root_node_index=-1):
    """
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.
    """
    nodes_upperBound_capVector = calc_upper_bound_vectors_tree(tree_info, capabilities, tasks, root_node_index)
    
    nodes_agents = nodes_constraints[1]
    
    nodes_caps_ranked = [[sorted([agents[i][c] for i in nodes_agents[node_index]], reverse=True) for c in capabilities] for node_index in range(0, len(tree_info))]
    
    nodes_upper_bound = [sum([sum(nodes_caps_ranked[node_index][c][:int(nodes_upperBound_capVector[node_index][c])]) for c in capabilities]) for node_index in range(0, len(tree_info))]

    nodes_upper_bound_min = [0 for node_index in range(0, len(tree_info))]

    def _min_upper_bound(node_id : int):
        node = tree_info[node_id]
        if node.node_type == NodeType.LEAF:
            nodes_upper_bound_min[node_id] = nodes_upper_bound[node_id]
            return nodes_upper_bound_min[node_id]
        if node.node_type == NodeType.OR:
            nodes_upper_bound_min[node_id] = max(_min_upper_bound(child_index) for child_index in node.children_ids)
            nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
            return nodes_upper_bound_min[node_id]
        if node.node_type == NodeType.AND:
            nodes_upper_bound_min[node_id] = sum(_min_upper_bound(child_index) for child_index in node.children_ids)
            nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
            return nodes_upper_bound_min[node_id]
        if node.node_type == NodeType.DUMMY:
            nodes_upper_bound_min[node_id] = 0
            return nodes_upper_bound_min[node_id]
        else:
            raise Exception("Unknown node type")
        
    _min_upper_bound(root_node_index)

    return nodes_upper_bound_min