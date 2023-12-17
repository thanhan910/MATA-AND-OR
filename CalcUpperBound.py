from TreeUtils import Node, NodeType, get_leafs_for_each_node

import itertools
import numpy as np

def upperBound(capabilities, tasks, agents):
    """
    Calculate the upper bound of the system reward, where the system consists of tasks and agents with constraints.

    This mathematical upper bound is calculated by sorting the agents based on their contribution values for each capability, in descending order, then count `m`, the number of tasks that require each capability, and sum up the contribution values of the top `m` agents for each capability.
    
    :param: `capabilities`: the list of capabilities
    :param: `tasks`: the list of tasks
    :param: `agents`: the list of agents
    :return: the upper bound of the system reward
    """
    cap_ranked = [sorted([a[c] for a in agents], reverse=True) for c in capabilities] # Time complexity: O(len(capabilities) * log(len(capabilities)) * len(agents))
    cap_req_all = list(itertools.chain(*tasks)) # Time complexity: O(size of tasks capabilities combined), around O(len(tasks) * len(capabilities))
    cap_req_num = [cap_req_all.count(c) for c in capabilities] # Time complexity: O(len(cap_req_all) * len(capabilities)). However, can be optimized to O(len(cap_req_all)).
    return sum([sum(cap_ranked[c][:cap_req_num[c]]) for c in capabilities]) # Time complexity: O(len(cap_req_all))
    # Evaluated time complexity: max(O(len(capabilities) * log(len(capabilities)) * len(agents)), O(len(tasks) * len(capabilities)))


def upperBound_ver2(capabilities, tasks, agents, constraints):
    """
    Calculate the upper bound of the system reward, where the system consists of tasks and agents with constraints.

    This upper bound is calculated by sorting the agents based on their contribution values for each capability, in descending order, then iteratively allocate the top agents to the tasks that require that capability.

    This allows for a more precise upper bound than upperBound, since it takes into account the `constraints`: the top agents might only be able to work on the same limited tasks.

    :param: `capabilities`: the list of capabilities
    :param: `tasks`: the list of tasks
    :param: `agents`: the list of agents
    :param: `constraints`: the list of constraints
    :return: the upper bound of the system reward
    """
    agent_num = len(agents)
    task_num = len(tasks)
    a_taskInds = constraints[0]
    cap_req_all = list(itertools.chain(*tasks))
    cap_req_num = [cap_req_all.count(c) for c in capabilities]

    sys_rewards = 0
    for c in capabilities:
        
        a_cap_vals = [agent[c] for agent in agents]

        # the list of tasks that each agent has the capability to perform and that require the capability c
        a_cap_tasks = [[j for j in a_taskInd if j != task_num and c in tasks[j]] for a_taskInd in a_taskInds] 

        # sort the agents based on their contribution values for the capability c, in descending order
        cap_rank_pos = np.argsort(a_cap_vals)[::-1]

        a_cap_vals_ordered = [0 for _ in range(0, agent_num)]
        a_cap_tasks_ordered = [[] for _ in range(0, agent_num)]
        for p, pos in enumerate(cap_rank_pos):
            a_cap_vals_ordered[p] = a_cap_vals[pos]
            a_cap_tasks_ordered[p] = a_cap_tasks[pos]

        cap_rewards = a_cap_vals_ordered[0]
        cap_tasks = set(a_cap_tasks_ordered[0])
        a_cap_num = 1
        for a_iter in range(1, agent_num):
            cap_tasks = cap_tasks.union(set(a_cap_tasks_ordered[a_iter]))
            if len(cap_tasks) > a_cap_num:
                cap_rewards += a_cap_vals_ordered[a_iter]
                a_cap_num += 1
            # break if they got enough agents to contribute the number of required cap c
            if (a_cap_num >= cap_req_num[c]):  
                break
        sys_rewards += cap_rewards
    return sys_rewards


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