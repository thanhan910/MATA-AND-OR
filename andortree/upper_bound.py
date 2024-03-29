import itertools
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
    For each descendant of 'query_nodeId', generate a "ubc vector", a vector of values.

    Each value in the vector represents each capability and the maximum number of agents' capacity values needed to perform that capability under that node (i.e. subtree, i.e. branch). "ubc" stands for "upper bound capability".
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
    Calculate the upper bound of the reward (utility) at each descendant of the queried node in the AND-OR goal tree.
    """

    descendant_nodes = list(traverse_tree(children_info, root_node_id=query_nodeId))
    
    return { 
        node_id : upperbound_node(
            ubcv_info,
            capabilities,
            agents,
            nodes_constraints,
            query_nodeId=node_id
        )
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


def upperbound_nodes_min(
        nodes_upper_bound : dict[int, float],
        nodes_upper_bound_min : dict[int, float],
        selected_nodes : list[int]
    ):
    """
    Calculate the upper bound of the reward (utility) of the system, where the system is composed of the selected nodes in the AND-OR goal tree (AND of selected nodes).
    """    
    up_1 = sum(nodes_upper_bound[n_id] for n_id in selected_nodes)
    up_2 = sum(nodes_upper_bound_min[n_id] for n_id in selected_nodes)
    return min(up_1, up_2)



def upperbound_v1(capabilities, tasks, agents):
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


def upperbound_v2(capabilities, tasks, agents, constraints):
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


def upperbound_v2_subset(capabilities, tasks, agents, constraints, task_selected):
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
        a_cap_tasks = [[j for j in a_taskInd if j != task_num and c in tasks[j] and task_selected[j]] for a_taskInd in a_taskInds] 

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



def upper_bound_subsytem(
        selected_nodes : list[int],
        nodes_upper_bound : dict[int, float],
        nodes_upper_bound_min : dict[int, float],
        node_type_info : dict[int, NodeType],
        leaf2task : dict[int, int],
        capabilities : list[int], 
        tasks : list[list[int]],
        agents : list[dict[int, float]],
        constraints,
    ):
    if all(node_type_info[node_id] == NodeType.LEAF for node_id in selected_nodes):
        task_selected = { j : False for j in range(len(tasks)) }
        for node_id in selected_nodes:
            task_selected[leaf2task[node_id]] = True
        return upperbound_v2_subset(capabilities, tasks, agents, constraints, task_selected)
    else:
        return upperbound_nodes_min(nodes_upper_bound, nodes_upper_bound_min, selected_nodes)