from .node_type import NodeType
from .GreedyNE import aGreedyNE_subset
from .upper_bound import upper_bound_subsytem
from .tree_utils import AO_star
from .rewards import task_reward, sys_rewards_tasks


def get_leaves(root_id: int, children_info: dict[int, list[int]]):
    leaves = []
    stack = [root_id]
    while len(stack) > 0:
        current_node = stack.pop()
        if current_node not in children_info or len(children_info[current_node]) == 0:
            leaves.append(current_node)
        else:
            stack += children_info[current_node]
    return leaves


def update_leaves_info(updated_node, leaves_info, children_info, parent_info, node_type_info):
    """
    updated_node: the node that has just been updated in leaves_info
    """
    current_node = updated_node
    while current_node != 0:
        parent_node = parent_info[current_node]
        if node_type_info[parent_node] == NodeType.AND:
            leaves_info[parent_node] = sum([leaves_info[child_id] for child_id in children_info[parent_node]], [])
        else:
            leaves_info[parent_node] = leaves_info[children_info[parent_node][0]].copy()
        current_node = parent_node

    return leaves_info



def AOGreedyNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        nodes_upper_bound: dict[int, float],
        nodes_upper_bound_min: dict[int, float],
        leaf2task: dict[int, int],
        capabilities: list[int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        eps=0, 
        gamma=1,
        root_node_id=0,
        # reward_function: dict[int, float],
    ):

    agent_num = len(agents)
    task_num = len(tasks)

    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition

    reward_function = {
        j: task_reward(tasks[j], [agents[i] for i in coalition_structure[j]], gamma)
        for j in range(0, task_num)
    }

    system_reward, current_tasks_solution = AO_star(children_info, node_type_info, reward_function, parent_info)

    coalition_structure, system_reward, iteration_count, re_assignment_count = aGreedyNE_subset(agents=agents, tasks=tasks, constraints=constraints, coalition_structure=coalition_structure, selected_tasks=current_tasks_solution, eps=eps, gamma=gamma)
    
    visited = {}
    # expanded = []
    solution_path_children_info : dict[int, list[int]] = {}
    solution_path_leaves_info : dict[int, list[int]] = {}

    def AOS_helper(node_id: int):

        if node_id not in visited:
            visited[node_id] = True
            solution_path_children_info[node_id] = []
            solution_path_leaves_info[node_id] = []

        node_type = node_type_info[node_id]

        best_solution = []
        
        total_reward = 0 if node_type == NodeType.AND else float('-inf')
        
        if node_type == NodeType.LEAF:
            # expanded.append(node_id)
            solution_path_leaves_info[node_id] = [node_id]
            # Update solution_path_leaves_info for all ancestors
            update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)

            current_tasks_solution = solution_path_leaves_info[root_node_id]

            best_solution = [node_id]

            # total_reward = rewards[node_id]

        elif node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                solution_path_children_info[node_id].append(child_id)

                child_reward, child_solution = AOS_helper(child_id)

                total_reward += child_reward
                best_solution += child_solution
                
        else:
            for child_id in children_info[node_id]:

                if solution_path_children_info[node_id] == []:
                    solution_path_children_info[node_id] = [child_id]
                
                child_reward, child_solution = AOS_helper(child_id)

                if child_reward > total_reward:
                    solution_path_children_info[node_id] = [child_id]
                    total_reward = child_reward
                    best_solution = child_solution
                    solution_path_leaves_info[node_id] = child_solution
                    # Update solution_path_leaves_info for all ancestors
                    update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)
                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    return AOS_helper(0)
