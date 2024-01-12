from .tree_types import NodeType, Node
from .rewards import task_reward

class GreedyNE:
    def __init__(self, agents, tasks, constraints, tree_info : list[Node], root_node_index=-1, gamma=1, coalition_structure=[], greedy_level=2):
        self.agents : list[list[float]] = agents
        self.tasks : list[list[float]] = tasks
        self.constraints = constraints
        self.tree_info = tree_info
        self.root_node_index = root_node_index
        self.gamma = gamma
        self.greedy_level = greedy_level
        self.dummy_task_id = len(tasks)

        self.task_num = len(tasks)
        self.agent_num = len(agents)
        self.agentIds = list(range(0, self.agent_num))
        self.a_taskInds : list[list[int]] = constraints[0]

        self.realtime_node_values = [0 for node in range(0, len(tree_info))]
        self.temp_node_values = {i : {node_id : node_value for node_id, node_value in enumerate(self.realtime_node_values)} for i in self.agentIds}

        if self.root_node_index < 0:
            self.root_node_index = len(tree_info) + self.root_node_index

        if coalition_structure == None or coalition_structure == []:
            self.coalition_structure = [[] for j in range(0, self.task_num)] + [list(range(0, self.agent_num))]
            self.allocation_structure = [self.task_num for i in range(0, self.agent_num)]
        else:
            self.coalition_structure = coalition_structure
            for j in range(0, self.task_num + 1):
                for i in coalition_structure[j]:
                    self.allocation_structure[i] = j

        self.cur_con = [
            self.get_agent_contribution(i, j)
            for i, j in enumerate(self.allocation_structure)
        ]

        self.task_cons = [
            [
                self.get_agent_contribution(i, j)
                if j in self.a_taskInds[i]
                else float("-inf")
                for j in range(0, self.task_num)
            ] + [0]
            for i in range(0, self.agent_num)
        ]

    def get_agent_contribution(self, query_agentIndex, query_taskIndex):
        """
        Return contribution of agent i to task j in coalition C_j
        
        = U_i(C_j, j) - U_i(C_j \ {i}, j) if i in C_j

        = U_i(C_j U {i}, j) - U_i(S, j) if i not in C_j
        """
        if query_taskIndex == self.dummy_task_id:
            return 0
        if query_taskIndex not in self.a_taskInds[query_agentIndex]:
            return 0
        cur_reward = task_reward(self.tasks[query_taskIndex], [self.agents[i] for i in self.coalition_structure[query_taskIndex]], self.gamma)
        coalition = self.coalition_structure[query_taskIndex]
        if self.allocation_structure[query_agentIndex] == query_taskIndex:
            return cur_reward - task_reward(self.tasks[query_taskIndex], [self.agents[i] for i in coalition if i != query_agentIndex], self.gamma)
        else:
            return task_reward(self.tasks[query_taskIndex], [self.agents[i] for i in coalition] + [self.agents[query_agentIndex]], self.gamma) - cur_reward
        
    def calc_temp_node_values(self, query_a_index, parent_info, children_info, root_node_index=-1):
        """
        Calculate temp_node_values[i], for when agent i is removed from the system
        """
        temp_node_values_i = {}
        
        node_id : int = self.allocation_structure[query_a_index]

        sys_lost_value = self.cur_con[query_a_index]

        temp_node_values_i[node_id] = self.realtime_node_values[node_id] - sys_lost_value
        
        prev_node_value = temp_node_values_i[node_id]
        prev_node_id = node_id
        node_id = parent_info[node_id]

        while sys_lost_value > 0:

            if self.tree_info[node_id].node_type == NodeType.AND:
                temp_node_values_i[node_id] = self.realtime_node_values[node_id] - sys_lost_value
            elif self.tree_info[node_id].node_type == NodeType.OR:
                max_node_value = max([self.realtime_node_values[j] for j in children_info[node_id] if j != prev_node_id] + [prev_node_value])
                temp_node_values_i[node_id] = max_node_value
                sys_lost_value = self.realtime_node_values[node_id] - max_node_value

            if node_id == root_node_index:
                break

            prev_node_value = temp_node_values_i[node_id]
            prev_node_id = node_id
            node_id = parent_info[node_id]

        return temp_node_values_i
    
    
    def update_coalitions(self, selected_a_index, old_t_index, new_t_index):
        """
        Move agent i to coalition j
        """

        # perform move
        self.allocation_structure[selected_a_index] = new_t_index
        self.coalition_structure[new_t_index].append(selected_a_index)
        self.coalition_structure[old_t_index].remove(selected_a_index)
    
    def update_tasks_contribution_values(self, selected_a_index, old_t_index, new_t_index):
        """
        Update the contribution values of agent i to each task
        """

        re_assigned = 0

        # update agents in the new coalition
        affected_t_indexes = []
        if new_t_index != self.dummy_task_id:
            affected_t_indexes.append(new_t_index)
            for i in self.coalition_structure[new_t_index]:
                self.task_cons[i][new_t_index] = self.get_agent_contribution(i, new_t_index)
                self.cur_con[i] = self.task_cons[i][new_t_index]
        else:
            self.task_cons[selected_a_index][new_t_index] = 0
            self.cur_con[selected_a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != self.dummy_task_id):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assigned = 1
            affected_t_indexes.append(old_t_index)
            for i in self.coalition_structure[old_t_index]:
                self.task_cons[i][old_t_index] = self.get_agent_contribution(i, old_t_index)
                self.cur_con[i] = self.task_cons[i][old_t_index]

        # update other agents with respect to the affected tasks
        for t_ind in affected_t_indexes:
            for i in self.agentIds:
                if (t_ind in self.a_taskInds[i]) and (i not in self.coalition_structure[t_ind]):
                    self.task_cons[i][t_ind] = self.get_agent_contribution(i, t_ind)

        return re_assigned


    def get_temp_node_value(self, query_node_id, temp_node_values : dict):
        """
        Return the temp_node_value of agent i at node j
        """
        return temp_node_values.get(query_node_id, self.realtime_node_values[query_node_id]) 


    def get_best_move(self, temp_node_values : dict[int, dict[int]], root_node_index=-1):
        """
        Calculate the best move values among all agents
        """
        best_sys_move_value, best_move_value, best_move_coalition, best_move_agent, best_move_node_revalue_sequence = float("-inf"), float("-inf"), self.allocation_structure[0], 0, {}
        
        for i in self.agentIds:
            # Movement value for moving agent from current coalition to dummy coalition (removing the agent from the system):
            sys_exit_value = temp_node_values[i].get(root_node_index, self.realtime_node_values[root_node_index]) - self.realtime_node_values[root_node_index]
            
            # Calculate the best move values for each agent
            for j in self.a_taskInds[i]:

                if j == self.allocation_structure[i]:
                    continue
            
                sys_added_value = self.task_cons[i][j]
                node_revalue_sequence = {}

                node_revalue_sequence[j] = temp_node_values[i].get(j, self.realtime_node_values[j]) + sys_added_value
                prev_node_id = j
                node_id = self.tree_info[j].parent_id

                # Backtrack to the root node
                while (sys_added_value + sys_exit_value) >= best_sys_move_value and sys_added_value > 0 and prev_node_id != root_node_index:

                    prev_node_val = node_revalue_sequence[prev_node_id]
                    node_val = temp_node_values[node_id].get(node_id, self.realtime_node_values[node_id])

                    if(self.tree_info[node_id].node_type == NodeType.AND):
                        node_revalue_sequence[node_id] = node_val + sys_added_value
                    
                    elif(self.tree_info[node_id].node_type == NodeType.OR):
                        node_revalue_sequence[node_id] = max(node_val, prev_node_val)
                        sys_added_value = node_revalue_sequence[node_id] - node_val

                    prev_node_id = node_id
                    node_id = self.tree_info[node_id].parent_id
                
                # Compare the calculated system move value with the best move value

                sys_move_val_i_j = sys_exit_value + sys_added_value
                task_move_val_i_j = self.task_cons[i][j] - self.cur_con[i]

                if(sys_move_val_i_j < best_sys_move_value):
                    continue

                if sys_move_val_i_j > best_sys_move_value:
                    best_sys_move_value, best_move_value, best_move_coalition, best_move_agent, best_move_node_revalue_sequence = sys_move_val_i_j, task_move_val_i_j, j, i, node_revalue_sequence
                    continue     
                
                # else: sys_move_val_i_j == best_move[0]:
                
                # Tie breaking: choose the one with higher contribution to a single task
                if task_move_val_i_j > best_move_value:
                    best_sys_move_value, best_move_value, best_move_coalition, best_move_agent, best_move_node_revalue_sequence = sys_move_val_i_j, task_move_val_i_j, j, i, node_revalue_sequence
                    continue
                
                # Tie breaking: choose the one that moves out of the dummy coalition
                if task_move_val_i_j == best_move_value and self.allocation_structure[i] == self.dummy_task_id:
                    best_sys_move_value, best_move_value, best_move_coalition, best_move_agent, best_move_node_revalue_sequence = sys_move_val_i_j, task_move_val_i_j, j, i, node_revalue_sequence
                    continue
            
        return best_sys_move_value, best_move_value, best_move_coalition, best_move_agent, best_move_node_revalue_sequence


    def solve(self):

        parent_info = {node.node_id: node.parent_id for node in self.tree_info}
        children_info = {node.node_id: node.children_ids.copy() for node in self.tree_info}

        temp_node_values = {i : self.calc_temp_node_values(i, parent_info, children_info, root_node_index=self.root_node_index) for i in self.agentIds}

        iteration_count = 0
        re_assignment_count = 0
        while True:
            iteration_count += 1

            sys_move_val, task_move_val, new_t_id, selected_a_id, node_revalue_sequence = self.get_best_move(temp_node_values, root_node_index=self.root_node_index)

            if sys_move_val < 0:
                break
            elif sys_move_val == 0:
                if task_move_val < 0:
                    break
                elif task_move_val == 0:
                    if self.allocation_structure[selected_a_id] != self.dummy_task_id:
                        break

            old_t_id = self.allocation_structure[selected_a_id]

            self.update_coalitions(selected_a_id, old_t_id, new_t_id)

            re_assigned = self.update_tasks_contribution_values(selected_a_id, old_t_id, new_t_id)
            
            re_assignment_count += re_assigned

            # Update real-time node values

            for node_id, node_value in temp_node_values[selected_a_id].items():
                self.realtime_node_values[node_id] = node_value

            for node_id, node_value in node_revalue_sequence.items():
                self.realtime_node_values[node_id] = node_value
            
            # For each agent, recalculate temp_node_values and the max move value
            
            temp_node_values = {i : self.calc_temp_node_values(i, parent_info, children_info, root_node_index=self.root_node_index) for i in self.agentIds}

            for i in self.agentIds:
                if (i != selected_a_id):
                    # We can skip calculating the temp_node_values of the selected agent, since it's just been updated
                    temp_node_values[i] = self.calc_temp_node_values(i, parent_info, children_info, root_node_index=self.root_node_index)

        return self.coalition_structure, self.realtime_node_values[self.root_node_index], iteration_count, re_assignment_count
