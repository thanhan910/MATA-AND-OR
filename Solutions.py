import itertools
import numpy as np
import time

from CalcRewards import sys_reward_agents, task_reward

def resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma=1):
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    a_msg_sum = [
        {
            j: sum([r_msgs[j][i][0] for j in a_taskInds[i] if j != j]) + r_msgs[j][i][1]
            for j in a_taskInds[i]
        }
        for i in range(0, agent_num)
    ]

    alloc = [max(ams, key=ams.get) if ams != {} else task_num for ams in a_msg_sum]

    return (
        alloc,
        sys_reward_agents(agents, tasks, alloc, gamma),
        iteration,
        iter_over,
        converge,
    )



def convert_alloc_to_CS(tasks, allocation_structure):
    task_num = len(tasks)
    coalition_structure = [[]] * task_num
    for i, j in enumerate(allocation_structure):
        if j < task_num:  # means allocated (!=task_num)
            coalition_structure[j].append(i)
    return coalition_structure





def FMS(agents, tasks, constraints, gamma, time_bound=500):
    '''
    Fast Max-Sum algorithm
    '''
    converge = False
    iter_over = False
    start_time = time.perf_counter()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    agent_num = len(agents)

    q_msgs = [{t_key: {} for t_key in a_taskInds[i]} for i in range(0, agent_num)]
    r_msgs = [
        {
            t_agentInds[j][i]: (
                {1: -100} if len(a_taskInds[t_agentInds[j][i]]) == 1
                else {key: -100 for key in [0, 1]}
            )
            for i in range(0, len(t_agentInds[j]))
        }
        for j in range(0, task_num)
    ]

    q_flags = [False] * agent_num
    r_flags = [False] * task_num

    iteration = 0
    while True:
        if time.perf_counter() - start_time >= time_bound:
            return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)
        iteration += 1

        if iteration > agent_num + task_num:
            iter_over = True
            return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)

        if all(q_flags) and all(r_flags):  # converge, msgs are all the same.
            converge = True
            # break
        for i in range(0, agent_num):
            linked_taskInds = a_taskInds[i]

            flag = True
            for t_key in linked_taskInds:
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)
                ####### check time bound
                msgs = {}

                if len(linked_taskInds) > 1:
                    msgs[1] = sum(
                        m[0]
                        for m in [
                            r_msgs[j][i] for j in linked_taskInds if j != t_key
                        ]
                    )
                    msg_0 = []
                    ts = list(linked_taskInds)
                    ts.remove(t_key)

                    for k in ts:
                        msg_0.append(
                            sum([m[0] for m in [r_msgs[j][i] for j in ts if j != k]])
                            + r_msgs[k][i][1]
                        )

                    msgs[0] = 0 if msg_0 == [] else max(msg_0)
                else:
                    msgs[1] = 0

                alphas = -sum(msgs.values()) / len(msgs.keys())

                msgs_regularised = {
                    d_key: msgs[d_key] + alphas for d_key in msgs.keys()
                }

                old_msg = q_msgs[i][t_key]
                if old_msg != {} and any(
                    abs(msgs_regularised[d_key] - old_msg[d_key]) > 10 ** (-5)
                    for d_key in old_msg.keys()
                ):
                    flag = False

                q_msgs[i][t_key] = msgs_regularised

            if flag:  # agent i sending the same info
                q_flags[i] = True

        if time.perf_counter() - start_time >= time_bound:
            break
        ###################### SAME thing, using comprehension
        #             msgs = {t_key:{d_key:sum([m[d_key] for m in [r_msgs[j][i] for j in linked_taskInds if j != t_key]])
        #                            for d_key in linked_taskInds}
        #                                 for t_key in linked_taskInds}
        #             alphas = {t_key:-sum(msgs[t_key].values())/len(msgs.keys())
        #                       for t_key in linked_taskInds}
        #             msgs_regularised = {t_key:{d_key:msgs[t_key][d_key] + alphas[t_key]
        #                            for d_key in linked_taskInds}
        #                                 for t_key in linked_taskInds}
        for j in range(0, task_num):
            linked_agentInds = t_agentInds[j]
            msg_con = [q_msgs[a][j] for a in linked_agentInds]

            com_dict = []
            com_rewards = []
            dom_com = [
                [0, 1] if len(a_taskInds[i]) > 1 else [1] for i in linked_agentInds
            ]

            for c in itertools.product(*dom_com):
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)
                ####### check time bound

                com_dict.append({linked_agentInds[i]: c[i] for i in range(0, len(c))})
                com_rewards.append(
                    task_reward(
                        tasks[j],
                        [agents[a_key] for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],
                        gamma,
                    )
                )

            flag = True
            for a_key in linked_agentInds:
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)
                ####### check time bound

                old_msg = r_msgs[j][a_key]
                q_table = []
                for c in range(0, len(com_dict)):
                    q_table.append(
                        sum([q_msgs[a][j][com_dict[c][a]] for a in linked_agentInds if a != a_key]) + com_rewards[c]
                    )

                r_msgs[j][a_key] = {
                    d_key: max(
                        [q_table[c] for c in range(0, len(com_dict)) if com_dict[c][a_key] == d_key]
                    )
                    for d_key in ([0, 1] if len(a_taskInds[a_key]) > 1 else [1])
                }

                if any( 
                    abs(r_msgs[j][a_key][d_key] - old_msg[d_key]) > 10 ** (-5)
                    for d_key in old_msg.keys()
                ):
                    flag = False

            if flag:  # task j sending the same info
                r_flags[j] = True

    return resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma)


def OPD(agents, tasks, constraints, gamma):
    edge_pruned = 0
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = [list(con) for con in constraints[0]]
    t_agentInds = [list(con) for con in constraints[1]]

    a_ubs = [[0] * len(a_taskInds[i]) for i in range(0, agent_num)]
    a_lbs = [[0] * len(a_taskInds[i]) for i in range(0, agent_num)]

    for j in range(0, task_num):
        linked_agentInds = t_agentInds[j]
        com_dict = []
        com_rewards = []
        for c in itertools.product(*[[0, 1] for i in linked_agentInds]):
            com_dict.append({linked_agentInds[i]: c[i] for i in range(0, len(c))})
            com_rewards.append(
                task_reward(
                    tasks[j],
                    [
                        agents[a_key]
                        for a_key in com_dict[-1].keys()
                        if com_dict[-1][a_key] == 1
                    ],
                    gamma,
                )
            )

        for i in t_agentInds[j]:
            t_ind = a_taskInds[i].index(j)
            a_ind = t_agentInds[j].index(i)
            reward_t = [
                com_rewards[c] for c in range(0, len(com_dict)) if com_dict[c][i] == 1
            ]
            reward_f = [
                com_rewards[c] for c in range(0, len(com_dict)) if com_dict[c][i] == 0
            ]
            cons_j = [reward_t[k] - reward_f[k] for k in range(0, len(reward_t))]
            a_lbs[i][t_ind] = min(cons_j)
            a_ubs[i][t_ind] = max(cons_j)

    for i in range(0, agent_num):
        t_flag = [True for j in a_taskInds[i]]
        for t_ind in range(0, len(a_taskInds[i])):
            for t2_ind in range(0, len(a_taskInds[i])):
                if t_ind != t2_ind and a_ubs[i][t_ind] < a_lbs[i][t2_ind]:
                    t_flag[t_ind] = False
                    edge_pruned += 1
                    break

        for t_ind in range(0, len(t_flag)):
            if not t_flag[t_ind]:
                t_agentInds[a_taskInds[i][t_ind]].remove(i)

        new_a_taskInds = [
            a_taskInds[i][t_ind]
            for t_ind in range(0, len(a_taskInds[i]))
            if t_flag[t_ind]
        ]
        a_taskInds[i] = new_a_taskInds

    return a_taskInds, t_agentInds, edge_pruned


def FMSNormalised(agents,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.perf_counter()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    agent_num =len(agents)
    
    q_msgs = [{t_key:{} for t_key in a_taskInds[i]} for i in range(0,agent_num)]
    r_msgs = [{t_agentInds[j][i]:({1:-100} if len(a_taskInds[t_agentInds[j][i]])==1 else {key:-100 for key in [0,1]})
               for i in range(0,len(t_agentInds[j]))}
              for j in range(0,task_num)]

    
    q_flags = [False for i in range(0,agent_num)]
    r_flags = [False for j in range(0,task_num)] 
    
    iteration = 0
    while True:
        if time.perf_counter() - start_time >= time_bound:
            return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        iteration += 1
        
        if iteration > agent_num +task_num:
            iter_over = True
            return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        
        if all(q_flags) and all(r_flags): #converge, msgs are all the same.
            converge = True
#             break
        for i in range(0,agent_num):
            linked_taskInds = a_taskInds[i]
            
            flag = True
            for t_key in linked_taskInds:
                
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                msgs = {}
                
                
                if len(linked_taskInds)>1:
                    msgs[1] = sum([m[0] for m in [r_msgs[j][i] for j in linked_taskInds if j != t_key]])
                    msg_0 = []
                    ts = list(linked_taskInds)
                    ts.remove(t_key)
                
                    for k in ts:
                        msg_0.append(sum([m[0] for m in [r_msgs[j][i] for j in ts if j != k]]) 
                                     + r_msgs[k][i][1])
                    
                    msgs[0]= (0 if msg_0 == [] else max(msg_0))
                else:
                    msgs[1] = 0
            
                alphas = -sum(msgs.values())/len(msgs.keys())
            
                msgs_regularised = {d_key:msgs[d_key] + alphas for d_key in msgs.keys()} 
                
                
                old_msg = q_msgs[i][t_key]
                if old_msg!={} and any([abs(msgs_regularised[d_key] - old_msg[d_key]) > 10**(-5) 
                                        for d_key in old_msg.keys()]):
                    flag = False
                
                q_msgs[i][t_key] = msgs_regularised         
            
            if flag: # agent i sending the same info
                q_flags[i] = True
        
        if time.perf_counter() - start_time >= time_bound:
            break
###################### SAME thing, using comprehension                
#             msgs = {t_key:{d_key:sum([m[d_key] for m in [r_msgs[j][i] for j in linked_taskInds if j != t_key]])
#                            for d_key in linked_taskInds} 
#                                 for t_key in linked_taskInds}            
#             alphas = {t_key:-sum(msgs[t_key].values())/len(msgs.keys()) 
#                       for t_key in linked_taskInds}            
#             msgs_regularised = {t_key:{d_key:msgs[t_key][d_key] + alphas[t_key]
#                            for d_key in linked_taskInds} 
#                                 for t_key in linked_taskInds}        
        for j in range(0, task_num):
            linked_agentInds = t_agentInds[j]            
            msg_con = [q_msgs[a][j] for a in linked_agentInds]
            
            com_dict = []
            com_rewards = []
            dom_com = [[0,1] if len(a_taskInds[i]) > 1 else [1] for i in linked_agentInds]

            for c in itertools.product(*dom_com):
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
                com_rewards.append(
                    task_reward(tasks[j],[agents[a_key] for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
                )
                
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                old_msg = r_msgs[j][a_key]
                q_table = []
                for c in range(0,len(com_dict)):
                    q_table.append(sum([q_msgs[a][j][com_dict[c][a]] for a in linked_agentInds if a != a_key]) 
                                          + com_rewards[c])
                
                r_msgs[j][a_key] = {d_key:max([q_table[c] for c in range(0,len(com_dict)) if com_dict[c][a_key] == d_key]) 
                                      for d_key in ([0,1] if len(a_taskInds[a_key])>1 else [1])}
                

                if any([abs(r_msgs[j][a_key][d_key] - old_msg[d_key]) > 10**(-5) for d_key in old_msg.keys()]):
                    flag = False
                    

            
            if flag: # task j sending the same info
                r_flags[j] = True
                
    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
           


def random(agents, tasks, constraints, gamma=1):
    '''
    Randomly allocate tasks to agents
    '''
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = constraints[0]
    alloc = [np.random.choice(a_taskInds[i] + [task_num]) for i in range(0, agent_num)]
    return alloc, sys_reward_agents(agents, tasks, alloc, gamma)