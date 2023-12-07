# %%
import numpy as np
import itertools
import time
import math
from functools import reduce # python3 compatibility
from operator import mul
import json
import os

# %%
def gen_tasks(task_num, max_capNum, capabilities): # n is the number of task, max_capNum is the maximum number of cap a task could require
    return [sorted(np.random.choice(capabilities,np.random.randint(3,max_capNum+1),replace=False)) for j in range(0, task_num)]

# %%
def gen_constraints(agent_num, task_num,power = 1, a_min_edge = 2,t_max_edge =5): #power is the inforce you put in the probabilities
    
    # the maximum tasks an agent could work on depends on the number of tasks available (e.g, if |T| = 1/2|A|, the roughly each agent can work on two tasks)
    # calculate the max and min edges for agents
    seats = math.floor(t_max_edge*task_num)
    a_taskInds = [[] for i in range(0,agent_num)]
    t_counter = [0 for j in range(0,task_num)] # each indicate the current number of agents on the task
    
    ## generating the number of tasks the agents could work on.
    a_taskNums = []
    for i in range(0,agent_num):
        a_max_edge = min((seats - (agent_num -1 -i)*a_min_edge),t_max_edge,task_num)
        a_min_edge = min(a_min_edge,a_max_edge)
        a_taskNums.append(np.random.randint(a_min_edge,a_max_edge+1)) #  indicate the number of task the agent could work on
        seats -= a_taskNums[i]
        
    t_indexes = [j for j in range(0,task_num) if t_counter[j] < t_max_edge] # make sure no further draw for those reached the maximum limit.
    for i in range(0,agent_num):
        if any([tc ==0 for tc in t_counter]):
            t_prob = [(math.e**(t_max_edge-t_counter[j]))**power for j in t_indexes] # power is used to manify the probability
            sum_prob = sum(t_prob)
            t_prop_2 = [prop/sum_prob for prop in t_prob]

            # draw tasks accounting to their current allocations 
            a_taskInds[i] = list(np.random.choice(t_indexes,min(a_taskNums[i],len(t_indexes)),replace=False,p = [prop/sum_prob for prop in t_prob])) 
            # increase the chosen task counters
        else:
            a_taskInds[i] = list(np.random.choice(t_indexes,min(a_taskNums[i],len(t_indexes)),replace=False)) 
                
        for j in a_taskInds[i]:
            t_counter[j] +=1
        t_indexes = [j for j in range(0,task_num) if t_counter[j] < t_max_edge] # make sure no further draw for those reached the maximum limit.




    # get also the list of agents for each task   
    t_agents = [[i for i in range(0,agent_num) if j in a_taskInds[i]] for j in range(0,task_num)]
    
        
    return a_taskInds, t_agents

# %%
def gen_agents(a_taskInds, tasks, max_capNum, capabilities, max_capVal): # m is the number of task, max_capNum is the maximum number of cap a task could require, max_capVal is the maximum capability value  
    
    agent_num = len(a_taskInds)
    caps_lists = []
    contri_lists = []
    for i in range(0,agent_num):
        t_caps = [tasks[j] for j in a_taskInds[i]] # lists of caps that each task agent could perform
        
        caps_union = set(itertools.chain(*t_caps)) # union of caps of tasks that agent could perform
        a_cap_num = np.random.randint(min(3,max_capNum,len(caps_union)),min(len(caps_union),max_capNum)+1) # the num of caps the agent will have
        
        a_caps = set([np.random.choice(t_c) for t_c in t_caps]) # initial draw to guarantee the agent has some contribution to each of the task that he could do
        
        rest_choices = list(caps_union.difference(a_caps))
        if rest_choices != []:
            update_len = max(0, a_cap_num - len(a_taskInds[i]))
            a_caps.update(np.random.choice(rest_choices, min(len(rest_choices), update_len), replace=False))
#             a_caps.update(np.random.choice(rest_choices, min(0,len(rest_choices),a_cap_num-len(a_taskInds[i])),replace = False))
        
        caps_lists.append(sorted(list(a_caps)))
        
        contri_lists.append([(np.random.randint(1,max_capVal+1) if c in caps_lists[i] else 0)  for c in range(0,len(capabilities))])
    
    return caps_lists, contri_lists

# %%
def upperBound(capabilities,tasks,agents):
    cap_ranks = [sorted([a[c] for a in agents],reverse=True) for c in capabilities]
    cap_req_all = list(itertools.chain(*tasks))
    cap_req_num = [cap_req_all.count(c) for c in capabilities]
    return sum([sum(cap_ranks[c][:cap_req_num[c]]) for c in capabilities])

# %%
def upperBound_ver2(capabilities,tasks,agents,constraints):
    agent_num = len(agents)
    task_num = len(tasks)
    a_taskInds = constraints[0]
    cap_req_all = list(itertools.chain(*tasks))
    cap_req_num = [cap_req_all.count(c) for c in capabilities]
    
    sys_rewards = 0
    for c in capabilities:
        a_cap_vals = [agents[i][c] for i in range(0,agent_num)]
        a_cap_tasks = [[j for j in a_taskInds[i] if j!=task_num and c in tasks[j]] for i in range(0,agent_num)]

        cap_rank_pos = np.argsort(a_cap_vals)

        a_cap_vals_ordered = [0 for i in range(0,agent_num)]
        a_cap_tasks_ordered = [[] for i in range(0,agent_num)]
        for p in range(0, len(cap_rank_pos)):
            a_cap_vals_ordered[p] = a_cap_vals[cap_rank_pos[p]]
            a_cap_tasks_ordered[p] = a_cap_tasks[cap_rank_pos[p]]
        
        cap_rewards = a_cap_vals_ordered[agent_num-1]
        cap_tasks = set(a_cap_tasks_ordered[agent_num-1])
        a_cap_num = 1
        for a_iter in range(agent_num-1,0,-1):
            cap_tasks = cap_tasks.union(set(a_cap_tasks_ordered[a_iter-1]))
            if len(cap_tasks) > a_cap_num:
                cap_rewards += a_cap_vals_ordered[a_iter-1]
                a_cap_num +=1
            if a_cap_num >= cap_req_num[c]: # if they got enough agents to contribute the number of required cap c
                break
        sys_rewards += cap_rewards
    return sys_rewards
        

# %%
def task_reward(task,agents,gamma = 1): # task is represented by a list of capabilities it requires, agents is a list agents, where each represented by a list cap contribution values 
        if agents == []: 
            return 0
        else:
            return sum([max([agent[cap] for agent in agents]) for cap in task])*(gamma**len(agents))

# %%
def sys_reward_agents(agents,tasks, alloc,gamma = 1): #alloc is a vector of size M each element indicate which task the agent is alloated to
    return sum([task_reward(tasks[j],[agents[i] for i in range(0,len(agents)) if alloc[i] == j],gamma) for j in range(0,len(tasks))])

def sys_rewards_tasks(tasks, agents, CS,gamma = 1): # CS is the coaltion structure
    return sum([task_reward(tasks[j],[agents[i] for i in CS[j]],gamma) for j in range(0,len(tasks))])

# %%
def agent_con(agents, tasks, query_agentIndex, query_taskIndex, cur_task_alloc,constraints,gamma = 1):
    a_taskInds = constraints[0]
    if query_taskIndex == len(tasks):
        return 0
    if query_taskIndex not in a_taskInds[query_agentIndex]:
        return 0
    cur_reward = task_reward(tasks[query_taskIndex],[agents[i] for i in cur_task_alloc],gamma)
    if query_agentIndex in cur_task_alloc:
        agents_list = [agents[i] for i in cur_task_alloc if i != query_agentIndex]
        return cur_reward - task_reward(tasks[query_taskIndex],agents_list,gamma)
    else:
        agents_list = [agents[i] for i in cur_task_alloc]
        agents_list.append(agents[query_agentIndex])
        return task_reward(tasks[query_taskIndex],agents_list,gamma)- cur_reward

# %%
def eGreedy2(agents,tasks,constraints,eps = 0,gamma = 1,CS=[]):
    re_assign = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)
    alloc = [task_num for i in range(0,agent_num)] # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if CS ==[]:

        CS = [[] for j in range(0,task_num+1)] # current coalition structure, the last one is dummy coalition
        cur_con = [0 for i in range(0,agent_num)]
    else:
        CS.append([])
        for j in range(0,task_num):
            for i in CS[j]:
                alloc[i] = j
        cur_con = [agent_con(agents,tasks,i,alloc[i],CS[alloc[i]],constraints,gamma) for i in range(0,agent_num)]

    task_cons = [[agent_con(agents,tasks,i,j,CS[j],constraints,gamma) 
                  if j in a_taskInds[i] else -1000 for j in range(0,task_num)]+[0] for i in range(0,agent_num)] 
    # the last 0 indicate not allocated
    
    move_vals = [[task_cons[i][j] - cur_con[i] 
                  if j in a_taskInds[i]+[task_num] else -1000 for j in range(0,task_num+1)] for i in range(0,agent_num)]

    
    max_moveIndexs = [np.argmax([move_vals[i][j] for j in a_taskInds[i]]+[0]) for i in range(0,agent_num)]
    
    max_moveVals = [move_vals[i][a_taskInds[i][max_moveIndexs[i]]] if max_moveIndexs[i] < len(a_taskInds[i]) 
                    else move_vals[i][task_num]
                                 for i in range(0,agent_num)]

    
    iteration = 0
    while (True):
        iteration += 1
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        feasible_choices = [i for i in range(0,agent_num) if max_moveVals[i]>0]
        if feasible_choices ==[]:
            break # reach NE solution
        if np.random.uniform() <= eps:
        # exploration: random allocation 
            a_index = np.random.choice(feasible_choices)
            t_index  = a_taskInds[a_index][max_moveIndexs[a_index]] if max_moveIndexs[a_index]< len(a_taskInds[a_index]) else task_num

            
        else:
            # exploitation: allocationelse based on reputation or efficiency
            a_index = np.argmax(max_moveVals)
            t_index  = a_taskInds[a_index][max_moveIndexs[a_index]] if max_moveIndexs[a_index]< len(a_taskInds[a_index]) else task_num


        # perfom move
        old_t_index = alloc[a_index] 
        alloc[a_index] = t_index
        CS[t_index].append(a_index)

        affected_a_indexes = []
        affected_t_indexes = []
        # update agents in the new coalition
        if t_index != task_num:
            affected_a_indexes.extend(CS[t_index])
            affected_t_indexes.append(t_index)
            
            #task_cons[i][t_index] 
            for i in CS[t_index]: 
                task_cons[i][t_index] = agent_con(agents,tasks, i,t_index, CS[t_index],constraints,gamma)
                cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0
        
        # update agent in the old coalition (if applicable)
        if old_t_index != task_num: # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assign += 1
            CS[old_t_index].remove(a_index)
            affected_a_indexes.extend(CS[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in CS[old_t_index]: 
                task_cons[i][old_t_index] = agent_con(agents,tasks, i, old_t_index,CS[old_t_index],constraints,gamma)

                cur_con[i] = task_cons[i][old_t_index]
   
        for i in affected_a_indexes:
            move_vals[i] = [task_cons[i][j] - cur_con[i] 
                  if j in a_taskInds[i]+[task_num] else -1000 for j in range(0,task_num+1)]
            

        ## updating other agents r.w.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0,agent_num):
                if (i not in CS[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_con(agents,tasks, i,t_ind,CS[t_ind],constraints,gamma) 
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

 
        max_moveIndexs = [np.argmax([move_vals[i][j] for j in a_taskInds[i]+[task_num]]) for i in range(0,agent_num)]
        max_moveVals = [move_vals[i][a_taskInds[i][max_moveIndexs[i]]] if max_moveIndexs[i] < len(a_taskInds[i]) 
                    else move_vals[i][task_num]
                                 for i in range(0,agent_num)]
        


    return CS, sys_rewards_tasks(tasks,agents, CS,gamma), iteration,re_assign

# %%
def resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma = 1):
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    
    a_msg_sum = [{d_key:sum([r_msgs[j][i][0] for j in a_taskInds[i] if j!=d_key]) 
                  + r_msgs[d_key][i][1] for d_key in a_taskInds[i]} 
                 for i in range(0,agent_num)]
    

    alloc = [max(ams, key = ams.get) if ams !={} else task_num for ams in a_msg_sum] 
    
    
    return alloc, sys_reward_agents(agents,tasks, alloc, gamma),iteration,iter_over,converge

# %%
def FMS(agents,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.time()
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
        if time.time() - start_time >= time_bound:
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
                if time.time() - start_time >= time_bound:
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
            
#                 alphas = -sum(msgs.values())/len(msgs.keys())            
#                 msgs_regularised = {d_key:msgs[d_key] + alphas for d_key in msgs.keys()}               
#                 old_msg = q_msgs[i][t_key]
#                 if old_msg!={} and any([abs(msgs_regularised[d_key] - old_msg[d_key]) > 10**(-5) 
#                                         for d_key in old_msg.keys()]):
#                     flag = False 
#                 q_msgs[i][t_key] = msgs_regularised         


                old_msg = q_msgs[i][t_key]
                if old_msg!={} and any([abs(msgs[d_key] - old_msg[d_key]) > 10**(-5) 
                                        for d_key in old_msg.keys()]):
                    flag = False                
                q_msgs[i][t_key] = msgs
                
            if flag: # agent i sending the same info
                q_flags[i] = True
        
        if time.time() - start_time >= time_bound:
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
                if time.time() - start_time >= time_bound:
                    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
                com_rewards.append(
                    task_reward(tasks[j],[agents[a_key] for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
                )
                
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
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
           

# %%
def FMSNormalised(agents,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.time()
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
        if time.time() - start_time >= time_bound:
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
                if time.time() - start_time >= time_bound:
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
        
        if time.time() - start_time >= time_bound:
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
                if time.time() - start_time >= time_bound:
                    return resultCal(agents,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
                com_rewards.append(
                    task_reward(tasks[j],[agents[a_key] for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
                )
                
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
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
           

# %%
def OPD(agents,tasks,constraints,gamma):
    edge_pruned = 0  
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = [list(con) for con in constraints[0]]
    t_agentInds = [list(con) for con in constraints[1]]

    a_ubs = [[0 for j in a_taskInds[i]] for i in range(0,agent_num)]
    a_lbs = [[0 for j in a_taskInds[i]] for i in range(0,agent_num)]
    
    for j in range(0,task_num):
        
        linked_agentInds = t_agentInds[j]
        com_dict = []
        com_rewards = []
        for c in itertools.product(*[[0,1] for i in linked_agentInds]):
            com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
            com_rewards.append(
                task_reward(tasks[j],[agents[a_key] for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
            )
        
        for i in t_agentInds[j]:
            t_ind = a_taskInds[i].index(j)
            a_ind = t_agentInds[j].index(i)
            reward_t = [com_rewards[c] 
                      for c in range(0,len(com_dict)) if com_dict[c][i]==1]
            reward_f = [com_rewards[c] 
                      for c in range(0,len(com_dict)) if com_dict[c][i]==0]
            cons_j = [reward_t[k] - reward_f[k] for k in range(0,len(reward_t))]
            a_lbs[i][t_ind] =min(cons_j) 
            a_ubs[i][t_ind] = max(cons_j)
    

    
    
    for i in range(0, agent_num):
        t_flag = [True for j in a_taskInds[i]]
        for t_ind in range(0,len(a_taskInds[i])): 
            for t2_ind in range(0,len(a_taskInds[i])): 
                if t_ind != t2_ind and a_ubs[i][t_ind] < a_lbs[i][t2_ind]:
                    t_flag[t_ind] = False
                    edge_pruned += 1
                    break
                
        for t_ind in range(0,len(t_flag)):
            if not t_flag[t_ind]:
                t_agentInds[a_taskInds[i][t_ind]].remove(i)
        
        new_a_taskInds = [a_taskInds[i][t_ind] 
                          for t_ind in range(0,len(a_taskInds[i])) if t_flag[t_ind]]
        a_taskInds[i] = new_a_taskInds
        
    return a_taskInds,t_agentInds,edge_pruned

# %%
def random(agents, tasks,constraints, gamma = 1):
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = constraints[0]
    alloc = [np.random.choice(a_taskInds[i]+[task_num]) for i in range(0,agent_num)]
    return alloc, sys_reward_agents(agents,tasks, alloc, gamma)

# %%
def alloc_to_CS(tasks,alloc):
    task_num = len(tasks)
    CS = [[] for j in range(0, len(tasks))]
    for i in range(0,len(alloc)):
        if alloc[i]< task_num: # means allocated (!=task_num)
            CS[alloc[i]].append(i)
    return CS

# %%
def append_record(record, filename, typ):
    with open(filename, 'a') as f:
        if typ != '':
            json.dump(record, f, default = typ)
        else:
            json.dump(record, f)
        f.write(os.linesep)
        f.close()

# %%
min_t_num = 100
max_t_num = 1000
run_num = 100
gamma = 1
a_min_edge = 1
t_max_edge = 5
capNum = 10
max_capVal = 10
max_capNum_task = 10
max_capNum_agent = 10
ex_identifier=0
time_bound  = 600


    
capabilities = list(range(0,capNum))    

for run in range(0,run_num):
    task_num = min_t_num
    while task_num<= max_t_num:
        ex_identifier +=1

#         agent_num = np.random.randint(task_num,3*task_num)
        agent_num = 2*task_num
        tasks = gen_tasks(task_num, max_capNum_task, capabilities)
        constraints = gen_constraints(agent_num,task_num,1, a_min_edge,t_max_edge)
        a_taskInds = constraints[0]
        agents_cap, agents = gen_agents(a_taskInds,tasks, max_capNum_agent, capabilities, max_capVal)
        # num_com = np.prod([1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])

        num_com = reduce(mul, [1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])

        print('ex_identifier',ex_identifier,'task_num:',task_num,'  agent_num:',
              agent_num)

        up = upperBound(capabilities,tasks,agents)

        up2= upperBound_ver2(capabilities,tasks,agents,constraints)
        print("UP:", up,"  UP2:", up2)

        result = {"ex_identifier":ex_identifier,"task_num":task_num,"agent_num":agent_num,"num_com":num_com,"up":up,"up2":up2}
        data = {"ex_identifier":ex_identifier,"tasks":tasks,"constraints":constraints,"agents_cap":agents_cap,"agents":agents}

        start= time.time()
        r = eGreedy2(agents,tasks,constraints,gamma = gamma)
        end = time.time()
        result['g'] = r[1] 
        result['g_iter'] = r[2]
        result['g_reass'] = r[3]
        result['g_t'] = end-start
        print("eGreedy time:", result['g_t'],'result:',result['g'], 
              "iteration:",result['g_iter'],' re-assignment',result['g_reass'])

        start= time.time()
        r= random(agents, tasks,constraints, gamma )
        end = time.time()
        alloc = r[0]
        result['rand'] = r[1]
        result['rand_t'] = end-start
        print("rand time:", result['rand_t'],'result:',result['rand'])
        
        start= time.time()
        r = eGreedy2(agents,tasks,constraints,CS = alloc_to_CS(tasks,alloc))
        end = time.time()      
        result['rand_g'] = r[1] 
        result['rand_g_iter'] = r[2]
        result['rand_g_reass'] = r[3]
        result['rand_g_t'] = end-start
        print("rand Greedy time:", result['rand_g_t'],'result:',result['rand_g'], 
             "iteration:",result['g_iter'],' re-assignment',result['rand_g_reass'])
        # do not show result when task number larger than 400        
        if task_num <= 400:        
                start= time.time()
                opd = OPD(agents,tasks,constraints,gamma)
                new_con = opd[0:2]
                result['OPD_t'] = time.time() - start
                result['OPD_pruned'] = opd[2]
                print("OPD time used:",result['OPD_t']," Edge pruned:",result['OPD_pruned'])
                r = FMS(agents,tasks,new_con,gamma = gamma,time_bound = time_bound)
                end = time.time()
                result['BnBFMS'] = r[1] 
                result['BnBFMS_iter'] = r[2]
                result['BnBFMS_t'] = end-start
                result['BnBFMS_converge'] = r[4]
                print("BnB FMS time:", result['BnBFMS_t'],'result:',result['BnBFMS'], 
                        "iteration:",result['BnBFMS_iter'],"converge?",result['BnBFMS_converge'])

                print()

        #append data and result 
        files = {'result_cap':[result,'']}
        for filename in list(files.keys()):
            append_record(files[filename][0],filename,typ = files[filename][1])

        #increase the task_num
        task_num += 100

# %%



