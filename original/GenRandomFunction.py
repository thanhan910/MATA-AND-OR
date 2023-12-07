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
def gen_tasks(task_num, agent_num, constraints,max_reward = 100): # task reward DB, for each task, each coaltion value is a random number between (0,100)
    t_agents = constraints[1]
    tasks = [] # tasks is a list of task reward dictinoary, key represents the agent allocation in binary sum, value is a random integer
    for j in range(0, task_num):
        coalitions = list(itertools.chain(*[itertools.combinations(t_agents[j],i+1) for i,_ in enumerate(t_agents[j])]))
        dict_list = [(sum([2**a for a in com]), np.random.randint(1,max_reward+1)) for com in coalitions]
        tasks.append({key: value for (key, value) in dict_list})
    return tasks

# %%
def task_reward(task,coalition,gamma = 1,prt = False): # task is a dict (reward DB), coalition is a list of agents (indexes)
    key = sum([2**int(coalition[i]) for i in range(0,len(coalition))])
    if key in task.keys():
        return task[key]
    else:
        return 0

# %%
def agent_con(task, query_agentIndex, cur_coalition,gamma = 1,prt=False): # cur_coalition is a list of agents (indexes)
    if query_agentIndex in cur_coalition:
        new_coalition = list(cur_coalition)
        new_coalition.remove(query_agentIndex)

        return task_reward(task,cur_coalition,gamma) - task_reward(task,new_coalition,gamma)
    else:
        return task_reward(task,cur_coalition+[query_agentIndex],gamma) - task_reward(task,cur_coalition,gamma)

# %%
def sys_rewards_tasks(tasks, CS,gamma = 1): # CS is a list of coalitions
    return sum([task_reward(tasks[j],CS[j],gamma) for j in range(0,len(tasks))])

# %%
def alloc_to_CS(tasks,alloc):
    task_num = len(tasks)
    CS = [[] for j in range(0, len(tasks))]
    for i in range(0,len(alloc)):
        if alloc[i]< task_num: # means allocated (!=task_num)
            CS[alloc[i]].append(i)
    return CS

# %%
def random(agent_num,tasks,constraints, gamma = 1):
    task_num = len(tasks)
    a_taskInds = constraints[0]
    alloc = [np.random.choice(a_taskInds[i]+[task_num]) for i in range(0,agent_num)]
    return alloc, sys_rewards_tasks(tasks, alloc_to_CS(tasks,alloc), gamma)

# %%
def resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma = 1):
    a_taskInds = constraints[0]
    task_num = len(tasks)

    
    a_msg_sum = [{d_key:sum([r_msgs[j][i][0] for j in a_taskInds[i] if j!=d_key]) 
                  + r_msgs[d_key][i][1] for d_key in a_taskInds[i]} 
                 for i in range(0,agent_num)]
    

    alloc = [max(ams, key = ams.get) if ams !={} else task_num for ams in a_msg_sum] 
    
    
    return alloc, sys_rewards_tasks(tasks, alloc_to_CS(tasks,alloc),gamma),iteration,iter_over,converge

# %%
def FMSNormalised(agent_num,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.time()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    
    q_msgs = [{t_key:{} for t_key in a_taskInds[i]} for i in range(0,agent_num)]
    r_msgs = [{t_agentInds[j][i]:({1:-100} if len(a_taskInds[t_agentInds[j][i]])==1 else {key:-100 for key in [0,1]})
               for i in range(0,len(t_agentInds[j]))}
              for j in range(0,task_num)]

    
    q_flags = [False for i in range(0,agent_num)]
    r_flags = [False for j in range(0,task_num)] 
    
    iteration = 0
    while True:
        if time.time() - start_time >= time_bound:
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        iteration += 1
        
        if iteration > agent_num +task_num:
            iter_over = True
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        
        if all(q_flags) and all(r_flags): #converge, msgs are all the same.
            converge = True
#             break
        for i in range(0,agent_num):
            linked_taskInds = a_taskInds[i]
            
            flag = True
            for t_key in linked_taskInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
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
#######################################################
#                 old_msg = q_msgs[i][t_key]
#                 if old_msg!={} and any([abs(msgs[d_key] - old_msg[d_key]) > 10**(-5) 
#                                         for d_key in old_msg.keys()]):
#                     flag = False                
#                 q_msgs[i][t_key] = msgs
########################################################

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
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
                com_rewards.append(
                    task_reward(tasks[j],[a_key for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
                )
                
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
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
                
    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
           

# %%
def FMS(agent_num,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.time()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    
    q_msgs = [{t_key:{} for t_key in a_taskInds[i]} for i in range(0,agent_num)]
    r_msgs = [{t_agentInds[j][i]:({1:-100} if len(a_taskInds[t_agentInds[j][i]])==1 else {key:-100 for key in [0,1]})
               for i in range(0,len(t_agentInds[j]))}
              for j in range(0,task_num)]

    
    q_flags = [False for i in range(0,agent_num)]
    r_flags = [False for j in range(0,task_num)] 
    
    iteration = 0
    while True:
        if time.time() - start_time >= time_bound:
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        iteration += 1
        
        if iteration > agent_num +task_num:
            iter_over = True
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        
        if all(q_flags) and all(r_flags): #converge, msgs are all the same.
            converge = True
#             break
        for i in range(0,agent_num):
            linked_taskInds = a_taskInds[i]
            
            flag = True
            for t_key in linked_taskInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
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
########################################################
                old_msg = q_msgs[i][t_key]
                if old_msg!={} and any([abs(msgs[d_key] - old_msg[d_key]) > 10**(-5) 
                                        for d_key in old_msg.keys()]):
                    flag = False                
                q_msgs[i][t_key] = msgs
########################################################

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
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                com_dict.append({linked_agentInds[i]:c[i] for i in range(0,len(c))})
                com_rewards.append(
                    task_reward(tasks[j],[a_key for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
                )
                
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
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
                
    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
           

# %%
def f_bounds(task,linked_agentInds,assign,constraints):

    a_taskInds = constraints[0]
    a_remain = list(set(linked_agentInds) - set(assign.keys()))
    doms = [[0,1] if len(a_taskInds[i])>1 else [1] for i in a_remain]
    com = list(itertools.product(*doms))
    com_dict = [{a_remain[i]:c[i] for i in range(0,len(c))} for c in com]
    com_dict2 = [{**assign,**cd} for cd in com_dict]
    com_rewards = [task_reward(task,[a for a in com_dict2[c].keys() if com_dict2[c][a] == 1],gamma)
                   for c in range(0,len(com_dict2))]  
    return min(com_rewards),max(com_rewards)

# %%
def q_bounds(taskInd,linked_agentInds,assign,q_msgs,a_key):
    j = taskInd
    a_remain = list(set(linked_agentInds) - set(assign.keys()))
    base = sum([q_msgs[a][j][assign[a]] for a in assign.keys() if a != a_key]) 
    ub = sum([max([q_msgs[a][j][key] for key in q_msgs[a][j].keys()]) for a in a_remain])
    lb = sum([min([q_msgs[a][j][key] for key in q_msgs[a][j].keys()]) for a in a_remain])
    return base+lb, base+ub

# %%
def expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign,constraints):
    a_taskInds = constraints[0]
    j = taskInd

    if len(assign.keys())==len(linked_agentInds):
        result = sum([q_msgs[a][j][assign[a]] for a in assign.keys() if a != a_key]) + task_reward(tasks[j],[a for a in assign.keys() if assign[a] == 1],gamma)
        return result
    else:
        a_remain = list(set(linked_agentInds) - set(assign.keys()))
        i = a_remain[0]
        
        assign_t = assign.copy()
        assign_t[i] = 1
        
        if len(a_taskInds[i])<=1:
            return expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_t,constraints)
        else:
            ft_lb,ft_ub = f_bounds(tasks[j],linked_agentInds,assign_t,constraints)
            qt_lb, qt_ub = q_bounds(taskInd,linked_agentInds,assign_t,q_msgs,a_key)
            rt_ub = ft_ub +qt_ub
            rt_lb = ft_lb +qt_lb

            assign_f = assign.copy()
            assign_f[i] = 0
            ff_lb,ff_ub = f_bounds(tasks[j],linked_agentInds,assign_f,constraints)
            qf_lb, qf_ub = q_bounds(taskInd,linked_agentInds,assign_f,q_msgs,a_key)
            rf_ub = ff_ub +qf_ub
            rf_lb = ff_lb +qf_lb

            if rt_ub < rf_lb:
#                 print('prune true branch')
                return expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_f,constraints)
            elif rf_ub < rt_lb:
#                 print('prune false branch')
                return expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_t,constraints)
            else:
                rt = expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_t,constraints)
                rf = expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_f,constraints)
                return max(rt,rf)

            rt = expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_t,constraints)
            rf = expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign_f,constraints)
            return max(rt,rf)

# %%
def findMax(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,constraints):
    q_db = {}
    f_db = {}
    assign={}
    assign[a_key] = d_key
    return expandTree(tasks,taskInd,q_msgs,linked_agentInds,a_key,d_key,assign,constraints)

# %%
def BnBFMS(agent_num,tasks,constraints,gamma,time_bound = 500):
    converge = False
    iter_over = False
    start_time = time.time()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    
    q_msgs = [{t_key:{} for t_key in a_taskInds[i]} for i in range(0,agent_num)]
    r_msgs = [{t_agentInds[j][i]:({1:-100} if len(a_taskInds[t_agentInds[j][i]])==1 else {key:-100 for key in [0,1]})
               for i in range(0,len(t_agentInds[j]))}
              for j in range(0,task_num)]

    
    q_flags = [False for i in range(0,agent_num)]
    r_flags = [False for j in range(0,task_num)] 
    
    iteration = 0
    while True:
        if time.time() - start_time >= time_bound:
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        iteration += 1
        
        if iteration > agent_num +task_num:
            iter_over = True
            return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
        
        if all(q_flags) and all(r_flags): #converge, msgs are all the same.
            converge = True
#             break
        for i in range(0,agent_num):
            linked_taskInds = a_taskInds[i]
            
            flag = True
            for t_key in linked_taskInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
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
########################################################
                old_msg = q_msgs[i][t_key]
                if old_msg!={} and any([abs(msgs[d_key] - old_msg[d_key]) > 10**(-5) 
                                        for d_key in old_msg.keys()]):
                    flag = False                
                q_msgs[i][t_key] = msgs
########################################################

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
            
            flag = True
            for a_key in linked_agentInds:
                
                ####### check time bound
                if time.time() - start_time >= time_bound:
                    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
                ####### check time bound
                
                old_msg = r_msgs[j][a_key]

                r_msgs[j][a_key] = {d_key:findMax(tasks,j,q_msgs,linked_agentInds,a_key,d_key,constraints) 
                                      for d_key in ([0,1] if len(a_taskInds[a_key])>1 else [1])}
                
                
                if any([abs(r_msgs[j][a_key][d_key] - old_msg[d_key]) > 10**(-5) for d_key in old_msg.keys()]):
                    flag = False
                    

            
            if flag: # task j sending the same info
                r_flags[j] = True
                
    return resultCal(agent_num,tasks,constraints,r_msgs,q_msgs,iteration,iter_over,converge,gamma)
           

# %%
def OPD(agent_num,tasks,constraints,gamma):
    edge_pruned = 0
    task_num = len(tasks)
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
                task_reward(tasks[j],[a_key for a_key in com_dict[-1].keys() if com_dict[-1][a_key] == 1],gamma)
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
                    edge_pruned +=1
                    break
                
        for t_ind in range(0,len(t_flag)):
            if not t_flag[t_ind]:
                t_agentInds[a_taskInds[i][t_ind]].remove(i)
        
        new_a_taskInds = [a_taskInds[i][t_ind] 
                          for t_ind in range(0,len(a_taskInds[i])) if t_flag[t_ind]]
        a_taskInds[i] = new_a_taskInds
    return a_taskInds,t_agentInds,edge_pruned

# %%
def eGreedy2(agent_num,tasks,constraints,eps = 0,gamma = 1,CS=[]):
    re_assign = 0
    task_num = len(tasks)
    alloc = [task_num for i in range(0,agent_num)] # each indicate the current task that agent i is allocated to, if = N, means not allocated
    a_taskInds = constraints[0]
    if CS ==[]:
        CS = [[] for j in range(0,task_num+1)] # current coalition structure, the last one is dummy coalition
        cur_con = [0 for i in range(0,agent_num)]
    else:
        CS.append([])
        for j in range(0,task_num):
            for i in CS[j]:
                alloc[i] = j
        cur_con = [0 if alloc[i] == task_num 
                   else agent_con(tasks[alloc[i]],i,CS[alloc[i]],gamma) for i in range(0,agent_num)]

    task_cons = [[agent_con(tasks[j],i,CS[j],gamma) 
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
#             t_index = max_moveIndexs[a_index]
            
        else:
            # exploitation: allocationelse based on reputation or efficiency
            a_index = np.argmax(max_moveVals)
            t_index  = a_taskInds[a_index][max_moveIndexs[a_index]] if max_moveIndexs[a_index]< len(a_taskInds[a_index]) else task_num

            #             t_index = max_moveIndexs[a_index]
                
        
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
                task_cons[i][t_index] = agent_con(tasks[t_index], i, CS[t_index],gamma)
                cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0
        
        # update agent in the old coalition (if applicable)
        if old_t_index != task_num: # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assign +=1
            CS[old_t_index].remove(a_index)
            affected_a_indexes.extend(CS[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in CS[old_t_index]: 
                task_cons[i][old_t_index] = agent_con(tasks[old_t_index], i, CS[old_t_index],gamma)
                cur_con[i] = task_cons[i][old_t_index]
   
        for i in affected_a_indexes:
            move_vals[i] = [task_cons[i][j] - cur_con[i] 
                  if j in a_taskInds[i]+[task_num] else -1000 for j in range(0,task_num+1)]
            

        ## updating other agents r.w.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0,agent_num):
                if (i not in CS[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_con(tasks[t_ind], i,CS[t_ind],gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]


        max_moveIndexs = [np.argmax([move_vals[i][j] for j in a_taskInds[i]+[task_num]]) for i in range(0,agent_num)]

        max_moveVals = [move_vals[i][a_taskInds[i][max_moveIndexs[i]]] if max_moveIndexs[i] < len(a_taskInds[i]) 
                    else move_vals[i][task_num]
                                 for i in range(0,agent_num)]
        


    return CS, sys_rewards_tasks(tasks, CS,gamma), iteration, re_assign

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
run_num = 1000
gamma = 1
a_min_edge = 1
t_max_edge = 5
ex_identifier = 0
time_bound = 600


for run in range(0,run_num):
    task_num = min_t_num
    while task_num<= max_t_num:
        ex_identifier +=1
        print('ex_identifier:',ex_identifier)
            
        #agent_num = np.random.randint(task_num,3*task_num)
        agent_num = 2*task_num
        constraints = gen_constraints(agent_num,task_num,1, a_min_edge,t_max_edge) #max(M,N)
        tasks = gen_tasks(task_num, agent_num, constraints)
            
        a_taskInds = constraints[0]
        num_com = reduce(mul, [1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])
        print('task_num:',task_num,'  agent_num:',agent_num)

            
        result = {"ex_identifier":ex_identifier,"task_num":task_num,"agent_num":agent_num}
        data = {"ex_identifier":ex_identifier,"agent_num":agent_num,"tasks":tasks,"constraints":constraints}
            
        start= time.time()
        r = eGreedy2(agent_num,tasks,constraints,eps = 0,gamma =gamma)
        end = time.time()
        result['g'] = r[1] 
        result['g_iter'] = r[2]
        result['g_reass'] = r[3]
        result['g_t'] = end-start
        print("eGreedy time:", result['g_t'],'result:',result['g'], 
              "iteration:",result['g_iter'],' re-assignment',result['g_reass'])
        
        
        r = random(agent_num,tasks,constraints, gamma)
        alloc = r[0]
        result['rand'] = r[1]
        print('rand result',result['rand'])
        start=time.time()
        r = eGreedy2(agent_num,tasks,constraints,eps = 0,gamma =gamma,CS = alloc_to_CS(tasks,alloc))
        end = time.time() 
        end - start
        result['rand_g'] = r[1] 
        result['rand_g_iter'] = r[2]
        result['rand_g_reass'] = r[3]
        result['rand_g_t'] = end-start
        print("Rand eGreedy time:", result['rand_g_t'],'result:',result['rand_g'], 
              "iteration:",result['rand_g_iter'],' re-assignment',result['rand_g_reass'])

        # do not show result when task number larger than 400       
        if task_num <= 400:
            start= time.time()
            opd = OPD(agent_num,tasks,constraints,gamma)
            new_con = opd[0:2]
            result['OPD_t'] = time.time() - start
            result['OPD_pruned'] = opd[2]
            print("OPD time used:",result['OPD_t']," Edge pruned:",result['OPD_pruned'])
            r = FMS(agent_num,tasks,new_con,gamma = gamma,time_bound = time_bound)
            end = time.time()
            
            result['BnBFMS'] = r[1] 
            result['BnBFMS_iter'] = r[2]
            result['BnBFMS_t'] = end-start
            result['BnBFMS_converge'] = r[4]
            print("BnB FMS time:", result['BnBFMS_t'],'result:',result['BnBFMS'], "iteration:",result['BnBFMS_iter'],"converge?",result['BnBFMS_converge'])
            print()

            
#         append data and result 
        files = {'result_rdfunc':[result,'']}
        for filename in list(files.keys()):
            append_record(files[filename][0],filename,typ = files[filename][1])
            
        #increase the task_num
        
        task_num += 100

# %%



