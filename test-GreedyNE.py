from functools import reduce  # python3 compatibility
from operator import mul
import json
import os
import time
import random

from utils.problem import gen_tasks, gen_constraints, gen_agents

from heterogeneous.GreedyNE import eGreedy2
from andortree.GreedyNE import adGreedyNE
# from heterogeneous.solutions import *
# from heterogeneous.upper_bound import *


def main():
    run_num = 1000
    gamma = 1
    a_min_edge = 1

    capNum = 10
    max_capVal = 10
    max_capNum_task = 10
    max_capNum_agent = 10
    ex_identifier = 0
    time_bound = 600
    task_num = 100
    agent_num = 200

    capabilities = list(range(0, capNum))

    for run in range(0, run_num):
        print("----------------------------------------------------------------------")
        print("ITERATION:", run)
        print("----------------------------------------------------------------------")
        t_max_edge = 3
        while t_max_edge <= 50:
            
            ex_identifier += 1

            print("----------------------------------------------------------------------")
            print("EX IDENTIFIER:", ex_identifier)
            print("----------------------------------------------------------------------")


            # Generate tasks, agents, and constraints
            #         agent_num = np.random.randint(task_num,3*task_num)
            tasks = gen_tasks(task_num, max_capNum_task, capabilities)
            constraints = gen_constraints(agent_num, task_num, 1, a_min_edge, t_max_edge)
            a_taskInds = constraints[0]
            agents_cap, agents = gen_agents(a_taskInds, tasks, max_capNum_agent, capabilities, max_capVal)
            # num_com = np.prod([1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])


            result = {}
            #         data = {"ex_identifier":ex_identifier,"tasks":tasks,"constraints":constraints,"agents_cap":agents_cap,"agents":agents}


            start = time.perf_counter()
            r = adGreedyNE(agents=agents, tasks=tasks, constraints=constraints, original_coalition_structure={}, selected_tasks=None, selected_agents=None, gamma=gamma)
            end = time.perf_counter()
            result["g"] = r[1]
            result["g_iter"] = r[2]
            result["g_reass"] = r[3]
            result["g_t"] = end - start
            print(
                "eGreedy:",
                "\ttime:",
                result["g_t"],
                "\tresult:",
                result["g"],
                "\titeration:",
                result["g_iter"],
                "\tre-assignment",
                result["g_reass"],
            )

            # random agents
            rand_agents = random.sample(range(0, agent_num), agent_num)
            rand_agents = list(rand_agents)
            rand_tasks = random.sample(range(0, task_num), task_num)
            rand_tasks = list(rand_tasks)

            print(rand_agents)
            print(rand_tasks)


            start = time.perf_counter()
            r = adGreedyNE(agents=agents, tasks=tasks, constraints=constraints, original_coalition_structure={}, selected_tasks=rand_tasks, selected_agents=list(rand_agents), gamma=gamma)
            end = time.perf_counter()
            result["g"] = r[1]
            result["g_iter"] = r[2]
            result["g_reass"] = r[3]
            result["g_t"] = end - start
            print(
                "eGreedy:",
                "\ttime:",
                result["g_t"],
                "\tresult:",
                result["g"],
                "\titeration:",
                result["g_iter"],
                "\tre-assignment",
                result["g_reass"],
            )

            
            # increase the task_num
            t_max_edge += 1
        
        break


if __name__ == "__main__":
    main()