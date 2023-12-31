from functools import reduce  # python3 compatibility
from operator import mul
import json
import os
import statistics

from utils.problem import *

from heterogeneous.GreedyNE import * 
from heterogeneous.solutions import *
from heterogeneous.upper_bound import *

from andortree.tree_utils import *
from andortree.upper_bound import upperBoundTree_allNodes_v1, upperBoundTree_allNodes_v2, upperBoundTree_root
from andortree.GreedyNE import *
from andortree.solutions import random_solution_and_or_tree
from andortree.GreedyNEClass import GreedyNE


def append_record(record, filename, typ):
    with open(filename, "a") as f:
        if typ != "":
            json.dump(record, f, default=typ)
        else:
            json.dump(record, f)
        f.write(os.linesep)
        f.close()

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
        t_max_edge = 3
        while t_max_edge <= 50:
            ex_identifier += 1

            #         agent_num = np.random.randint(task_num,3*task_num)
            tasks = gen_tasks(task_num, max_capNum_task, capabilities)
            constraints = gen_constraints(
                agent_num, task_num, 1, a_min_edge, t_max_edge
            )
            a_taskInds = constraints[0]
            agents_cap, agents = gen_agents(
                a_taskInds, tasks, max_capNum_agent, capabilities, max_capVal
            )
            tree_info = gen_tree_info(task_num=task_num)
            # num_com = np.prod([1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])

            num_com = reduce(
                mul,
                [
                    1 if a_taskInds[i] == [] else len(a_taskInds[i]) + 1
                    for i in range(0, agent_num)
                ],
            )

            up = upperBound(capabilities, tasks, agents)

            up2 = upperBound_ver2(capabilities, tasks, agents, constraints)
            print("UP:", up, "  UP2:", up2)

            result = {
                "ex_identifier": ex_identifier,
                "task_num": task_num,
                "agent_num": agent_num,
                "up": up,
                "up2": up2,
            }
            #         data = {"ex_identifier":ex_identifier,"tasks":tasks,"constraints":constraints,"agents_cap":agents_cap,"agents":agents}

            a_den = [len(c) for c in constraints[0]]
            t_den = [len(c) for c in constraints[1]]
            result["a_den_avg"] = statistics.mean(a_den)
            result["a_den_max"] = max(a_den)
            result["a_den_min"] = min(a_den)

            result["t_den_avg"] = statistics.mean(t_den)
            result["t_den_max"] = max(t_den)
            result["t_den_min"] = min(t_den)
            result["t_max_edge"] = t_max_edge

            print(
                "density", t_max_edge, "task_num:", task_num, "  agent_num:", agent_num
            )
            print(
                "a_den_avg",
                result["a_den_avg"],
                "a_den_max:",
                result["a_den_max"],
                "  a_den_min:",
                result["a_den_min"],
            )

            print(
                "t_den_avg",
                result["t_den_avg"],
                "t_den_max:",
                result["t_den_max"],
                "  t_den_min:",
                result["t_den_min"],
            )
            print("-----------------------------------")
            print("Heterogeneous Tasks")
            print("-----------------------------------")

            start = time.perf_counter()
            r = eGreedy2(agents, tasks, constraints, gamma=gamma)
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

            start = time.perf_counter()
            rand_sol_a, rand_sol_reward = random_solution_heterogeneous(agents, tasks, constraints, gamma=gamma)
            end = time.perf_counter()
            print(
                "Random Solution:",
                "\ttime:",
                end - start,
                "\tresult:",
                rand_sol_reward,
            )

            print("-----------------------------------")
            print("AND-OR Goal Tree Tasks")
            print("-----------------------------------")

            nodes_constraints = get_nodes_constraints(tree_info, constraints, root_node_index=-1)

            start = time.perf_counter()
            up_tree_root = upperBoundTree_root(tree_info, capabilities, tasks, agents)
            end = time.perf_counter()

            print("UP Tree:", up_tree_root, "\ttime:", end - start)

            start = time.perf_counter()
            up_tree_1 = upperBoundTree_allNodes_v1(tree_info, capabilities, tasks, agents, nodes_constraints)
            end = time.perf_counter()

            print("UP1:", up_tree_1, "\ttime:", end - start)

            start = time.perf_counter()
            up_tree_2 = upperBoundTree_allNodes_v2(tree_info, capabilities, tasks, agents, nodes_constraints)
            end = time.perf_counter()

            print("UP2:", up_tree_2, "\ttime:", end - start)

            start = time.perf_counter()
            r_tree_solver = GreedyNE(agents, tasks, constraints, tree_info=tree_info, gamma=gamma)
            result_c = r_tree_solver.solve()
            end = time.perf_counter()
            print(
                "GreedyNE Class:",
                "\ttime:",
                end - start,
                "\tresult:",
                result_c[1],
                "\titeration:",
                result_c[2],
                "\tre-assignment",
                result_c[3],
            )

            start = time.perf_counter()
            r_tree_0 = greedyNETree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma, greedy_level=0)
            end = time.perf_counter()
            print(
                "GreedyNE 0:",
                "\ttime:",
                end - start,
                "\tresult:",
                r_tree_0[1],
                "\titeration:",
                r_tree_0[2],
                "\tre-assignment",
                r_tree_0[3],
            )

            start = time.perf_counter()
            r_tree_1 = greedyNETree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma, greedy_level=1)
            end = time.perf_counter()
            print(
                "GreedyNE 1:",
                "\ttime:",
                end - start,
                "\tresult:",
                r_tree_1[1],
                "\titeration:",
                r_tree_1[2],
                "\tre-assignment",
                r_tree_1[3],
            )
            start = time.perf_counter()
            r_tree_2 = greedyNETree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma, greedy_level=2)
            end = time.perf_counter()
            print(
                "GreedyNE 2:",
                "\ttime:",
                end - start,
                "\tresult:",
                r_tree_2[1],
                "\titeration:",
                r_tree_2[2],
                "\tre-assignment",
                r_tree_2[3],
            )
            start = time.perf_counter()
            r_tree_0 = greedyNETree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma, greedy_level=0, eps=1)
            end = time.perf_counter()
            print(
                "GreedyNE 0, eps=1:",
                "\ttime:",
                end - start,
                "\tresult:",
                r_tree_0[1],
                "\titeration:",
                r_tree_0[2],
                "\tre-assignment",
                r_tree_0[3],
            )

            start = time.perf_counter()
            r_tree_1 = greedyNETree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma, greedy_level=1, eps=1)
            end = time.perf_counter()
            print(
                "GreedyNE 1, eps=1:",
                "\ttime:",
                end - start,
                "\tresult:",
                r_tree_1[1],
                "\titeration:",
                r_tree_1[2],
                "\tre-assignment",
                r_tree_1[3],
            )
            start = time.perf_counter()
            rand_sol_a, rand_sol_reward = random_solution_and_or_tree(agents, tasks, constraints, tree_info=tree_info, gamma=gamma)
            end = time.perf_counter()
            print(
                "Random Solution:",
                "\ttime:",
                end - start,
                "\tresult:",
                rand_sol_reward,
            )

            print("-----------------------------------")
            print("Heterogeneous Tasks (OPD, FMS)")
            print("-----------------------------------")

            if t_max_edge < 15:
                start = time.perf_counter()
                opd = OPD(agents, tasks, constraints, gamma)
                new_con = opd[0:2]
                result["OPD_t"] = time.perf_counter() - start
                r = FMS(agents, tasks, new_con, gamma=gamma, time_bound=time_bound)
                end = time.perf_counter()
                result["OPD_pruned"] = opd[2]
                print(
                    "OPD time used:",
                    result["OPD_t"],
                    " Edge pruned:",
                    result["OPD_pruned"],
                )
                result["BnBFMS"] = r[1]
                result["BnBFMS_iter"] = r[2]
                result["BnBFMS_t"] = end - start
                result["BnBFMS_converge"] = r[4]
                print(
                    "BnB FMS time:",
                    result["BnBFMS_t"],
                    "result:",
                    result["BnBFMS"],
                    "iteration:",
                    result["BnBFMS_iter"],
                    "converge?",
                    result["BnBFMS_converge"],
                )
            print()

            # # append data and result
            # files = {"density100_result_cap": [result, ""]}

            # for filename in list(files.keys()):
            #     append_record(files[filename][0], filename, typ=files[filename][1])

            # increase the task_num
            t_max_edge += 1
        
        break


if __name__ == "__main__":
    main()