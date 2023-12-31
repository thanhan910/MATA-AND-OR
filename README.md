# Multiagent Task Allocation AND-OR

2023/24 Summer Research Project. 

## Program details

### Program execution

Execute the following command to run the program:

```bash
python main.py
```

### Folder structure

heterogeneous/ - contains the code for the original heterogeneous task allocation problem, where tasks are independent.

andortree/ - contains the code for the AND-OR tree task allocation problem.

utils/ - contains general utility functions, including the functions that generates tasks, agents, and constraints.

main.py - The main driver script that generate problems and evaluates solutions.

## Project details

### Project title: Multiagent coordination and self-organisation to enable resilient satellite constellations.

### Project description

In this project, we aim to extend the GreedyNE algorithm developed in Qinyuan Li's PhD thesis (2022, Bao Quoc Vo's PhD student) for multiagent task allocation problem to accommodate the scenarios where the tasks are inter-related and represented as an AND-OR goal tree. 

The Summer Project student is expected to implement a standard Branch-and-Bound algorithm and also develop a new heuristic-based approximate algorithm to efficiently solve the problem and compare the performance of the two algorithms.

### References

Q. Li, M. Li, B. Quoc Vo and R. Kowalczyk, "An Anytime Algorithm for Large-scale Heterogeneous Task Allocation," 2020 25th International Conference on Engineering of Complex Computer Systems (ICECCS), Singapore, 2020, pp. 206-215, doi: 10.1109/ICECCS51672.2020.00031.
