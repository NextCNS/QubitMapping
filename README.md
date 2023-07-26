# Qubit Mapping


This repository contains the implementation of a suite of algorithms proposed in the research paper "Towards Fidelity-Optimal Qubit Mapping on NISQ Computers" by Sri Khandavilli, Indu Palanisamy, Manh V. Nguyen, Thinh V. Le, Thang N. Dinh, and Tu N. Nguyen. The proposed algorithms were evaluated on the IBM-provided Noisy Intermediate-Scale Quantum (NISQ) computer, using a dataset consisting of 17 different quantum circuits of various sizes. The circuits were executed on the IBM Toronto Falcon processor. 


IBMQ Toronto Falcon processor Topology:
![image](https://github.com/NextCNS/Qubit_mapping/assets/22426590/ca8bc9d2-5320-4251-8e47-116332dcaba8)


The research project addresses the challenges of implementing quantum algorithms on near-term quantum devices. These challenges include limited physical qubit connections, which necessitate the use of quantum SWAP gates to dynamically transform the logical topology during execution, and the need to optimize fidelity by ensuring that the allocated hardware has a low error rate and that the number of SWAP gates injected into the circuit is minimized.

To address these challenges, the paper proposes a suite of algorithms, including:

•	Fidelity-aware Graph Extraction Algorithm (FGEA): used to identify the hardware region with the lowest probability of error.

•	Frequency-based Mapping Algorithm (FMA): allocates logical-physical qubits that reduce the potential distance of topological transformation.

•	Heuristic Routing Algorithm (HRA): searches for an optimal swapping injection strategy.

Algorithm flow-map for Optimal Mapping:
![image](https://github.com/NextCNS/Qubit_mapping/assets/22426590/77604304-a42a-4a1a-bdc8-15a56c6c0f20)


# File Structure
map.py - consists of FGEA, FMA, HRA functions and benchmark methods.

# Libraries
1. Python 3

2. NumPy

3. Matplotlib
