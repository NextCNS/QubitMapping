import itertools
import networkx as nx
import csv
error_rate = []
changed_error_rate =[]
fidelity = []
execution_time = []
entire_sum = []
subgraphs = []
min_error_graph = []
isomorphic_subgraphs = []


def get_error_value(S):
    with open('/content/ibmq_toronto_calibrations_2023-04-04T00_31_29Z.csv', newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      error_value = 0
      set = []
      for val in S:
        set.append(int(val))
      for node in set:
        for row in reader:
          if int(row['Qubit']) == int(node):
            error_value = error_value + float(row['Readout assignment error '])
            gate_errors = row['CNOT error '].split(';')
            #for g in gate_errors:
             # print(g)
            gate_error = [float(g.split(':')[1]) for g in gate_errors]
            sum_gate_error = sum(gate_error)
            error_value = error_value + sum_gate_error
            break
    return error_value

import itertools
import networkx as nx
import csv
changed_error_rate = []
fidelity = []
execution_time = []
entire_sum = []
subgraphs = []
min_error_graph = []
isomorphic_subgraphs = []


def get_changed_error_value(S):
    with open('/content/ibmq_toronto_calibrations_2023-04-02T20_41_14Z.csv', newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      changed_error_value = 0
      set = []
      for val in S:
        set.append(int(val))
      for node in set:
        for row in reader:
          if int(row['Qubit']) == int(node):
            changed_error_value = changed_error_value + float(row['Readout assignment error '])
            changed_gate_errors = row['CNOT error '].split(';')
            #for g in gate_errors:
             # print(g)
            changed_gate_error = [float(g.split(':')[1]) for g in changed_gate_errors]
            changed_sum_gate_error = sum(changed_gate_error)
            changed_error_value = changed_error_value + changed_sum_gate_error
            break
    return changed_error_value

def get_fidelity(S):
    with open('/content/ibmq_toronto_calibrations_2023-04-02T20_41_14Z.csv', newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      error_value = 0
      fidelity_value = 1
      set = []
      for val in S:
        set.append(int(val))
      for node in set:
        for row in reader:
          if int(row['Qubit']) == int(node):
            # error_value = error_value + float(row['Readout assignment error '])
            gate_error = row['CNOT error '].split(';')
            #for g in gate_errors:
             # print(g)
            gate_errors = [float(g.split(':')[1]) for g in gate_error]
            for list in gate_errors:
              fidelity_value= fidelity_value*(1-list)
              # print(fidelity)
            # sum_gate_error = sum(gate_error)
            # error_value = error_value + sum_gate_error
            # fidelity_value = fidelity
            break
    return fidelity_value

def get_gate_execution_time(S):
  with open('/content/ibmq_toronto_calibrations_2023-04-04T00_31_29Z.csv', newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      #error_value = 0
      set = []
      for val in S:
        set.append(int(val))
      for node in set:
        for row in reader:
          if int(row['Qubit']) == int(node):
            #error_value = error_value + float(row['Gate time (ns)'])
            gate_execution_time = row['Gate time (ns)'].split(';')
            #for g in gate_errors:
             # print(g)
            gate_execution_time = [float(g.split(':')[1]) for g in gate_execution_time]
            sum_gate_execution_time = sum(gate_execution_time)
            time_value = (sum_gate_execution_time)/2000
            break
  return time_value

from math import inf
def extract_unique_subgraphs(G, n):
    prev_error_rate = 1
    prev_execution_time = inf
    nodes = list(G.nodes())
    for i, node_combination in enumerate(itertools.combinations(nodes, n)):
        # Create a subgraph from the current combination of nodes
        subgraph = G.subgraph(node_combination)
        # Check if the subgraph is connected
        if not nx.is_connected(subgraph):
            continue
        # Check if the subgraph is unique and not isomorphic to previous subgraphs
        is_unique = True
        for j, previous_subgraph in enumerate(subgraphs):
            if nx.is_isomorphic(subgraph, previous_subgraph):
                is_unique = False
                isomorphic_subgraphs.append((node_combination, previous_subgraph.nodes()))
                break
        if is_unique:
            subgraphs.append(subgraph)
    return isomorphic_subgraphs

T = nx.Graph()
#T.add_edges_from([(0,1),(1,2),(1,4),(2,3),(3,5),(4,7),(5,8),(6,7),(7,10),(8,9),(8,11),(10,12),(11,14),(12,13),(13,14),(14,16),(15,18),(16,19),(17,18),(18,21),(19,20),(21,23),(22,25),(23,24),(24,25),(25,26)])
T.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
T.add_edge(0,1)
T.add_edge(1,2)
T.add_edge(1,4)
T.add_edge(2,3)
T.add_edge(3,5)
T.add_edge(4,7)
T.add_edge(5,8)
T.add_edge(6,7)
T.add_edge(7,10)
T.add_edge(8,9)
T.add_edge(8,11)
T.add_edge(10,12)
T.add_edge(11,14)
T.add_edge(12,13)
T.add_edge(13,14)
T.add_edge(14,16)
T.add_edge(15,18)
T.add_edge(16,19)
T.add_edge(17,18)
T.add_edge(18,21)
T.add_edge(19,20)
T.add_edge(21,23)
T.add_edge(22,25)
T.add_edge(23,24)
T.add_edge(24,25)
T.add_edge(25,26)
k = 4

subgraph = extract_unique_subgraphs(T, k)
for graph in subgraph:
  print(graph)

import matplotlib.pyplot as plt
import numpy as np
import random
x = np.random
y1 = error_rate[15::]
y2 = execution_time[15::]
y3 = changed_error_rate[15::]


ax = plt.axes()
plt.plot(y1, label = 'Error_rate', color = 'red')
plt.plot(y2, label = 'Execution_time', color = 'blue')
plt.title("Error rates and execution time rates of \n different mapping to physical topology \n executing same circuit", fontsize = 20)
plt.xlabel('Different Topologies', fontsize = 20)
plt.ylabel('Error rate and \n Execution time rate', fontsize = 20)
plt.legend()
plt.grid()
plt.savefig("Error_rate_subgraph.pdf", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt

x_data = range(0, len(fidelity))
plt.bar(x_data, fidelity)
plt.xlabel("Subgraphs from physical topology")
plt.ylabel("Fidelity")
plt.title("Fidelity for different mapping to physical qubits")
plt.show()

import matplotlib.pyplot as plt

x_data = range(0, len(execution_time))
y_data = range(0, len(error_rate))
plt.bar(x_data, execution_time)
plt.xlabel("Error rate")
plt.ylabel("Execution time")
plt.title("Execution times for different mapping to physical qubits")

plt.savefig("qubit_subgraph_2.pdf", bbox_inches="tight")
plt.show()

from google.colab import drive
drive.mount('/content/drive')

%%capture
!pip install qiskit

from qiskit import QuantumCircuit, transpile
#from qiskit.providers.fake_provider import FakeToronto
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import BasicSwap
#from qiskit import Aer

# Define the logical circuit
circ = QuantumCircuit(4)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 3)
circ.cx(0, 2)
circ.cx(1, 3)
circ.cx(0, 3)
circ.cx(0, 2)
circ.cx(1, 2)
circ.cx(0, 2)
circ.cx(1, 3)
circ.cx(2, 3)
circ.cx(0, 2)
circ.cx(2, 3)

qubit_count = circ.num_qubits
print(qubit_count)
coupling_map = CouplingMap([(0, 1), (1, 2), (1, 3)])
mapping = {circ.qubits[0]: 0, circ.qubits[1]: 1, circ.qubits[2]: 2, circ.qubits[3]: 3}
initial_layout = Layout(mapping)
transpiled_circ = transpile(circ, coupling_map=coupling_map, initial_layout=initial_layout, optimization_level=0)
print("Transpiled Circuit:")
print(transpiled_circ)
print("Number of swap operations:", transpiled_circ.count_ops()['swap'])

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import BasicSwap

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

coupling_map = CouplingMap([(0, 1), (1, 2)])

initial_layout = Layout({0: 0, 1: 1, 2: 2})

mapped_circuit = transpile(circuit, coupling_map=coupling_map, initial_layout=initial_layout,
                           optimization_level=1, layout_method='basic_aware')

print(mapped_circuit)

from qiskit import QuantumCircuit, execute, Aer
from qiskit.compiler import transpile
from qiskit.providers.aer import QasmSimulator

# Define the circuit
qasm = """
OPENQASM 2.0;

include "qelib1.inc";

qreg q[20];

cx q[17], q[11];
cx q[10], q[15];
cx q[7], q[16];
cx q[11], q[6];
t q[18];
h q[6];
h q[6];
y q[18];
z q[6];
h q[12];
y q[7];
cx q[6], q[14];
z q[14];
z q[11];
cx q[0], q[1];
cx q[14], q[6];
cx q[3], q[1];
cx q[6], q[14];
h q[6];
cx q[9], q[14];
cx q[6], q[14];
y q[9];
cx q[6], q[11];
cx q[13], q[10];
cx q[6], q[9];
cx q[6], q[14];
h q[9];
cx q[19], q[3];
y q[6];
y q[6];
z q[13];
cx q[16], q[8];
"""

circ = QuantumCircuit.from_qasm_str(qasm)

circ.draw()

import numpy as np

def initial_mapping(freq_matrix, swap_matrix):
    n = freq_matrix.shape[0]
    qubit_mapping = {}
    available_qubits = set(range(n))
    for i in range(n):
        logical_qubit = np.argmax(freq_matrix[i])
        available_physical_qubits = [j for j in range(n) if j in available_qubits]
        physical_qubit = min(available_physical_qubits, key=lambda j: swap_matrix[logical_qubit, j])
        qubit_mapping[i] = physical_qubit
        available_qubits.remove(physical_qubit)

    print("Initial mapping:")
    for i in range(n):
        print(f"q[{i}] -> q[{qubit_mapping[i]}]")

    return qubit_mapping

freq_matrix = np.array([[0, 2, 8, 4],
                       [2, 0, 2, 0],
                       [8, 2, 0, 6],
                       [4, 0, 6, 0]])

swap_matrix = np.array([[0, 0, 1, 2],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [2, 1, 0, 0]])

qubit_mapping = initial_mapping(freq_matrix, swap_matrix)


def check_executable(logical_circuit, qubit_mapping, physical_topology):
    for gate in logical_circuit:
        gate_type, qubit_indices = gate
        print("gate_type", gate_type)
        print("qubit_indices", qubit_indices)
        physical_qubit_indices = [qubit_mapping[i] for i in qubit_indices]
        for qubit_index in physical_qubit_indices:
            print("qubit_index", qubit_index)
            print("physical_topology", physical_topology)
            print("physical_topology[qubit_index]", physical_topology[qubit_index])
            if not all(neighbor in physical_qubit_indices or neighbor in physical_topology[qubit_index] for neighbor in physical_topology[qubit_index]):
              print("if")
              for neighbor in physical_topology[qubit_index]:
                print(neighbor)
              if qubit_index not in physical_topology:
                return False
    return True


logical_circuit = [('H', [0]), ('CX', [0, 1]), ('H', [1]), ('CX', [1, 2]), ('CX', [0, 3])]
qubit_mapping = qubit_mapping

physical_topology = {0: [1], 1: [0, 2], 2: [1, 3], 3:[2]}

is_executable = check_executable(logical_circuit, qubit_mapping, physical_topology)
print(is_executable)

import numpy as np

logical_map = []
neighbor_qubit = {}

def initial_mapping(freq_matrix, distance_matrix):
    # Compute the initial mapping using the frequency interaction matrix and swap distance matrix
    n = freq_matrix.shape[0]
    qubit_mapping = []
    available_qubits = set(range(n))
    for i in range(n):
        logical_qubit = np.argmax(freq_matrix[i])
        available_physical_qubits = [j for j in range(n) if j in available_qubits]
        physical_qubit = min(available_physical_qubits, key=lambda j: distance_matrix[logical_qubit, j])
        qubit_mapping.append(physical_qubit)
        logical_map.append(logical_qubit)
        available_qubits.remove(physical_qubit)

    # Associate qubit_mapping with logical_map
    for i in range(n):
        neighbor_qubit[qubit_mapping[i]] = logical_map[i]

    # Print the initial mapping
    print("Initial mapping:")
    for i in range(n):
      print(f"Q[{qubit_mapping[i]}] -> q[{logical_map[i]}]")

    return qubit_mapping, logical_map

# Define the frequency interaction matrix
freq_matrix = np.array([[0, 1, 4, 1],
                        [1, 0, 1, 3],
                        [4, 1, 0, 2],
                        [1, 3, 2, 0]])

# Define the swap distance matrix
swap_matrix = np.array([[0, 0, 1, 2],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [2, 1, 0, 0]])


qubit_mapping, logical_map = initial_mapping(freq_matrix, swap_matrix)

sorted_mapping = sorted(zip(qubit_mapping, logical_map), key=lambda x: x[1])

# Print the association
print("Association:")
for i in range(len(sorted_mapping)):
    print(f"Q[{sorted_mapping[i][0]}] -> q[{sorted_mapping[i][1]}]")
