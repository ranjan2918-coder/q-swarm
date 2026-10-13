"""QAOA Quantum Optimization for Mesh Relay Selection and Network Partition Repair.

Positions quantum optimization as a rigorous feasibility experiment for repairing
disconnected multi-drone communication graphs on small topologies (4-6 drones).
"""

import itertools
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


@dataclass
class OptimizationComparisonResult:
    graph_nodes: int
    graph_edges: int
    exact_optimal_val: float
    exact_solution: List[int]
    exact_time_s: float
    greedy_val: float
    greedy_solution: List[int]
    greedy_time_s: float
    qaoa_val: float
    qaoa_solution: List[int]
    qaoa_approx_ratio: float
    qaoa_time_s: float
    qaoa_depth: int
    qaoa_cx_count: int


def build_relay_graph_from_partition(drone_positions: np.ndarray, comm_range: float = 8.0) -> nx.Graph:
    """Build bridge candidate graph where edges represent feasible links between drones."""
    n = len(drone_positions)
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, pos=drone_positions[i])

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(drone_positions[i] - drone_positions[j]))
            if dist <= comm_range:
                weight = max(0.1, 1.0 - (dist / comm_range))
                g.add_edge(i, j, weight=weight)
    return g


class ExactRelaySolver:
    """Exhaustive brute-force classical solver evaluating all 2^n configurations."""

    @staticmethod
    def solve(graph: nx.Graph) -> Tuple[float, List[int], float]:
        t0 = time.perf_counter()
        n = graph.number_of_nodes()
        best_val = -float("inf")
        best_cfg = [0] * n

        for bitstring in itertools.product([0, 1], repeat=n):
            val = evaluate_relay_cut_value(graph, bitstring)
            if val > best_val:
                best_val = val
                best_cfg = list(bitstring)

        elapsed = time.perf_counter() - t0
        return float(best_val), best_cfg, elapsed


class GreedyRelaySolver:
    """Fast polynomial-time greedy heuristic selecting nodes with highest bridge score."""

    @staticmethod
    def solve(graph: nx.Graph) -> Tuple[float, List[int], float]:
        t0 = time.perf_counter()
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        # Sort nodes by weighted degree
        deg_sorted = sorted(nodes, key=lambda u: graph.degree(u, weight="weight"), reverse=True)
        current_cfg = [0] * n
        best_val = 0.0

        for u in deg_sorted:
            trial_cfg = list(current_cfg)
            trial_cfg[u] = 1
            trial_val = evaluate_relay_cut_value(graph, trial_cfg)
            if trial_val >= best_val:
                best_val = trial_val
                current_cfg = trial_cfg

        elapsed = time.perf_counter() - t0
        return float(best_val), current_cfg, elapsed


def evaluate_relay_cut_value(graph: nx.Graph, config: Sequence[int]) -> float:
    """Objective function: Maximize bridged edges between activated relays and mesh."""
    val = 0.0
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1.0)
        # Max-cut style relay connection
        if config[u] != config[v]:
            val += w
        elif config[u] == 1 and config[v] == 1:
            val += 0.5 * w
    # Regularization penalty for energy consumption of active relays
    val -= 0.15 * sum(config)
    return float(val)


class QAOARelayOptimizer:
    """Quantum Approximate Optimization Algorithm (QAOA) for relay repair."""

    def __init__(self, graph: nx.Graph, p_steps: int = 1):
        self.graph = graph
        self.p_steps = p_steps
        self.n_qubits = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.circuit, self.gamma_params, self.beta_params = self._build_qaoa_circuit()

    def _build_qaoa_circuit(self) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        qc = QuantumCircuit(self.n_qubits)
        gammas = ParameterVector("γ", self.p_steps)
        betas = ParameterVector("β", self.p_steps)

        # 1. Equal superposition
        for i in range(self.n_qubits):
            qc.h(i)

        qc.barrier()

        # 2. Alternating Cost and Mixer layers
        for step in range(self.p_steps):
            gamma = gammas[step]
            beta = betas[step]

            # Cost unitary e^{-i γ H_C}: Rzz rotations on graph edges
            for u, v, d in self.graph.edges(data=True):
                w = d.get("weight", 1.0)
                # Two-qubit ZZ interaction
                qc.cx(u, v)
                qc.rz(2 * gamma * w, v)
                qc.cx(u, v)

            # Mixer unitary e^{-i β H_M}: Rx rotations on each qubit
            for i in range(self.n_qubits):
                qc.rx(2 * beta, i)

            qc.barrier()

        return qc, gammas, betas

    def solve(self, max_iter: int = 25) -> Tuple[float, List[int], float, int, int]:
        t0 = time.perf_counter()

        def objective_func(params):
            gamma_vals = params[:self.p_steps]
            beta_vals = params[self.p_steps:]

            param_dict = {}
            for p, v in zip(self.gamma_params, gamma_vals):
                param_dict[p] = float(v)
            for p, v in zip(self.beta_params, beta_vals):
                param_dict[p] = float(v)

            bound_qc = self.circuit.assign_parameters(param_dict)
            sv = Statevector.from_instruction(bound_qc)
            probs = sv.probabilities()

            # Minimize negative expectation value
            expected_val = 0.0
            for idx, prob in enumerate(probs):
                bitstr = [int(b) for b in format(idx, f"0{self.n_qubits}b")]
                val = evaluate_relay_cut_value(self.graph, bitstr)
                expected_val += prob * val

            return -expected_val

        # Classical parameter optimization
        init_params = np.random.uniform(0.1, np.pi / 2, size=2 * self.p_steps)
        res = minimize(objective_func, init_params, method="COBYLA", options={"maxiter": max_iter})

        # Sample most probable statevector bitstring
        gamma_opt = res.x[:self.p_steps]
        beta_opt = res.x[self.p_steps:]
        param_dict = {}
        for p, v in zip(self.gamma_params, gamma_opt):
            param_dict[p] = float(v)
        for p, v in zip(self.beta_params, beta_opt):
            param_dict[p] = float(v)

        bound_qc = self.circuit.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_qc)
        probs = sv.probabilities()
        best_idx = int(np.argmax(probs))
        best_bitstring = [int(b) for b in format(best_idx, f"0{self.n_qubits}b")]

        qaoa_val = evaluate_relay_cut_value(self.graph, best_bitstring)
        elapsed = time.perf_counter() - t0

        cx_count = dict(self.circuit.count_ops()).get("cx", 0)
        depth = self.circuit.depth()

        return qaoa_val, best_bitstring, elapsed, depth, cx_count


def run_qaoa_benchmark(num_nodes: int = 4, comm_range: float = 8.0, seed: int = 42) -> OptimizationComparisonResult:
    """Run comparative benchmark between Exact, Greedy, and QAOA on drone mesh repair."""
    # Generate 4-drone positions creating a partitioned / bridge scenario
    positions = np.array([
        [2.0, 2.0],
        [4.0, 3.0],
        [8.0, 8.0],
        [10.0, 9.0],
    ][:num_nodes], dtype=np.float32)

    graph = build_relay_graph_from_partition(positions, comm_range=comm_range)
    # Ensure at least connected or partitioned components
    if graph.number_of_edges() == 0:
        graph.add_edge(0, 1, weight=0.8)
        graph.add_edge(2, 3, weight=0.7)
        graph.add_edge(1, 2, weight=0.4)

    # 1. Exact Classical Solver
    exact_val, exact_sol, exact_time = ExactRelaySolver.solve(graph)

    # 2. Greedy Classical Solver
    greedy_val, greedy_sol, greedy_time = GreedyRelaySolver.solve(graph)

    # 3. QAOA Quantum Solver
    qaoa = QAOARelayOptimizer(graph, p_steps=1)
    qaoa_val, qaoa_sol, qaoa_time, depth, cx_count = qaoa.solve(max_iter=20)

    approx_ratio = qaoa_val / max(1e-4, abs(exact_val)) if exact_val != 0.0 else 0.0

    return OptimizationComparisonResult(
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        exact_optimal_val=round(exact_val, 3),
        exact_solution=exact_sol,
        exact_time_s=round(exact_time, 5),
        greedy_val=round(greedy_val, 3),
        greedy_solution=greedy_sol,
        greedy_time_s=round(greedy_time, 5),
        qaoa_val=round(qaoa_val, 3),
        qaoa_solution=qaoa_sol,
        qaoa_approx_ratio=round(float(approx_ratio), 3),
        qaoa_time_s=round(qaoa_time, 4),
        qaoa_depth=depth,
        qaoa_cx_count=cx_count,
    )
