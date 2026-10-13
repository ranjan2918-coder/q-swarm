"""Hybrid Variational Quantum Circuit (VQC) Policy with Qiskit."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

from qswarm.policies.features import compress_features_deterministic


def build_vqc_circuit(
    num_qubits: int = 4,
    num_layers: int = 2,
    entanglement: str = "ring",
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Construct parameterized quantum circuit with angle encoding and entangling ansatz.

    Supported entanglement options:
    - 'ring': Hardware-efficient periodic ring CNOTs (qubit i -> (i+1)%n)
    - 'linear': Nearest-neighbor linear CNOTs (qubit i -> i+1)
    - 'none': Disentangled separable product state for ablation studies
    """
    qc = QuantumCircuit(num_qubits)
    input_params = ParameterVector("x", num_qubits)
    num_var_params = num_qubits * 2 * num_layers
    var_params = ParameterVector("θ", num_var_params)

    # 1. Angle Encoding: Ry(x_i) on each qubit
    for i in range(num_qubits):
        qc.ry(input_params[i], i)

    qc.barrier()

    # 2. Variational Layers: Ry + Rz rotations + entangling block
    idx = 0
    for layer in range(num_layers):
        for i in range(num_qubits):
            qc.ry(var_params[idx], i)
            qc.rz(var_params[idx + 1], i)
            idx += 2

        # Entanglement block
        if entanglement == "ring":
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)
        elif entanglement == "linear":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "none":
            # Disentangled product state ablation baseline: 0 entangling gates
            pass

        qc.barrier()

    return qc, input_params, var_params


class VQCExecutionEngine:
    """Evaluates parameterized quantum circuits and computes exact Pauli-Z expectation values."""

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        entanglement: str = "ring",
        noise_model: Optional[dict] = None,
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.noise_model = noise_model
        self.circuit, self.input_params, self.var_params = build_vqc_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            entanglement=entanglement,
        )
        self.num_weights = len(self.var_params)

        # Predefine Pauli-Z observables for each qubit
        self.observables = []
        for i in range(num_qubits):
            # Qiskit qubit string order: qubit 0 is rightmost (index -1)
            pauli_str = ["I"] * num_qubits
            pauli_str[num_qubits - 1 - i] = "Z"
            self.observables.append(SparsePauliOp("".join(pauli_str)))

    def evaluate_expectations(
        self,
        x_vals: np.ndarray,      # Shape (num_qubits,)
        theta_vals: np.ndarray,  # Shape (num_weights,)
        add_noise: bool = False,
    ) -> np.ndarray:
        """Compute expectation values <Z_i> for all qubits."""
        param_dict = {}
        for p, v in zip(self.input_params, x_vals):
            param_dict[p] = float(v)
        for p, v in zip(self.var_params, theta_vals):
            param_dict[p] = float(v)

        bound_qc = self.circuit.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_qc)

        exp_vals = np.zeros(self.num_qubits, dtype=np.float32)
        for i, obs in enumerate(self.observables):
            ev = float(np.real(sv.expectation_value(obs)))
            # Calibrated noise perturbation
            if add_noise and self.noise_model:
                depol = self.noise_model.get("depolarizing_error", 0.0)
                ro_err = self.noise_model.get("readout_error", 0.0)
                ev = ev * (1.0 - 2.0 * depol)
                # Use a dedicated Generator instead of the global numpy random state
                noise_jitter = float(np.random.default_rng().normal(0.0, (depol + ro_err) * 0.5))
                ev += noise_jitter
            exp_vals[i] = np.clip(ev, -1.0, 1.0)

        return exp_vals

    def compute_parameter_shift_gradient(
        self,
        x_vals: np.ndarray,
        theta_vals: np.ndarray,
    ) -> np.ndarray:
        """Compute exact gradients d<Z>/d(theta) using the parameter-shift rule."""
        shift = np.pi / 2.0
        jacobian = np.zeros((self.num_qubits, self.num_weights), dtype=np.float32)

        for j in range(self.num_weights):
            th_plus = theta_vals.copy()
            th_plus[j] += shift
            exp_plus = self.evaluate_expectations(x_vals, th_plus)

            th_minus = theta_vals.copy()
            th_minus[j] -= shift
            exp_minus = self.evaluate_expectations(x_vals, th_minus)

            jacobian[:, j] = 0.5 * (exp_plus - exp_minus)

        return jacobian


class VQCFunction(torch.autograd.Function):
    """Custom PyTorch autograd function bridging Qiskit VQC via parameter-shift."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor, engine: VQCExecutionEngine, use_noise: bool):
        ctx.engine = engine
        ctx.use_noise = use_noise
        batch_size = inputs.shape[0]
        device = inputs.device

        inputs_np = inputs.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        outputs = np.zeros((batch_size, engine.num_qubits), dtype=np.float32)
        jacobians = np.zeros((batch_size, engine.num_qubits, engine.num_weights), dtype=np.float32)

        for b in range(batch_size):
            outputs[b] = engine.evaluate_expectations(inputs_np[b], weights_np, add_noise=use_noise)
            jacobians[b] = engine.compute_parameter_shift_gradient(inputs_np[b], weights_np)

        ctx.save_for_backward(inputs, weights)
        ctx.jacobians = torch.tensor(jacobians, dtype=torch.float32, device=device)

        return torch.tensor(outputs, dtype=torch.float32, device=device)

    @staticmethod
    def backward(ctx, grad_output):
        jacobians = ctx.jacobians
        grad_weights = torch.einsum("bk,bkw->w", grad_output, jacobians)
        return None, grad_weights, None, None


class HybridVQCPolicy(nn.Module):
    """Full Hybrid Quantum-Classical Policy for Swarm Reinforcement Learning.

    Architecture:
    Compressed Features x in [-pi, pi]^4 -> 4-qubit VQC -> <Z_i> in [-1, 1]^4 -> Linear(4, 5) -> Q(s, a).
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_dim: int = 5,
        entanglement: str = "ring",
        encoding: str = "graph_aware",
        noise_model: Optional[dict] = None,
        use_noise: bool = False,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.entanglement = entanglement
        self.encoding = encoding
        self.use_noise = use_noise
        self.noise_model = noise_model

        self.engine = VQCExecutionEngine(
            num_qubits=num_qubits,
            num_layers=num_layers,
            entanglement=entanglement,
            noise_model=noise_model,
        )

        init_weights = np.random.uniform(-0.1, 0.1, size=self.engine.num_weights).astype(np.float32)
        self.quantum_weights = nn.Parameter(torch.tensor(init_weights))
        self.head = nn.Linear(num_qubits, output_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_quantum_parameters(self) -> int:
        return self.quantum_weights.numel()

    def get_circuit_depth(self) -> int:
        return self.engine.circuit.depth()

    def get_circuit_diagram(self) -> str:
        return str(self.engine.circuit.draw(output="text"))

    def estimate_resources(self) -> Dict[str, Any]:
        """Compute exhaustive quantum resource estimation evidence."""
        qc = self.engine.circuit
        ops = dict(qc.count_ops())
        two_q_gates = ops.get("cx", 0)
        one_q_gates = ops.get("ry", 0) + ops.get("rz", 0)

        # Transpile against target superconducting coupling map
        coupling_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 3]]
        cmap = CouplingMap(coupling_edges)
        transpiled = transpile(
            qc,
            basis_gates=["ecr", "id", "rz", "sx", "x"],
            coupling_map=cmap,
            optimization_level=2,
            seed_transpiler=42,
        )
        transpiled_ops = dict(transpiled.count_ops())

        return {
            "num_qubits": self.num_qubits,
            "variational_parameters": self.count_quantum_parameters(),
            "classical_head_parameters": self.count_parameters() - self.count_quantum_parameters(),
            "total_trainable_parameters": self.count_parameters(),
            "circuit_depth_ideal": qc.depth(),
            "circuit_depth_transpiled": transpiled.depth(),
            "one_qubit_gates_ideal": one_q_gates,
            "two_qubit_gates_ideal": two_q_gates,
            "two_qubit_gates_transpiled": transpiled_ops.get("ecr", 0),
            "circuit_evaluations_per_action": 1,
            "circuit_evaluations_per_update": 2 * self.count_quantum_parameters(),
            "shots": 1024,
            "entanglement_type": self.entanglement,
            "encoding_type": self.encoding,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expectations = VQCFunction.apply(x, self.quantum_weights, self.engine, self.use_noise)
        q_values = self.head(expectations)
        return q_values

    def get_last_expectations(self, obs: np.ndarray) -> np.ndarray:
        comp_feat = compress_features_deterministic(obs, target_dim=self.num_qubits, encoding_type=self.encoding)
        weights_np = self.quantum_weights.detach().cpu().numpy()
        return self.engine.evaluate_expectations(comp_feat, weights_np, add_noise=self.use_noise)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(0, self.output_dim))

        self.eval()
        with torch.no_grad():
            comp_feat = compress_features_deterministic(obs, target_dim=self.num_qubits, encoding_type=self.encoding)
            weights_np = self.quantum_weights.cpu().numpy()
            exp_vals = self.engine.evaluate_expectations(comp_feat, weights_np, add_noise=self.use_noise)
            exp_t = torch.tensor(exp_vals, dtype=torch.float32).unsqueeze(0)
            q_vals = self.head(exp_t)
            return int(torch.argmax(q_vals, dim=-1).item())

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "entanglement": self.entanglement,
            "encoding": self.encoding,
            "type": "hybrid_vqc",
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HybridVQCPolicy":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model = cls(
            num_qubits=checkpoint.get("num_qubits", 4),
            num_layers=checkpoint.get("num_layers", 2),
            output_dim=checkpoint.get("output_dim", 5),
            entanglement=checkpoint.get("entanglement", "ring"),
            encoding=checkpoint.get("encoding", "graph_aware"),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
