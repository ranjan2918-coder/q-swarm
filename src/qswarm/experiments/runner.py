"""Benchmark experiment runner executing multi-model, quantum ablation, and optimization trials."""

import datetime
import json
import os
from typing import Dict, List

import pandas as pd
import yaml

from qswarm.experiments.metrics import ScenarioResult
from qswarm.optimization.qaoa_mesh_repair import run_qaoa_benchmark
from qswarm.training.evaluate import evaluate_policy


class BenchmarkRunner:
    """Executes deterministic reproducible comparisons across policies, ablations, and scenarios."""

    def __init__(self, config_path: str = "configs/experiments/benchmark.yaml", smoke: bool = False):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.smoke = smoke
        mode = "smoke" if smoke else "full"
        self.seeds = self.cfg[mode].get("seeds", [42, 101])
        self.max_steps = self.cfg[mode].get("max_steps", 30 if smoke else 60)
        self.models = self.cfg.get(
            "models",
            [
                "heuristic",
                "classical_matched",
                "classical",
                "hybrid_vqc_no_entanglement",
                "hybrid_vqc_entangled",
                "hybrid_vqc_noisy",
            ],
        )
        self.scenarios = self.cfg.get(
            "scenarios",
            ["open_field", "rescue_maze", "dynamic_hazard", "mesh_tradeoff", "scale_test"],
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"benchmark_{'smoke_' if smoke else ''}{timestamp}"
        self.output_dir = os.path.join(self.cfg["experiment"].get("output_dir", "results"), self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> str:
        """Run full benchmark matrix and serialize raw metrics and summary tables."""
        print("\n" + "=" * 70)
        print(f"Launching Q-Swarm Benchmark Suite [{self.run_id}]")
        print(f"Mode: {'SMOKE (Fast)' if self.smoke else 'FULL'} | Seeds: {self.seeds}")
        print(f"Models: {self.models}")
        print(f"Scenarios: {self.scenarios}")
        print("=" * 70)

        results: List[ScenarioResult] = []
        raw_data: Dict[str, Dict[str, any]] = {}

        # 1. Main Policy Evaluation Matrix
        for model_name in self.models:
            raw_data[model_name] = {}
            ckpt = None
            if model_name == "classical":
                c = "results/checkpoints/classical/checkpoint.pt"
                if os.path.exists(c):
                    ckpt = c
            elif model_name == "classical_matched":
                c = "results/checkpoints/classical_matched/checkpoint.pt"
                if os.path.exists(c):
                    ckpt = c
            elif model_name in ["hybrid_vqc", "hybrid_vqc_entangled"]:
                c = "results/checkpoints/hybrid_vqc/checkpoint.pt"
                if os.path.exists(c):
                    ckpt = c

            train_time = 0.0
            if ckpt:
                meta_file = os.path.join(os.path.dirname(ckpt), "train_metrics.json")
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r") as f:
                            train_time = json.load(f).get("wall_clock_time", 0.0)
                    except Exception:
                        pass

            for scenario in self.scenarios:
                eval_out = evaluate_policy(
                    policy_type=model_name,
                    checkpoint=ckpt,
                    scenario=scenario,
                    seeds=self.seeds,
                    max_steps=self.max_steps,
                    encoding="graph_aware",
                    verbose=False,
                )
                summary = eval_out["summary"]
                raw_data[model_name][scenario] = eval_out

                res = ScenarioResult(
                    model_name=model_name,
                    scenario=scenario,
                    num_seeds=len(self.seeds),
                    trainable_parameters=summary["trainable_parameters"],
                    success_rate=summary["success_rate"],
                    mean_return=summary["mean_return"],
                    return_std=summary["return_std"],
                    mean_coverage=summary["mean_coverage"],
                    mean_collisions=summary["mean_collisions"],
                    mesh_connected_ratio=summary["mesh_connected_ratio"],
                    mean_latency_ms=summary["mean_latency_ms"],
                    train_time_s=train_time,
                )
                results.append(res)
                print(
                    f"  [DONE] {model_name:<28} | {scenario:<15} -> Return: {res.mean_return:>8.1f} | Cov: {res.mean_coverage:>5.1f}% | Conn: {res.mesh_connected_ratio:>5.1f}% | Latency: {res.mean_latency_ms:>5.2f}ms"
                )

        df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = os.path.join(self.output_dir, "benchmark_summary.csv")
        df.to_csv(csv_path, index=False)

        # 2. Quantum Graph-Aware vs Simple Angle Encoding Ablation
        print("\n" + "-" * 70)
        print("Running Quantum Feature Encoding Ablation (Graph-Aware vs Simple Angle)...")
        encoding_results = []
        for enc in ["graph_aware", "simple_angle"]:
            for sc in ["rescue_maze", "mesh_tradeoff"]:
                ev = evaluate_policy(
                    policy_type="hybrid_vqc_entangled",
                    scenario=sc,
                    seeds=self.seeds,
                    max_steps=self.max_steps,
                    encoding=enc,
                    verbose=False,
                )
                s = ev["summary"]
                encoding_results.append({
                    "encoding": enc,
                    "scenario": sc,
                    "mean_return": s["mean_return"],
                    "mesh_connected_ratio": s["mesh_connected_ratio"],
                    "mean_coverage": s["mean_coverage"],
                    "mean_latency_ms": s["mean_latency_ms"],
                })
                print(f"  [ENC] {enc:<14} | {sc:<15} -> Return: {s['mean_return']:>7.1f} | Conn: {s['mesh_connected_ratio']:>5.1f}% | Cov: {s['mean_coverage']:>5.1f}%")

        enc_df = pd.DataFrame(encoding_results)
        enc_csv = os.path.join(self.output_dir, "encoding_ablation.csv")
        enc_df.to_csv(enc_csv, index=False)

        # 3. QAOA Quantum Optimization Benchmark for Mesh Repair
        print("\n" + "-" * 70)
        print("Running QAOA Relay Selection & Mesh Partition Repair Benchmark...")
        qaoa_res = run_qaoa_benchmark(num_nodes=4)
        qaoa_json = os.path.join(self.output_dir, "qaoa_optimization_benchmark.json")
        with open(qaoa_json, "w") as f:
            json.dump({
                "nodes": qaoa_res.graph_nodes,
                "edges": qaoa_res.graph_edges,
                "exact_optimal_val": qaoa_res.exact_optimal_val,
                "exact_solution": qaoa_res.exact_solution,
                "exact_time_s": qaoa_res.exact_time_s,
                "greedy_val": qaoa_res.greedy_val,
                "greedy_solution": qaoa_res.greedy_solution,
                "greedy_time_s": qaoa_res.greedy_time_s,
                "qaoa_val": qaoa_res.qaoa_val,
                "qaoa_solution": qaoa_res.qaoa_solution,
                "qaoa_approx_ratio": qaoa_res.qaoa_approx_ratio,
                "qaoa_time_s": qaoa_res.qaoa_time_s,
                "qaoa_depth": qaoa_res.qaoa_depth,
                "qaoa_cx_count": qaoa_res.qaoa_cx_count,
            }, f, indent=2)
        print(f"  [QAOA] Ratio: {qaoa_res.qaoa_approx_ratio:.3f} | QAOA: {qaoa_res.qaoa_time_s*1000:.1f}ms | Greedy: {qaoa_res.greedy_time_s*1000:.2f}ms | Exact: {qaoa_res.exact_time_s*1000:.2f}ms")

        # 4. Serialize raw telemetry and update pointer
        raw_data["encoding_ablation"] = encoding_results
        raw_data["qaoa_benchmark"] = qaoa_json
        json_path = os.path.join(self.output_dir, "raw_metrics.json")
        with open(json_path, "w") as f:
            json.dump(raw_data, f, indent=2, default=str)

        latest_file = os.path.join("results", "latest_dir.txt")
        os.makedirs("results", exist_ok=True)
        with open(latest_file, "w") as f:
            f.write(self.output_dir)

        print("\n" + "=" * 70)
        print("Benchmark Suite Complete!")
        print(f"  Summary CSV   : {csv_path}")
        print(f"  Encoding CSV  : {enc_csv}")
        print(f"  QAOA JSON     : {qaoa_json}")
        print(f"  Raw JSON      : {json_path}")
        print("=" * 70 + "\n")

        return self.output_dir
