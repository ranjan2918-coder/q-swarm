"""Command Line Interface for Q-Swarm RL."""

import argparse
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qswarm",
        description=(
            "Q-Swarm RL: Hybrid Quantum-Reinforcement Learning for "
            "Real-Time Swarm Pathfinding & Dynamic Mesh Optimization in GPS-Denied Environments."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
        help="Show program version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # validate command
    val_parser = subparsers.add_parser("validate", help="Validate installation and dependencies")
    val_parser.add_argument("--verbose", action="store_true", help="Print verbose environment details")

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run a simulation demo")
    demo_parser.add_argument(
        "--policy",
        choices=["heuristic", "classical", "classical_matched", "hybrid_vqc"],
        default="heuristic",
        help="Policy to execute (default: heuristic)",
    )
    demo_parser.add_argument(
        "--scenario",
        default="rescue_maze",
        help="Scenario name (open_field, rescue_maze, dynamic_hazard, mesh_tradeoff, scale_test)",
    )
    demo_parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--smoke", action="store_true", help="Run quick smoke benchmark")
    bench_parser.add_argument(
        "--config", default="configs/experiments/benchmark.yaml", help="Path to benchmark config"
    )

    # dashboard command
    subparsers.add_parser("dashboard", help="Launch Streamlit interactive dashboard")

    return parser


def validate_environment(verbose: bool = False) -> int:
    """Validate core libraries and runtime capabilities."""
    print("=" * 60)
    print("Q-Swarm RL System Validation")
    print("=" * 60)

    modules = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("networkx", "NetworkX"),
        ("gymnasium", "Gymnasium"),
        ("qiskit", "Qiskit"),
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
    ]

    all_ok = True
    for mod_name, label in modules:
        try:
            m = __import__(mod_name)
            ver = getattr(m, "__version__", "unknown")
            print(f"[OK] {label:<15} (version: {ver})")
        except ImportError as e:
            print(f"[FAILED] {label:<15} Error: {e}")
            all_ok = False

    if all_ok:
        print("\nAll core dependencies are satisfied and operational.")
        return 0
    else:
        print("\nSome dependencies are missing. Run: pip install -e .[dev]")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    if args.subcommand == "validate":
        return validate_environment(verbose=args.verbose)
    elif args.subcommand == "demo":
        if args.policy == "heuristic":
            from qswarm.policies.heuristic import run_heuristic_demo
            run_heuristic_demo(scenario=args.scenario, seed=args.seed)
        else:
            from qswarm.training.evaluate import evaluate_policy
            evaluate_policy(
                policy_type=args.policy,
                scenario=args.scenario,
                seeds=[args.seed],
                verbose=True,
            )
        return 0
    elif args.subcommand == "benchmark":
        from qswarm.experiments.runner import BenchmarkRunner
        runner = BenchmarkRunner(config_path=args.config, smoke=args.smoke)
        runner.run()
        return 0
    elif args.subcommand == "dashboard":
        import subprocess
        print("Launching Q-Swarm Streamlit dashboard on http://localhost:8501 ...")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "src/qswarm/dashboard/app.py"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
