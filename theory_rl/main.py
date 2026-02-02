"""CLI entry point for the theory-based RL model.

Usage:
    python -m theory_rl.main --all        # Generate all figures
    python -m theory_rl.main --fig 3      # Generate a specific figure
    python -m theory_rl.main --validate   # Run IS vs exact posterior validation
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")


def run_validate():
    from .core import validate_is_vs_exact

    print("Running IS vs exact categorical posterior validation...")
    result = validate_is_vs_exact(n_samples=50000)
    print(f"  Exact posterior:  {result['exact_posterior']}")
    print(f"  IS posterior:     {result['is_posterior']}")
    print(f"  Max error:        {result['max_error']:.6f}")

    if result["max_error"] < 0.02:
        print("PASS: Max error < 0.02")
    else:
        print(f"FAIL: Max error {result['max_error']:.6f} >= 0.02")

    return result["max_error"] < 0.02


def run_fig(n):
    from .figs import ALL_PLOTS

    if n not in ALL_PLOTS:
        print(f"Unknown figure {n}. Available: {sorted(ALL_PLOTS.keys())}")
        sys.exit(1)
    print(f"Generating Figure {n}...")
    ALL_PLOTS[n]()


def run_all():
    from .figs import plot_all
    plot_all()


def main():
    parser = argparse.ArgumentParser(
        description="Theory-based RL as a purely functional fold (Tomov et al. 2023)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures")
    parser.add_argument("--fig", type=int,
                        help="Generate a specific figure (1-6)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate IS vs exact categorical posterior")
    args = parser.parse_args()

    if not (args.all or args.fig or args.validate):
        parser.print_help()
        sys.exit(1)

    if args.validate:
        success = run_validate()
        if not success:
            sys.exit(1)

    if args.fig:
        run_fig(args.fig)

    if args.all:
        run_all()


if __name__ == "__main__":
    main()
