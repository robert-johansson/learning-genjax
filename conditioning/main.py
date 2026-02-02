"""CLI entry point for the conditioning model.

Usage:
    python -m conditioning.main --all        # Generate all figures
    python -m conditioning.main --fig 4      # Generate a specific figure
    python -m conditioning.main --validate   # Run GenJAX IS vs analytical validation
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")


def run_validate():
    from .core import validate_genjax_vs_analytical

    print("Running IS vs analytical Gamma-Poisson posterior validation...")
    result = validate_genjax_vs_analytical(n_samples=50000)
    print(f"  Analytical mean: {result['analytical_mean']:.4f}")
    print(f"  IS mean:         {result['is_mean']:.4f}")
    print(f"  Mean error:      {result['mean_error']:.4f}")
    print(f"  Analytical var:  {result['analytical_var']:.4f}")
    print(f"  IS var:          {result['is_var']:.4f}")
    print(f"  Var error:       {result['var_error']:.4f}")

    if result["mean_error"] < 0.05:
        print("PASS: Mean error < 0.05")
    else:
        print(f"FAIL: Mean error {result['mean_error']:.4f} >= 0.05")

    return result["mean_error"] < 0.05


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
        description="Classical conditioning as Bayesian rate estimation (Gershman 2025)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures")
    parser.add_argument("--fig", type=int,
                        help="Generate a specific figure (2-9)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate GenJAX IS vs analytical posterior")
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
