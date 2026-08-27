import argparse
from pathlib import Path

from pinn_project.applications.burgers.plotting import (
    load_burgers_prediction_fields,
    plot_burgers_predictions,
    relative_l2_error,
)


def _parser() -> argparse.ArgumentParser:
    """Build the Burgers prediction-plot command-line parser.

    Returns:
        Parser accepting an artifact path, optional output path, and interactive-display flag.
    """
    parser = argparse.ArgumentParser(
        description="Plot Burgers reference, prediction, and absolute error fields from a PhiJAX artifact."
    )
    parser.add_argument("predictions", type=Path, help="PhiJAX Burgers prediction `.npz` artifact.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Figure path; defaults beside the artifact as `.png`."
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving it.")
    return parser


def main() -> None:
    """Plot the selected artifact and report its relative L2 error and figure path."""
    arguments = _parser().parse_args()
    reference, prediction, _, _ = load_burgers_prediction_fields(arguments.predictions)
    error = relative_l2_error(reference, prediction)
    output_path = plot_burgers_predictions(arguments.predictions, arguments.output, show=arguments.show)
    print(f"Relative L2 error: {error:.2e}")
    print(f"Saved Burgers comparison figure: {output_path}")


if __name__ == "__main__":
    main()
