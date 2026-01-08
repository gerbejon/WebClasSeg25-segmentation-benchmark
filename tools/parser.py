# cli_args.py
import argparse

def get_common_parser(description: str = "WebClasSeg25 model script"):
    """Return a base ArgumentParser with common training arguments."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--classification",
        choices=["fc", "mc"],
        required=True,
        help="Classification type: 'fc' (functional) or 'mc' (maturity). Default: fc. required argument",
    )

    parser.add_argument(
        "--modelversion",
        choices=["v1", "v2"],
        default='v1',
        help="Model version: v1 (tag_head + xpath), v2 (reserved). Default: v1.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Default: 3.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per device for training and evaluation. Default: 4.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default: 2e-5.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory for model checkpoints and logs.",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to Hugging Face Hub after training.",
    )

    return parser


def parse_args(description: str = "WebClasSeg25 model training script"):
    """Parse arguments and return them as a Namespace."""
    parser = get_common_parser(description)
    args = parser.parse_args()
    return args