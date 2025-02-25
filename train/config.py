import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import argparse

def get_args():
    """
    Defines and parses command-line arguments for experimentation.
    """
    parser = argparse.ArgumentParser(description="U-Net Training Experiments")

    # Experiment configuration
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=250, help="Patch size for cropping")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--scheduler", type=str, default="StepLR", choices=["StepLR", "ReduceLROnPlateau"], help="Scheduler type")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--batch_norm", action="store_true", help="Use Batch Normalization")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained encoder")

    return parser.parse_args()
