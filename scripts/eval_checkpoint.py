from gr00t.utils.eval import calc_mse_for_single_trajectory
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
import torch
import warnings
import re

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up saving path
finetuned_model_path = "/home/namikosaito/work2/Isaac-GR00T/eval/finetune-0/checkpoint-1500"

# Extract epoch number from checkpoint path
epoch_match = re.search(r'checkpoint-(\d+)', finetuned_model_path)
epoch_number = epoch_match.group(1) if epoch_match else "unknown"

# Create plot filename with epoch number
save_plot_path = finetuned_model_path + f"/../trajectory_plot-{epoch_number}.png"

# Load data config and set up modality config
data_config = load_data_config("fourier_gr1_arms_only")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

# Set up dataset path and embodiment tag
dataset_path = "../demo_data/robot_sim.PickNPlace/"
embodiment_tag = EmbodimentTag.GR1

# Create dataset
dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=embodiment_tag,
)

finetuned_policy = Gr00tPolicy(
    model_path=finetuned_model_path,
    embodiment_tag="new_embodiment",
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

warnings.simplefilter("ignore", category=FutureWarning)

mse = calc_mse_for_single_trajectory(
    finetuned_policy,
    dataset,
    traj_id=0,
    modality_keys=["right_arm", "right_hand"],   # we will only evaluate the right arm and right hand
    steps=150,
    action_horizon=16,
    plot=True,
    save_plot_path=save_plot_path
)

print("MSE loss for trajectory 0:", mse)
print(f"Plot saved to: {save_plot_path}")