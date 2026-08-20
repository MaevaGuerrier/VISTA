import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points, compare_waypoints_pred_to_label
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy, numpy_to_img, RED, GREEN, CYAN, MAGENTA
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from timm.scheduler import CosineLRScheduler

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: Optional[torch.Tensor] = None,
    return_per_sample: bool = False,
    distance_loss_coeff: float = 0.01,
    action_loss_type: str = "mse",
    distance_loss_type: str = "mse",
    metric_waypoint_spacing: Optional[torch.Tensor] = None,
):
    """
    Compute losses for distance and action prediction.

    Args:
        return_per_sample: if True, also return per-sample total losses
        distance_loss_coeff: coefficient to multiply the distance loss (default: 0.01)
        action_loss_type: type of action loss to use ("mse", "mape", "waypoint_spacing_scaled_mse")
        distance_loss_type: type of distance loss to use ("mse", "waypoint_spacing_scaled_mse")
        metric_waypoint_spacing: per-sample metric waypoint spacing, required for "waypoint_spacing_scaled_mse"
    """
    # Compute distance loss based on distance_loss_type
    if distance_loss_type == "mse":
        dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())
    elif distance_loss_type == "waypoint_spacing_scaled_mse":
        # Scale distance residuals by metric waypoint spacing to equally penalize errors
        # across datasets with different waypoint spacings. Divide residual by spacing
        # so that denser datasets (smaller spacing) get scaled up.
        assert metric_waypoint_spacing is not None, "metric_waypoint_spacing is required for waypoint_spacing_scaled_mse"
        scaled_dist_label = dist_label.float() / metric_waypoint_spacing
        scaled_dist_pred = dist_pred.squeeze(-1) / metric_waypoint_spacing
        dist_loss = F.mse_loss(scaled_dist_pred, scaled_dist_label)
    else:
        raise ValueError(f"Unsupported distance_loss_type: {distance_loss_type}")

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    def action_reduce_per_sample(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask) / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    if action_loss_type == "mse":
        unreduced_loss = F.mse_loss(action_pred, action_label, reduction="none")
    elif action_loss_type == "mape":
        unreduced_loss = F.l1_loss(action_pred, action_label, reduction="none") / (torch.abs(action_label) + 1e-2)
    elif action_loss_type == "waypoint_spacing_scaled_mse":
        # Scale residuals by metric waypoint spacing to equally penalize errors across datasets
        # with different waypoint spacings. Divide residual by spacing so that denser datasets
        # (smaller spacing) get scaled up and sparser datasets get scaled down.
        assert metric_waypoint_spacing is not None, "metric_waypoint_spacing is required for waypoint_spacing_scaled_mse"
        # Reshape spacings for broadcasting: [B] -> [B, 1, 1]
        spacings = metric_waypoint_spacing.view(-1, 1, 1)
        # Scale the action residuals (for x, y dimensions; angle is left as-is)
        scaled_action_label = action_label.clone()
        scaled_action_pred = action_pred.clone()
        # Only scale the x, y coordinates (first 2 dimensions), not the angle
        scaled_action_label[:, :, :2] = scaled_action_label[:, :, :2] / spacings
        scaled_action_pred[:, :, :2] = scaled_action_pred[:, :, :2] / spacings
        unreduced_loss = F.mse_loss(scaled_action_pred, scaled_action_label, reduction="none")
    else:
        raise ValueError(f"Unsupported action_loss_type: {action_loss_type}")
    action_loss = action_reduce(unreduced_loss)

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * distance_loss_coeff * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    if return_per_sample:
        # Compute per-sample losses for high-loss detection
        dist_loss_per_sample = F.mse_loss(dist_pred.squeeze(-1), dist_label.float(), reduction='none')
        action_loss_per_sample = action_reduce_per_sample(F.mse_loss(action_pred, action_label, reduction="none"))
        total_loss_per_sample = alpha * distance_loss_coeff * dist_loss_per_sample + (1 - alpha) * action_loss_per_sample
        results["total_loss_per_sample"] = total_loss_per_sample

    return results


def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if print_log_freq != 0 and i % print_log_freq == 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if print_log_freq != 0 and i % print_log_freq == 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def _log_high_loss_samples(
    obs_images: torch.Tensor,
    goal_images: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    dist_label: torch.Tensor,
    total_loss_per_sample: torch.Tensor,
    high_loss_indices: torch.Tensor,
    dataset_index: torch.Tensor,
    epoch: int,
    batch_idx: int,
    project_folder: str,
    normalized: bool,
    mode: str = "train",
    max_samples: int = 10,
    use_wandb: bool = True,
):
    """
    Log samples with unusually high loss for debugging.
    Uses existing visualization utilities from vint_train.visualizing.
    """
    num_to_log = min(len(high_loss_indices), max_samples)
    log_dir = os.path.join(project_folder, f"high_loss_{mode}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    obs_images_np = to_numpy(obs_images)
    goal_images_np = to_numpy(goal_images)
    action_pred_np = to_numpy(action_pred)
    action_label_np = to_numpy(action_label)
    dataset_index_np = to_numpy(dataset_index)
    
    # Denormalize images (undo ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if obs_images_np.shape[1] > 3:
        # If there are multiple observation images concatenated, only denormalize the last 3 channels (the most recent observation)
        obs_images_np = obs_images_np[:, -3:, :, :]
    obs_images_np = obs_images_np * std[None, :, None, None] + mean[None, :, None, None]
    goal_images_np = goal_images_np * std[None, :, None, None] + mean[None, :, None, None]
    obs_images_np = np.clip(obs_images_np, 0, 1)
    goal_images_np = np.clip(goal_images_np, 0, 1)
    
    wandb_list = []
    
    for idx_in_batch, sample_idx in enumerate(high_loss_indices[:num_to_log]):
        sample_idx = sample_idx.item()
        loss_value = total_loss_per_sample[sample_idx].item()
        
        # Get data for this sample
        obs_img = obs_images_np[sample_idx]
        goal_img = goal_images_np[sample_idx]
        pred_waypoints = action_pred_np[sample_idx]
        label_waypoints = action_label_np[sample_idx]
        dataset_idx = int(dataset_index_np[sample_idx])
        
        # Get dataset name from data_config (already loaded at module level)
        dataset_names = sorted(list(data_config.keys()))
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else "unknown"
        
        # Use existing visualization function
        save_path = os.path.join(log_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_idx}_loss{loss_value:.4f}.png")
        
        compare_waypoints_pred_to_label(
            obs_img=numpy_to_img(obs_img),
            goal_img=numpy_to_img(goal_img),
            dataset_name=dataset_name,
            goal_pos=np.array([0, 0]),  # Placeholder, actual goal pos not directly available
            pred_waypoints=pred_waypoints,
            label_waypoints=label_waypoints,
            save_path=save_path,
            display=False,
        )
        
        if use_wandb:
            wandb_list.append(wandb.Image(save_path, caption=f"Loss: {loss_value:.4f}"))
    
    # Log all high-loss samples to wandb at once
    if use_wandb and wandb_list:
        wandb.log({
            f"{mode}/high_loss_samples": wandb_list,
            f"{mode}/high_loss_count": len(wandb_list),
            f"{mode}/high_loss_threshold": total_loss_per_sample[high_loss_indices[0]].item(),
        }, commit=False)

def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    scheduler: Optional[CosineLRScheduler] = None,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
    log_high_loss_samples: bool = False,
    high_loss_threshold: float = 10.0,
    max_high_loss_samples: int = 10,
    distance_loss_coeff: float = 0.01,
    action_loss_type: str = "mse",
    distance_loss_type: str = "mse",
    pass_action_history: bool = False
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
        log_high_loss_samples: whether to log samples with high loss
        high_loss_threshold: threshold for considering a loss as high
        max_high_loss_samples: maximum number of high loss samples to log
        distance_loss_coeff: coefficient to multiply the distance loss (default: 0.01)
        action_loss_type: type of action loss to use ("mse", "mape", "waypoint_spacing_scaled_mse")
        distance_loss_type: type of distance loss to use ("mse", "waypoint_spacing_scaled_mse")
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
            metric_waypoint_spacing,
            action_history,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        action_history = action_history.to(device)
        if pass_action_history:
            model_outputs = model(obs_image, goal_image, action_history)
        else:
            model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)
        metric_waypoint_spacing = metric_waypoint_spacing.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
            return_per_sample=log_high_loss_samples,
            distance_loss_coeff=distance_loss_coeff,
            action_loss_type=action_loss_type,
            distance_loss_type=distance_loss_type,
            metric_waypoint_spacing=metric_waypoint_spacing,
        )

        # Log high-loss samples if enabled
        if log_high_loss_samples and "total_loss_per_sample" in losses:
            total_loss_per_sample = losses["total_loss_per_sample"]
            high_loss_mask = total_loss_per_sample > high_loss_threshold
            if high_loss_mask.any():
                high_loss_indices = torch.where(high_loss_mask)[0]
                print(f"[Epoch {epoch} Batch {i}] Found {len(high_loss_indices)} high-loss samples (threshold: {high_loss_threshold})")
                _log_high_loss_samples(
                    obs_images=obs_image,
                    goal_images=goal_image,
                    action_pred=action_pred,
                    action_label=action_label,
                    dist_pred=dist_pred,
                    dist_label=dist_label,
                    total_loss_per_sample=total_loss_per_sample,
                    high_loss_indices=high_loss_indices,
                    dataset_index=dataset_index,
                    epoch=epoch,
                    batch_idx=i,
                    project_folder=project_folder,
                    normalized=normalized,
                    mode="train",
                    max_samples=max_high_loss_samples,
                    use_wandb=use_wandb,
                )

        losses["total_loss"].backward()
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, CosineLRScheduler):
            scheduler.step(epoch + i / num_batches)

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
    distance_loss_coeff: float = 0.01,
    action_loss_type: str = "mse",
    distance_loss_type: str = "mse",
    pass_action_history: bool = False
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
        distance_loss_coeff: coefficient to multiply the distance loss (default: 0.01)
        action_loss_type: type of action loss to use ("mse", "mape", "waypoint_spacing_scaled_mse")
        distance_loss_type: type of distance loss to use ("mse", "waypoint_spacing_scaled_mse")
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
                metric_waypoint_spacing,
                action_history,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            
            action_history = action_history.to(device)
            if pass_action_history:
                model_outputs = model(obs_image, goal_image, action_history)
            else:
                model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)
            metric_waypoint_spacing = metric_waypoint_spacing.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
                distance_loss_coeff=distance_loss_coeff,
                action_loss_type=action_loss_type,
                distance_loss_type=distance_loss_type,
                metric_waypoint_spacing=metric_waypoint_spacing,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    print(stats['max'], stats['min'])
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)

