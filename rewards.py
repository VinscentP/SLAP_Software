import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


REWARD_DECAY = 2.5
WARMUP_SECONDS = 5.0
TARGET_HEIGHT = 0.45
GROUND_CLEARANCE = 0.3


def _warmup_scale(env: "ManagerBasedRLEnv") -> torch.Tensor:
    elapsed = env.episode_length_buf * env.step_dt
    return torch.clamp(elapsed / WARMUP_SECONDS, max=1.0)


def upright_bonus(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = euler_xyz_from_quat(quat)
    z = asset.data.root_state_w[:, 2]

    orientation_ok = torch.exp(-REWARD_DECAY * (roll**2 + pitch**2))
    height_ok = torch.exp(-8.0 * (z - TARGET_HEIGHT)**2)

    return orientation_ok * height_ok * _warmup_scale(env)


def recovery_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_state_w[:, 2]
    vz = asset.data.root_state_w[:, 9]

    below_target = (z < TARGET_HEIGHT - 0.05).float()
    upward_velocity = torch.relu(vz)

    return below_target * upward_velocity * 2.0


def fall_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_state_w[:, 2]

    height_penalty = (z < min_height).float()
    wx = asset.data.root_state_w[:, 10]
    wy = asset.data.root_state_w[:, 11]
    angular_velocity_penalty = torch.clamp(wx**2 + wy**2, max=10.0)
    vz = asset.data.root_state_w[:, 9]
    linear_velocity_penalty = torch.relu(-vz)

    penalty = 10.0 * height_penalty + 0.1 * angular_velocity_penalty + 0.5 * linear_velocity_penalty
    return -penalty


