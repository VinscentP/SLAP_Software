#ADD TO mdp.py file
import torch
from isaaclab.assets import Articulation

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REWARD_DECAY = -4.0     #can be tweaked for reward sensitivity
SELF_LEVEL_GAIN = 2.0
def orientation_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    #Type hinting for object type RigidObject
    asset: Articulation = env.scene[asset_cfg.name] 

    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = euler_xyz_from_quat(quat)

    return torch.exp(REWARD_DECAY * (roll**2 + pitch**2))

#reward being at a certain height to avoid body getting to low to floor
def height_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, desired_height) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    height = asset.data.root_state_w[:, :2]
    height_error = height - desired_height

    return torch.exp(-2.0 * height_error**2)
    
#minimize angular velocity and penalize sudden movement
def self_leveling_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
 
    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = euler_xyz_from_quat(quat)

    wx = asset.data.root_state_w[:, 10]  # Roll rate
    wy = asset.data.root_state_w[:, 11]  # Pitch rate
    
    corrective_signal = -(roll * wx + pitch * wy)
    
    return SELF_LEVEL_GAIN * torch.tanh(corrective_signal)


#fall penalty
def fall_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, min_height) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    #height
    z = asset.data.root_state_w[:,2]
    height_penalty = (z < min_height).float()
    
    #angular velocity roll
    wx = asset.data.root_state_w[:, 10]  # Roll rate
    wy = asset.data.root_state_w[:, 11]  # Pitch rate
    angular_velocity_penalty = (wx**2 + wy**2)
    
    #linear velocity z
    vz = asset.data.root_state_w[:, 9]
    linear_velocity_penalty = torch.relu(-vz)
    
    #Can change weights for fine tuning 
    penalty = 10.0 * height_penalty + 0.1 * angular_velocity_penalty + 0.5 * linear_velocity_penalty

    return -penalty



#reward yaw rate/orientation changes
