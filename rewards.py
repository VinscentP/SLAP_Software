import torch
from isaaclab.utils.math import quat_to_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REWARD_DECAY = -4.0     #can be tweaked for reward sensitivity
SELF_LEVEL_GAIN = 2.0
def stand_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    #Type hinting for object type RigidObject
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = quat_to_euler_xyz(quat)

    return torch.exp(REWARD_DECAY * (roll**2 + pitch**2))

#reward being at a certain height to avoid body getting to low to floor
def height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, desired_height) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    
    _, _, height = asset.data.root_state_w[:, :3]

    height_error = height - desired_height

    return torch.exp(-2.0 * height_error**2)
    
#minimize angular velocity and penalize sudden movement
def self_leveling_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
 
    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = quat_to_euler_xyz(quat)

    wx, wy, _ = asset.data.root_state_w[:, 10:13]
    
    corrective_signal = -(roll * wx + pitch * wy)
    
    return SELF_LEVEL_GAIN * torch.tanh(corrective_signal)

#reward forward velocity
# v_world ​= q_world​⋅v_body​⋅q_world^(−1​)
def forward_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    
    quat = asset.data.root_state_w[:, 3:7]
    vel_world  = asset.data.root_state_w[:, 7:10]

    vel_body = quat_rotate_inverse(quat, vel_world)

    v_forward = vel_body[:, 0]

    return torch.tanh(v_forward)


 

#reward going at a certain velocity based on 

#reward yaw rate/orientation changes

#fall penalty