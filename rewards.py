import torch
from isaaclab.utils.math import quat_to_euler_xyz
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REWARD_DECAY = -4.0     #can be tweaked for reward sensitivity

def stand_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    #Type hinting for object type RigidObject
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = quat_to_euler_xyz(quat)

    reward = torch.exp(REWARD_DECAY* (roll**2 + pitch**2))

    return reward

#minimize angular velocity and penalize sudden movement
def angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    wx, wy, wz = asset.data.root_state_w[:, 10:13]
    
    reward = torch.exp(REWARD_DECAY * (wx**2 + wy**2))
    
    return reward

#reward velocity in desired direction

#reward function to harshly penalize falling during early testing