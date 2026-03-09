import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


REWARD_DECAY = 2.5          # sharper orientation penalty
WARMUP_SECONDS = 5.0
TARGET_HEIGHT = 0.45        # tune to your robot's standing height
GROUND_CLEARANCE = 0.3      # min acceptable height before fall penalty kicks in


def _warmup_scale(env: "ManagerBasedRLEnv") -> torch.Tensor:
    elapsed = env.episode_length_buf * env.step_dt
    return torch.clamp(elapsed / WARMUP_SECONDS, max=1.0)


def orientation_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sharper Gaussian — small tilts still get penalized meaningfully."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = euler_xyz_from_quat(quat)
    reward = torch.exp(-REWARD_DECAY * (roll**2 + pitch**2))
    return reward * _warmup_scale(env)


def base_height_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Gaussian centered on TARGET_HEIGHT instead of tanh.
    Robot gets max reward only near the target — too high or too low is penalized.
    This prevents the tanh saturation that made sprawling acceptable.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_state_w[:, 2]
    height_error = z - TARGET_HEIGHT
    return torch.exp(-8.0 * height_error**2)  # tight Gaussian around target


def upright_bonus(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Joint reward: must be BOTH upright AND at target height simultaneously.
    Prevents gaming one without the other.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_state_w[:, 3:7]
    roll, pitch, _ = euler_xyz_from_quat(quat)
    z = asset.data.root_state_w[:, 2]

    orientation_ok = torch.exp(-REWARD_DECAY * (roll**2 + pitch**2))
    height_ok = torch.exp(-8.0 * (z - TARGET_HEIGHT)**2)

    return orientation_ok * height_ok * _warmup_scale(env)


def recovery_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Explicitly rewards upward velocity when below target height.
    This is the key signal that teaches the robot to push itself back up.
    When z < TARGET_HEIGHT, vz > 0 (moving up) gives positive reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_state_w[:, 2]
    vz = asset.data.root_state_w[:, 9]

    # Only active when below target height
    below_target = (z < TARGET_HEIGHT - 0.05).float()
    upward_velocity = torch.relu(vz)  # only reward moving up, not penalize moving down

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


def undesired_body_contact_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Penalizes contact forces on non-foot bodies (shins, knees, torso).
    You need a ContactSensor covering your lower leg / shin bodies.
    In your env cfg, set sensor_cfg to cover those bodies explicitly.

    Example env cfg sensor:
        undesired_contact_sensor: ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_lower_leg",  # adjust to your URDF link names
            track_air_time=False,
        )
    """
    from isaaclab.sensors import ContactSensor
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Net contact force magnitude per body, shape: (num_envs, num_bodies)
    contact_forces = torch.norm(sensor.data.net_forces_w, dim=-1)

    # Any body in the sensor over threshold counts as undesired contact
    has_contact = (contact_forces > threshold).float()
    penalty = torch.sum(has_contact, dim=1)  # count of offending bodies

    return -penalty * 0.5


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    penalty = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
    return -penalty * _warmup_scale(env)


def joint_vel_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.joint_vel), dim=1) * _warmup_scale(env)


