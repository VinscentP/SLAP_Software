# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math
import carb
from .mdp.rewards import orientation_reward


NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
"""Path to the root directory on the Nucleus Server."""


NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""


ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.assets import ArticulationCfg as ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg


from . import mdp


##
# Pre-defined configs
##


#from .ur_gripper import UR_GRIPPER_CFG  # isort:skip




##
# Scene definition
##




@configclass
class BostonrlPhamSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene."""


    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )


    # robot
    robot = ArticulationCfg(
        prim_path="/World/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Users/danym/Documents/issac_sim_projects/boston_pham.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "fl_kn": -1.0,   # within [-2.793, -0.247]
                "fr_kn": -1.0,
                "hl_kn": -1.0,
                "hr_kn": -1.0,
            }
        ),
        actuators={
            "robot_legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=60.0,
                damping=6.0,
                effort_limit_sim=80.0,
            ),
        },
    )
   
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )


    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP settings
##




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


    robot_action: ActionTerm = mdp.JointPositionActionCfg(
        use_default_offset=False,
        asset_name="robot",
        debug_vis=True,
        clip= {
            ".*_hx": (-0.785, 0.785),
            ".*_hy": (-1.0, 2.3),
            ".*_kn": (-2.79, -0.247),
        },                 #may have to clip knee joint so it doesn't overextend
     
        joint_names=
            ["fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",],
        scale=.5,
        offset={  # start knees in the middle of the safe range
            "fl_kn": -1.5,
            "fr_kn": -1.5,
            "hl_kn": -1.5,
            "hr_kn": -1.5,
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    #NullCommandCfg
    null_command = mdp.NullCommandCfg(
        resampling_time_range=(1,7),
    )
    #UniformVelocityCommandCfg
    velocity_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(1,7),
        rel_standing_envs=0.1,
        heading_command=True,
        rel_heading_envs=0.3,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""


        # observation terms (order preserved)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})
        actions = ObsTerm(func=mdp.last_action)
        #height_scan
        #imu_*


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()






@configclass
class EventCfg:
    """Configuration for events."""
    random_forces = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=[4.0, 6.0],
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-50.0, 50.0),      
            "torque_range": (-20.0, 20.0),    
    },
)


    #reset_joints = EventTerm(
    #    func=mdp.reset_joints_by_offset,
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot"),
    #        "position_range": (-0.05, 0.05),
    #        "velocity_range": (-0.05, 0.05),
    #    },
    #)
   


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    orientation_reward = RewTerm(
        func=mdp.orientation_reward, weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    self_leveling_reward = RewTerm(
        func=mdp.self_leveling_reward, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    height_reward = RewTerm(
        func=mdp.height_reward, weight=0.5,
         params={
            "asset_cfg": SceneEntityCfg("robot"),  
            "desired_height": 0.3,                      #CHECK default height position              
        },
    )
    fall_penalty = RewTerm(
        func=mdp.fall_penalty, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.15,                  
        },
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""


    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_height = DoneTerm(func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        })


##
# Environment configuration
##


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


    orientation_reward = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "orientation_reward", "weight": 2.0, "num_steps": 4500},
    )


    height_reward = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "height_reward", "weight": 1.0, "num_steps": 4500},
    )


@configclass
class BostonrlPhamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""


    # Scene settings - how many robots, how far apart?
    scene = BostonrlPhamSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4                         #after decimation number of physics steps policy outputs action
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 240.0                   #around 4.17 ms physics step


@configclass
class BostonrlPhamEnvCfg_PLAY(BostonrlPhamEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

