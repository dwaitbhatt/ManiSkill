from typing import Any, List, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Reachy2
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
import transforms3d as t3d


@register_env("ScrewBulb-v1", max_episode_steps=50)
class ScrewBulbEnv(BaseEnv):
    """
    **Task Description:**
    A task where the objective is to screw a bulb.

    **Randomizations:**
    - TBD

    **Success Conditions:**
    - TBD
    """

    _sample_video_link = ""
    SUPPORTED_ROBOTS = [
        "reachy2"
    ]
    SUPPORTED_REWARD_MODES = ["none"]

    agent: Union[Reachy2]

    def __init__(self, *args, robot_uids="reachy2", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        base_camera = CameraConfig("base_camera", pose, 1024, 1024, np.pi / 2, 0.01, 100, shader_pack="rt")
        
        pose = sapien.Pose(p=[0.1, -1.1, 0.5], q=[0.825787, -0.087173, 0.13304, 0.541089])
        gsplat_camera = CameraConfig("gsplat_camera", pose=pose, width=1848, height=1016, fov=0.9, near=0.1, far=1e+03, shader_pack="rt")
        return [base_camera, gsplat_camera]
        
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 0.7, 0.5], [0.0, 0.1, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_bulb(self, color="orange"):
        color_map = {
            "blue":    [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1],
            "orange":  [1.0, 0.4980392156862745, 0.054901960784313725, 1],
            "green":   [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1],
            "red":     [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1],
            "yellow":  [1.0, 0.8941176470588236, 0.0, 1],
            "purple":  [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1],
            "brown":   [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1],
            "pink":    [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1],
            "gray":    [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1],
            "olive":   [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1],
            "cyan":    [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1],
        }
        self.bulb_model_path = str(PACKAGE_ASSET_DIR / "custom/bulb.obj")

        self.bulb_builder = self.scene.create_actor_builder()
        bulb_mat = sapien.render.RenderMaterial()
        bulb_mat.set_base_color(color_map[color])
        bulb_mat.metallic = 0.0
        bulb_mat.roughness = 1.0
        bulb_mat.specular = 1.0
        self.bulb_builder.add_visual_from_file(
            self.bulb_model_path,
            scale = [0.04]*3,
            material=bulb_mat
        )
        # self.bulb_builder.add_convex_collision_from_file(self.bulb_model_path)
        self.bulb_builder.add_convex_collision_from_file(
            filename=self.bulb_model_path,
            scale=[0.04]*3,
            density=5000
        )
        self.bulb_builder.initial_pose = sapien.Pose(p=[-0.0034, -0.0151, 0.0259], q=[0.9861, 0.1506, -0.0689, 0.0141])
        bulb = self.bulb_builder.build(name=f"bulb_{color}")
        return bulb

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.bulbs: List[Actor] = []
        self.bulbs.append(self._build_bulb(color="red"))
        self.bulbs.append(self._build_bulb(color="green"))
        self.bulbs.append(self._build_bulb(color="yellow"))

        model_ids = ["035_power_drill", "043_phillips_screwdriver", "044_flat_screwdriver", "048_hammer", "042_adjustable_wrench"]
        self.ycb_objects: List[Actor] = []
        for model_id in model_ids:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            self.ycb_objects.append(builder.build(name=f"ycb_{model_id}"))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Define the overall area on the table for bulb placement
            bulb_area_x_min = 0.1
            bulb_area_x_max = 0.3
            bulb_area_y_min = -0.2
            bulb_area_y_max = 0.2
            bulb_area_z = 0.026  # height above the table
            
            # Divide the area into three non-overlapping rectangles for bulbs
            rectangle_width = (bulb_area_x_max - bulb_area_x_min) / 3
            
            # Define the three rectangle areas (x_min, x_max, y_min, y_max) for bulbs
            bulb_rectangles = [
                [bulb_area_x_min, bulb_area_x_min + rectangle_width, bulb_area_y_min, bulb_area_y_max],  # Left rectangle
                [bulb_area_x_min + rectangle_width, bulb_area_x_min + 2*rectangle_width, bulb_area_y_min, bulb_area_y_max],  # Middle rectangle
                [bulb_area_x_min + 2*rectangle_width, bulb_area_x_max, bulb_area_y_min, bulb_area_y_max]  # Right rectangle
            ]
            
            # For each bulb, generate positions within its designated rectangle for all environments
            for bulb_idx, bulb in enumerate(self.bulbs):
                rect = bulb_rectangles[bulb_idx]
                
                # Generate random positions for all environments in the batch
                x = torch.rand(b, device=self.device) * (rect[1] - rect[0]) + rect[0]
                y = torch.rand(b, device=self.device) * (rect[3] - rect[2]) + rect[2]
                z = torch.full((b,), bulb_area_z, device=self.device)
                
                # Create position vectors for all environments
                positions = torch.stack([x, y, z], dim=1)
                
                # Get current orientation (quaternion) and expand for all environments
                current_pose = bulb.pose
                q = torch.tensor(
                    [current_pose.q[:, 0], current_pose.q[:, 1], current_pose.q[:, 2], current_pose.q[:, 3]], 
                    device=self.device
                ).expand(b, 4)
                
                # Set new poses with randomized positions but original orientations
                bulb.set_pose(Pose.create_from_pq(positions, q))
            
            # Define a different area on the table for YCB objects
            # Using the provided table xy half sizes (1.2, 0.6)
            ycb_area_x_min = -0.1  # Different area from bulbs
            ycb_area_x_max = 0.1   # Up to where bulbs start
            ycb_area_y_min = -0.5
            ycb_area_y_max = 0.5
            ycb_area_z = 0.02  # Height above the table
            
            # Determine the layout based on number of YCB objects
            num_ycb_objects = len(self.ycb_objects)
            
            # Create a column-oriented layout (spread out in y, not x)
            # We'll use a single column with multiple rows
            grid_cols = 1  # Single column
            grid_rows = num_ycb_objects  # Each object gets its own row
            
            # Calculate the height of each row
            row_height = (ycb_area_y_max - ycb_area_y_min) / grid_rows
            
            # Calculate the width of the column (we'll use the full width)
            col_width = ycb_area_x_max - ycb_area_x_min
            
            # Initialize YCB objects in their grid cells
            for obj_idx, obj in enumerate(self.ycb_objects):
                # Each object gets its own row
                row = obj_idx
                
                # Calculate cell boundaries - full width, limited height
                cell_x_min = ycb_area_x_min
                cell_x_max = ycb_area_x_max
                cell_y_min = ycb_area_y_min + row * row_height
                cell_y_max = cell_y_min + row_height
                
                # Add some margin within each cell to avoid objects being at the edges
                margin_x = 0.01  # Small margin for x
                margin_y = 0.03  # Larger margin for y to ensure separation
                cell_x_min += margin_x
                cell_x_max -= margin_x
                cell_y_min += margin_y
                cell_y_max -= margin_y
                
                # Generate random positions within the cell for all environments
                # Limited x randomization (more concentrated)
                x_range = (cell_x_max - cell_x_min) * 0.5  # Reduce x randomization to 50%
                x_center = (cell_x_min + cell_x_max) / 2
                x = torch.rand(b, device=self.device) * x_range + (x_center - x_range/2)
                
                # Full y randomization within the row
                y = torch.rand(b, device=self.device) * (cell_y_max - cell_y_min) + cell_y_min
                
                z = torch.full((b,), ycb_area_z, device=self.device)
                
                # Create position vectors
                positions = torch.stack([x, y, z], dim=1)
                
                # Randomize orientation around z-axis only (keep object upright)
                # Generate random rotation angles around z-axis
                theta = torch.rand(b, device=self.device) * 2 * np.pi
                
                # Convert to quaternions (rotation around z-axis only)
                qw = torch.cos(theta / 2)
                qx = torch.zeros_like(qw)
                qy = torch.zeros_like(qw)
                qz = torch.sin(theta / 2)
                
                # Stack quaternion components
                q = torch.stack([qw, qx, qy, qz], dim=1)
                
                # Set new poses
                obj.set_pose(Pose.create_from_pq(positions, q))

    def _get_obs_extra(self, info: Dict):
        return dict()
    
    def evaluate(self):
        return {}
