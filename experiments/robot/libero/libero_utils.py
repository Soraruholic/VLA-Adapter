"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import Optional, Tuple

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


# ========== Environment Perturbation Settings ==========
# Global variables to store perturbation settings (used by monkey-patched methods)
# Agentview camera (third-person, fixed in world frame)
_AGENTVIEW_POS_OFFSET = (0.0, 0.0, 0.0)
_AGENTVIEW_RPY_OFFSET = (0.0, 0.0, 0.0)  # Roll, Pitch, Yaw in degrees
# Wrist camera (robot0_eye_in_hand, mounted on gripper)
_WRIST_CAM_POS_OFFSET = (0.0, 0.0, 0.0)
_WRIST_CAM_RPY_OFFSET = (0.0, 0.0, 0.0)  # Roll, Pitch, Yaw in degrees
# Table height
_TABLE_HEIGHT_OFFSET = 0.0

# Store original methods for restoration
_ORIGINAL_SETUP_CAMERA = None
_ORIGINAL_TABLE_ARENA_INIT = None
_ORIGINAL_ROBOT_INIT = None


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Roll-Pitch-Yaw angles (in degrees) to quaternion (w, x, y, z).
    
    Args:
        roll: Rotation around x-axis in degrees
        pitch: Rotation around y-axis in degrees
        yaw: Rotation around z-axis in degrees
    
    Returns:
        Quaternion as (w, x, y, z)
    """
    # Convert degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Compute quaternion components
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)


def quat_multiply(q1: Tuple[float, float, float, float], 
                  q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Multiply two quaternions (w, x, y, z format).
    Result = q1 * q2 (q2 rotation applied first, then q1)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return (w, x, y, z)


def set_env_perturbation(
    agentview_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    agentview_rpy_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    wrist_cam_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    wrist_cam_rpy_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    table_height_offset: float = 0.0,
) -> None:
    """
    Set environment perturbation parameters for robustness evaluation.
    
    Args:
        agentview_pos_offset: (x, y, z) offset for agentview camera position in meters
            Default agentview pos: (0.5886, 0.0, 1.4903)
        agentview_rpy_offset: (roll, pitch, yaw) offset in DEGREES
            roll:  rotation around x-axis (tilting left/right)
            pitch: rotation around y-axis (tilting up/down)  
            yaw:   rotation around z-axis (rotating left/right)
        wrist_cam_pos_offset: (x, y, z) offset for wrist camera position in meters
            Default wrist cam pos: (0.05, 0.0, 0.0)
        wrist_cam_rpy_offset: (roll, pitch, yaw) offset in DEGREES
        table_height_offset: Height offset for table in meters (default table height is 0.8m)
    """
    global _AGENTVIEW_POS_OFFSET, _AGENTVIEW_RPY_OFFSET
    global _WRIST_CAM_POS_OFFSET, _WRIST_CAM_RPY_OFFSET
    global _TABLE_HEIGHT_OFFSET
    
    _AGENTVIEW_POS_OFFSET = agentview_pos_offset
    _AGENTVIEW_RPY_OFFSET = agentview_rpy_offset
    _WRIST_CAM_POS_OFFSET = wrist_cam_pos_offset
    _WRIST_CAM_RPY_OFFSET = wrist_cam_rpy_offset
    _TABLE_HEIGHT_OFFSET = table_height_offset
    
    # Apply monkey patches (for compatibility, though direct modification is preferred)
    _apply_agentview_camera_patch()
    _apply_wrist_camera_patch()
    _apply_table_height_patch()
    
    # Log perturbation settings
    if any(v != 0.0 for v in agentview_pos_offset):
        print(f"[ENV PERTURBATION] Agentview position offset: {agentview_pos_offset} (meters)")
    if any(v != 0.0 for v in agentview_rpy_offset):
        print(f"[ENV PERTURBATION] Agentview RPY offset: {agentview_rpy_offset} (degrees)")
    if any(v != 0.0 for v in wrist_cam_pos_offset):
        print(f"[ENV PERTURBATION] Wrist camera position offset: {wrist_cam_pos_offset} (meters)")
    if any(v != 0.0 for v in wrist_cam_rpy_offset):
        print(f"[ENV PERTURBATION] Wrist camera RPY offset: {wrist_cam_rpy_offset} (degrees)")
    if table_height_offset != 0.0:
        print(f"[ENV PERTURBATION] Table height offset: {table_height_offset}m")


def _apply_agentview_camera_patch() -> None:
    """Apply monkey patch to modify agentview camera setup."""
    global _ORIGINAL_SETUP_CAMERA
    
    # Save original method if not already saved
    if _ORIGINAL_SETUP_CAMERA is None:
        _ORIGINAL_SETUP_CAMERA = BDDLBaseDomain._setup_camera
    
    def patched_setup_camera(self, mujoco_arena):
        """Modified camera setup with configurable offset."""
        print(f"[DEBUG PATCH] patched_setup_camera CALLED! Offset: {_AGENTVIEW_POS_OFFSET}")
        
        # Default agentview camera parameters (from original LIBERO)
        # Unit: meters
        default_pos = np.array([0.5886131746834771, 0.0, 1.4903500240372423])
        default_quat = np.array([
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ])
        
        # Apply position offset
        new_pos = default_pos + np.array(_AGENTVIEW_POS_OFFSET)
        print(f"[DEBUG PATCH] Setting agentview camera to pos: {new_pos}")
        
        # Apply RPY offset (convert to quaternion and multiply)
        if any(v != 0.0 for v in _AGENTVIEW_RPY_OFFSET):
            rpy_quat = rpy_to_quat(_AGENTVIEW_RPY_OFFSET[0], _AGENTVIEW_RPY_OFFSET[1], _AGENTVIEW_RPY_OFFSET[2])
            default_quat_tuple = tuple(default_quat)
            new_quat = np.array(quat_multiply(rpy_quat, default_quat_tuple))
            new_quat = new_quat / np.linalg.norm(new_quat)  # Normalize
        else:
            new_quat = default_quat
        
        # Set canonical_agentview camera
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771 + _AGENTVIEW_POS_OFFSET[0], 
                 _AGENTVIEW_POS_OFFSET[1], 
                 1.4903500240372423 + _AGENTVIEW_POS_OFFSET[2]],
            quat=new_quat.tolist(),
        )
        
        # Set agentview camera (main camera used for observations)
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=new_pos.tolist(),
            quat=new_quat.tolist(),
        )
    
    # Apply patch
    BDDLBaseDomain._setup_camera = patched_setup_camera


def _apply_wrist_camera_patch() -> None:
    """Apply monkey patch to modify wrist camera (robot0_eye_in_hand) setup."""
    global _ORIGINAL_ROBOT_INIT
    
    # Skip if no offset is applied
    if (all(v == 0.0 for v in _WRIST_CAM_POS_OFFSET) and 
        all(v == 0.0 for v in _WRIST_CAM_RPY_OFFSET)):
        return
    
    from robosuite.models.robots.manipulators.panda_robot import Panda
    
    # Save original method if not already saved
    if _ORIGINAL_ROBOT_INIT is None:
        _ORIGINAL_ROBOT_INIT = Panda.__init__
    
    original_init = _ORIGINAL_ROBOT_INIT
    
    def patched_panda_init(self, *args, **kwargs):
        """Modified Panda init that adjusts wrist camera after initialization."""
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Find and modify the eye_in_hand camera in the robot's XML
        # Default wrist camera: pos="0.05 0 0", quat="0 0.707108 0.707108 0"
        camera = self.robot_model.find(".//camera[@name='eye_in_hand']")
        if camera is not None:
            # Default values
            default_pos = np.array([0.05, 0.0, 0.0])
            default_quat = np.array([0.0, 0.707108, 0.707108, 0.0])
            
            # Apply position offsets
            new_pos = default_pos + np.array(_WRIST_CAM_POS_OFFSET)
            
            # Apply RPY offsets (convert to quaternion and multiply)
            if any(v != 0.0 for v in _WRIST_CAM_RPY_OFFSET):
                rpy_quat = rpy_to_quat(_WRIST_CAM_RPY_OFFSET[0], _WRIST_CAM_RPY_OFFSET[1], _WRIST_CAM_RPY_OFFSET[2])
                default_quat_tuple = tuple(default_quat)
                new_quat = quat_multiply(rpy_quat, default_quat_tuple)
                new_quat = np.array(new_quat)
                new_quat = new_quat / np.linalg.norm(new_quat)
            else:
                new_quat = default_quat
            
            # Update camera attributes
            from robosuite.utils.mjcf_utils import array_to_string
            camera.set("pos", array_to_string(new_pos))
            camera.set("quat", array_to_string(new_quat))
    
    # Apply patch
    Panda.__init__ = patched_panda_init


def _apply_table_height_patch() -> None:
    """Apply monkey patch to modify table height."""
    global _ORIGINAL_TABLE_ARENA_INIT
    
    if _TABLE_HEIGHT_OFFSET == 0.0:
        return  # No need to patch if offset is zero
    
    from libero.libero.envs.arenas.table_arena import TableArena
    
    # Save original __init__ if not already saved
    if _ORIGINAL_TABLE_ARENA_INIT is None:
        _ORIGINAL_TABLE_ARENA_INIT = TableArena.__init__
    
    def patched_table_arena_init(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/table_arena.xml",
        floor_style="light-gray",
        wall_style="light-gray-plaster",
    ):
        # Modify table_offset with height perturbation
        modified_offset = (
            table_offset[0],
            table_offset[1],
            table_offset[2] + _TABLE_HEIGHT_OFFSET,
        )
        
        # Call original __init__ with modified offset
        _ORIGINAL_TABLE_ARENA_INIT(
            self,
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=modified_offset,
            has_legs=has_legs,
            xml=xml,
            floor_style=floor_style,
            wall_style=wall_style,
        )
    
    # Apply patch
    TableArena.__init__ = patched_table_arena_init


def reset_env_perturbation() -> None:
    """Reset environment perturbation to default (restore original methods)."""
    global _ORIGINAL_SETUP_CAMERA, _ORIGINAL_TABLE_ARENA_INIT, _ORIGINAL_ROBOT_INIT
    global _AGENTVIEW_POS_OFFSET, _AGENTVIEW_RPY_OFFSET
    global _WRIST_CAM_POS_OFFSET, _WRIST_CAM_RPY_OFFSET
    global _TABLE_HEIGHT_OFFSET
    
    # Reset global variables
    _AGENTVIEW_POS_OFFSET = (0.0, 0.0, 0.0)
    _AGENTVIEW_RPY_OFFSET = (0.0, 0.0, 0.0)
    _WRIST_CAM_POS_OFFSET = (0.0, 0.0, 0.0)
    _WRIST_CAM_RPY_OFFSET = (0.0, 0.0, 0.0)
    _TABLE_HEIGHT_OFFSET = 0.0
    
    # Restore original methods
    if _ORIGINAL_SETUP_CAMERA is not None:
        BDDLBaseDomain._setup_camera = _ORIGINAL_SETUP_CAMERA
    
    if _ORIGINAL_TABLE_ARENA_INIT is not None:
        from libero.libero.envs.arenas.table_arena import TableArena
        TableArena.__init__ = _ORIGINAL_TABLE_ARENA_INIT
    
    if _ORIGINAL_ROBOT_INIT is not None:
        from robosuite.models.robots.manipulators.panda_robot import Panda
        Panda.__init__ = _ORIGINAL_ROBOT_INIT


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    
    # Apply camera perturbation directly to MuJoCo model
    _apply_camera_perturbation_to_env(env)
    
    return env, task_description


def _apply_camera_perturbation_to_env(env):
    """Apply camera position/orientation perturbation directly to the MuJoCo model."""
    if not hasattr(env, 'env') or not hasattr(env.env, 'sim'):
        print("[WARNING] Cannot access env.env.sim for camera perturbation")
        return
    
    sim = env.env.sim
    
    # Apply agentview camera position perturbation
    if any(v != 0.0 for v in _AGENTVIEW_POS_OFFSET):
        try:
            cam_id = sim.model.camera_name2id("agentview")
            original_pos = sim.model.cam_pos[cam_id].copy()
            sim.model.cam_pos[cam_id] = original_pos + np.array(_AGENTVIEW_POS_OFFSET)
            print(f"[CAMERA PERTURB] agentview pos: {original_pos} -> {sim.model.cam_pos[cam_id]}")
        except Exception as e:
            print(f"[WARNING] Failed to modify agentview camera pos: {e}")
    
    # Apply agentview camera RPY perturbation
    if any(v != 0.0 for v in _AGENTVIEW_RPY_OFFSET):
        try:
            cam_id = sim.model.camera_name2id("agentview")
            original_quat = sim.model.cam_quat[cam_id].copy()
            # Convert RPY offset (degrees) to quaternion
            rpy_quat = rpy_to_quat(_AGENTVIEW_RPY_OFFSET[0], _AGENTVIEW_RPY_OFFSET[1], _AGENTVIEW_RPY_OFFSET[2])
            # Multiply: new_quat = rpy_quat * original_quat (apply rotation offset)
            original_quat_tuple = tuple(original_quat)
            new_quat = quat_multiply(rpy_quat, original_quat_tuple)
            new_quat = np.array(new_quat)
            new_quat = new_quat / np.linalg.norm(new_quat)  # Normalize
            sim.model.cam_quat[cam_id] = new_quat
            print(f"[CAMERA PERTURB] agentview RPY {_AGENTVIEW_RPY_OFFSET}deg: quat {original_quat} -> {new_quat}")
        except Exception as e:
            print(f"[WARNING] Failed to modify agentview camera orientation: {e}")
    
    # Apply wrist camera position perturbation (robot0_eye_in_hand)
    if any(v != 0.0 for v in _WRIST_CAM_POS_OFFSET):
        try:
            cam_id = sim.model.camera_name2id("robot0_eye_in_hand")
            original_pos = sim.model.cam_pos[cam_id].copy()
            sim.model.cam_pos[cam_id] = original_pos + np.array(_WRIST_CAM_POS_OFFSET)
            print(f"[CAMERA PERTURB] wrist pos: {original_pos} -> {sim.model.cam_pos[cam_id]}")
        except Exception as e:
            print(f"[WARNING] Failed to modify wrist camera pos: {e}")
    
    # Apply wrist camera RPY perturbation
    if any(v != 0.0 for v in _WRIST_CAM_RPY_OFFSET):
        try:
            cam_id = sim.model.camera_name2id("robot0_eye_in_hand")
            original_quat = sim.model.cam_quat[cam_id].copy()
            # Convert RPY offset (degrees) to quaternion
            rpy_quat = rpy_to_quat(_WRIST_CAM_RPY_OFFSET[0], _WRIST_CAM_RPY_OFFSET[1], _WRIST_CAM_RPY_OFFSET[2])
            # Multiply: new_quat = rpy_quat * original_quat
            original_quat_tuple = tuple(original_quat)
            new_quat = quat_multiply(rpy_quat, original_quat_tuple)
            new_quat = np.array(new_quat)
            new_quat = new_quat / np.linalg.norm(new_quat)  # Normalize
            sim.model.cam_quat[cam_id] = new_quat
            print(f"[CAMERA PERTURB] wrist RPY {_WRIST_CAM_RPY_OFFSET}deg: quat {original_quat} -> {new_quat}")
        except Exception as e:
            print(f"[WARNING] Failed to modify wrist camera orientation: {e}")


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, save_version=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{save_version}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
