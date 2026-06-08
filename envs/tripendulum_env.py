import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from envs.goals import GOAL_NAMES, get_goal, sample_goal
from utils.angle_utils import relative_to_absolute, angle_error
from utils.reward import compute_reward
from utils.logging_utils import setup_logger

logger = setup_logger("tripendulum_env")

class TriPendulumGoalEnv(gym.Env):
    """
    Goal-Conditioned Gym Environment for the Cart-Triple Pendulum (TriPendulum-8)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config_path=None, render_mode=None):
        super().__init__()
        
        # Load configuration
        if config_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_dir, "..", "configs", "default.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Extract environment parameters
        self.x_max = self.config["env"]["x_max"]
        self.f_max = self.config["env"]["f_max"]
        self.max_episode_steps = self.config["env"]["max_episode_steps"]
        self.frame_skip = self.config["env"]["frame_skip"]
        self.random_initial_state = self.config["env"]["random_initial_state"]
        self.omega_limit = self.config["env"].get("omega_limit", 40.0)
        
        # Extract success conditions
        self.pose_threshold = self.config["success"]["pose_threshold"]
        self.omega_threshold = self.config["success"]["omega_threshold"]
        self.stable_x_threshold = self.config["success"]["stable_x_threshold"]
        self.stable_steps_required = self.config["success"]["stable_steps_required"]
        
        # Curriculum state
        self.curriculum_stage = 1
        self.goal_switch_steps = 200 # Frequency of switching goals in Stage 6
        
        # Load MuJoCo Model
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(curr_dir, "mujoco_model.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML model not found at {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space: horizontal force on the cart [F_cart]
        self.action_space = spaces.Box(
            low=-self.f_max,
            high=self.f_max,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space shape (20,)
        # [x, x_dot, sin(q_i), cos(q_i), sin(theta_abs_i), cos(theta_abs_i), omega_i, goal_i]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.renderer = None
        
        # Internal episode variables
        self.goal = None
        self.last_action = None
        self.stable_steps = 0
        self.max_abs_x = 0.0
        self.num_steps = 0
        
    def _get_allowed_goals(self):
        """
        Returns the list of goals permitted in the current curriculum stage.
        """
        if self.curriculum_stage == 1:
            return ["DDD", "UUU"]
        elif self.curriculum_stage == 2:
            return ["DDU", "DUD", "UDD"]
        elif self.curriculum_stage == 3:
            return ["DUU", "UDU", "UUD"]
        else: # Stages 4, 5, 6
            return GOAL_NAMES

    def reset(self, goal=None, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Handle Goal Assignment
        if options is not None and "goal" in options:
            self.goal = get_goal(options["goal"])
        elif goal is not None:
            self.goal = get_goal(goal)
        else:
            allowed = self._get_allowed_goals()
            self.goal = sample_goal(allowed)
            
        # 2. Reset MuJoCo Data
        mujoco.mj_resetData(self.model, self.data)
        
        # 3. Initialize Positions and Velocities based on Stage
        if self.curriculum_stage <= 4:
            # Stage 1-4: Start near the downward hanging position (DDD) with small noise
            cart_pos = 0.0
            q = np.random.uniform(-0.05, 0.05, size=3)
            cart_vel = 0.0
            omega = np.random.uniform(-0.05, 0.05, size=3)
        else:
            # Stage 5-6: Fully random initial state within bounds
            # Keep cart within rails (e.g. -0.5 to 0.5)
            cart_pos = np.random.uniform(-0.5, 0.5)
            # Random angles for all three poles in [-pi, pi]
            q = np.random.uniform(-np.pi, np.pi, size=3)
            # Random velocities
            cart_vel = np.random.uniform(-0.5, 0.5)
            omega = np.random.uniform(-1.0, 1.0, size=3)
            
        # Write to MuJoCo qpos/qvel
        self.data.qpos[0] = cart_pos
        self.data.qpos[1:] = q
        self.data.qvel[0] = cart_vel
        self.data.qvel[1:] = omega
        
        mujoco.mj_forward(self.model, self.data)
        
        # 4. Reset tracker variables
        self.last_action = np.zeros(1, dtype=np.float32)
        self.stable_steps = 0
        self.max_abs_x = abs(cart_pos)
        self.num_steps = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info
        
    def step(self, action):
        # 1. Apply Curriculum Stage 6 Goal-Switching
        if self.curriculum_stage == 6 and self.num_steps > 0 and self.num_steps % self.goal_switch_steps == 0:
            allowed = self._get_allowed_goals()
            self.goal = sample_goal(allowed)
            self.stable_steps = 0 # Must restabilize for new goal
            
        self.num_steps += 1
        
        # 2. Clip and Apply action
        action = np.clip(action, -self.f_max, self.f_max)
        self.data.ctrl[0] = action[0]
        
        # 3. Simulate steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        # 4. Retrieve state information
        x = self.data.qpos[0]
        x_dot = self.data.qvel[0]
        q = self.data.qpos[1:]
        omega = self.data.qvel[1:]
        theta_abs = relative_to_absolute(q)
        
        self.max_abs_x = max(self.max_abs_x, abs(x))
        
        # 5. Check constraints
        track_collision = abs(x) > self.x_max
        overspin = np.any(np.abs(omega) > self.omega_limit)
        
        # 6. Evaluate success criteria
        pose_error_vec = angle_error(theta_abs, self.goal['abs_angles'])
        pose_error = np.max(np.abs(pose_error_vec))
        
        is_pose_ok = pose_error < self.pose_threshold
        is_vel_ok = np.all(np.abs(omega) < self.omega_threshold)
        is_cart_ok = abs(x) < self.stable_x_threshold
        
        if is_pose_ok and is_vel_ok and is_cart_ok:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
            
        success = self.stable_steps >= self.stable_steps_required
        
        # 7. Compute reward
        reward, r_components = compute_reward(
            theta_abs=theta_abs,
            theta_goal_abs=self.goal['abs_angles'],
            x=x,
            x_dot=x_dot,
            omega=omega,
            action=action,
            last_action=self.last_action,
            x_max=self.x_max,
            weights=self.config['reward'],
            success=success,
            collision=track_collision,
            overspin=overspin
        )
        
        # 8. Compute termination conditions
        terminated = False
        if track_collision or overspin:
            terminated = True
            
        truncated = self.num_steps >= self.max_episode_steps
        
        # Update last action
        self.last_action = action.copy()
        
        obs = self._get_obs()
        
        # Pack info dictionary
        info = self._get_info()
        # Merge reward component breakdown for training analytics
        info.update(r_components)
        info.update({
            "success": success,
            "stable_steps": self.stable_steps,
            "track_collision": track_collision,
            "overspin": overspin,
            "pose_error": float(pose_error),
        })
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x = self.data.qpos[0]
        x_dot = self.data.qvel[0]
        q = self.data.qpos[1:]
        omega = self.data.qvel[1:]
        theta_abs = relative_to_absolute(q)
        
        obs = np.array([
            x,
            x_dot,
            np.sin(q[0]), np.cos(q[0]),
            np.sin(q[1]), np.cos(q[1]),
            np.sin(q[2]), np.cos(q[2]),
            np.sin(theta_abs[0]), np.cos(theta_abs[0]),
            np.sin(theta_abs[1]), np.cos(theta_abs[1]),
            np.sin(theta_abs[2]), np.cos(theta_abs[2]),
            omega[0], omega[1], omega[2],
            self.goal['binary'][0],
            self.goal['binary'][1],
            self.goal['binary'][2]
        ], dtype=np.float32)
        
        return obs

    def _get_info(self):
        q = self.data.qpos[1:]
        theta_abs = relative_to_absolute(q)
        
        return {
            "goal_name": self.goal['name'],
            "q_relative": q.copy(),
            "theta_abs": theta_abs.copy(),
            "theta_goal_abs": self.goal['abs_angles'].copy(),
            "x": self.data.qpos[0],
            "max_abs_x": self.max_abs_x,
        }

    def set_curriculum_stage(self, stage):
        """
        Sets the environment's curriculum stage (1 to 6).
        """
        if stage < 1 or stage > 6:
            raise ValueError("Curriculum stage must be between 1 and 6")
        self.curriculum_stage = stage
        logger.info(f"Environment curriculum stage set to {stage}")

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        else:
            super().render()

    def close(self):
        if self.renderer is not None:
            self.renderer = None
