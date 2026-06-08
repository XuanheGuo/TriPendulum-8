import numpy as np

def compute_reward(
    theta_abs,
    theta_goal_abs,
    x,
    x_dot,
    omega,
    action,
    last_action,
    x_max,
    weights,
    success,
    collision,
    overspin
):
    """
    Computes the reward and its constituent parts for debugging and logging.
    
    Args:
        theta_abs (np.ndarray): Absolute angles of the three links, shape (3,)
        theta_goal_abs (np.ndarray): Target absolute angles, shape (3,)
        x (float): Cart position
        x_dot (float): Cart velocity
        omega (np.ndarray): Pendulum joint angular velocities, shape (3,)
        action (np.ndarray): Current action applied to cart, shape (1,)
        last_action (np.ndarray or None): Previous action applied to cart, shape (1,)
        x_max (float): Track limit
        weights (dict): Weight parameters dictionary containing w_pose, w_vel, w_act, etc.
        success (bool): Whether the success stabilization criteria are met
        collision (bool): Whether the cart hit the boundary
        overspin (bool): Whether joint velocity limits were exceeded
        
    Returns:
        reward (float): Combined scalar reward
        components (dict): Dict of float values for each component reward/penalty term
    """
    # 1. Pose error based on cosine distance of absolute angles (range: 0 to 2 per link, total 0 to 6)
    r_pose = np.sum(1.0 - np.cos(theta_abs - theta_goal_abs))
    
    # 2. Cart velocity damping
    r_vel = x_dot ** 2
    
    # 3. Action magnitude penalty (energy consumption)
    act_val = float(action[0])
    r_act = act_val ** 2
    
    # 4. Continuous soft track limit penalty
    r_track = (abs(x) / x_max) ** 2
    
    # 5. Joint spin damping (stabilizes wild rotations)
    r_spin = np.sum(omega ** 2)
    
    # 6. Smoothness penalty (action rate of change)
    if last_action is not None:
        last_act_val = float(last_action[0])
        r_delta_a = (act_val - last_act_val) ** 2
    else:
        r_delta_a = 0.0
        
    # Scale components by their weights
    term_pose = weights.get('w_pose', 1.0) * r_pose
    term_vel = weights.get('w_vel', 0.05) * r_vel
    term_act = weights.get('w_act', 0.001) * r_act
    term_track = weights.get('w_track', 0.5) * r_track
    term_spin = weights.get('w_spin', 0.02) * r_spin
    term_delta_a = weights.get('w_delta_a', 0.01) * r_delta_a
    
    # Discontinuous rewards / penalties
    bonus_success = weights.get('success_bonus', 50.0) if success else 0.0
    penalty_collision = weights.get('collision_penalty', 200.0) if collision else 0.0
    penalty_overspin = weights.get('overspin_penalty', 100.0) if overspin else 0.0
    
    # Combined reward
    total_reward = -(term_pose + term_vel + term_act + term_track + term_spin + term_delta_a)
    total_reward += bonus_success - penalty_collision - penalty_overspin
    
    components = {
        "r_pose": float(r_pose),
        "r_vel": float(r_vel),
        "r_act": float(r_act),
        "r_track": float(r_track),
        "r_spin": float(r_spin),
        "r_delta_a": float(r_delta_a),
        "term_pose": float(term_pose),
        "term_vel": float(term_vel),
        "term_act": float(term_act),
        "term_track": float(term_track),
        "term_spin": float(term_spin),
        "term_delta_a": float(term_delta_a),
        "bonus_success": float(bonus_success),
        "penalty_collision": float(penalty_collision),
        "penalty_overspin": float(penalty_overspin)
    }
    
    return float(total_reward), components
