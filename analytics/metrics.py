import numpy as np

def aggregate_episode_metrics(episode_logs, dt=0.01):
    """
    Aggregates metrics across multiple evaluation episodes for a single goal.
    
    Args:
        episode_logs (list of dict): Each entry represents an episode and contains:
            - reward_sum (float)
            - length (int)
            - success (bool)
            - collision (bool)
            - overspin (bool)
            - pose_errors (list of float)
            - energies (list of float)
            - max_abs_x (float)
            - stable_steps (int)
            - final_pose_error (float)
        dt (float): Duration of one environment step (frame_skip * mujoco_timestep)
        
    Returns:
        dict: Aggregated metrics
    """
    if not episode_logs:
        return {}
        
    rewards = [log["reward_sum"] for log in episode_logs]
    lengths = [log["length"] for log in episode_logs]
    successes = [log["success"] for log in episode_logs]
    collisions = [log["collision"] for log in episode_logs]
    overspins = [log["overspin"] for log in episode_logs]
    max_x_vals = [log["max_abs_x"] for log in episode_logs]
    stable_steps_vals = [log["stable_steps"] for log in episode_logs]
    final_errors = [log["final_pose_error"] for log in episode_logs]
    
    # Flat list of all pose errors and energy across all steps of all episodes
    all_pose_errors = []
    all_energies = []
    for log in episode_logs:
        all_pose_errors.extend(log["pose_errors"])
        all_energies.extend(log["energies"])
        
    success_rate = float(np.mean(successes))
    avg_reward = float(np.mean(rewards))
    avg_pose_error = float(np.mean(all_pose_errors)) if all_pose_errors else 0.0
    avg_final_pose_error = float(np.mean(final_errors))
    avg_energy = float(np.mean(all_energies)) if all_energies else 0.0
    avg_max_cart_displacement = float(np.mean(max_x_vals))
    avg_episode_length = float(np.mean(lengths))
    track_collision_rate = float(np.mean(collisions))
    overspin_rate = float(np.mean(overspins))
    
    # Stability duration in seconds
    stability_time = float(np.mean(stable_steps_vals) * dt)
    
    return {
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "average_pose_error": avg_pose_error,
        "average_final_pose_error": avg_final_pose_error,
        "average_energy": avg_energy,
        "average_max_cart_displacement": avg_max_cart_displacement,
        "average_episode_length": avg_episode_length,
        "track_collision_rate": track_collision_rate,
        "overspin_rate": overspin_rate,
        "stability_time": stability_time
    }
