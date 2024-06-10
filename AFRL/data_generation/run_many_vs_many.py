import numpy as np
from pettingzoo.mpe import simple_tag_v3
import pandas as pd
import random

#Compute the angle between two points in radians
def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

# Calculate the Apollonius circle for the interception point
# Apple! 
def calculate_apollonius_circle(position1, position2, speed_ratio):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    value_inside_sqrt = 1 - speed_ratio**2
    if value_inside_sqrt <= 0:
        value_inside_sqrt = 1e-6  # Small positive value to prevent division by zero or negative values
    radius = (distance * speed_ratio) / np.sqrt(value_inside_sqrt)
    circle_position = position1 + (radius * np.array([delta_x, delta_y]) / distance)
    return circle_position

# Update positions based on velocity and heading

def update_positions(positions, velocities, headings, delta_t):
    new_positions = []
    for pos, vel, heading in zip(positions, velocities, headings):
        new_x = pos[0] + vel * np.cos(heading) * delta_t
        new_y = pos[1] + vel * np.sin(heading) * delta_t
        new_positions.append(np.array([new_x, new_y]))
    return new_positions

# Initialize agent types based on the distance to evaders
def initialize_agent_types(observations, num_pursuers, num_evaders):
    distances_to_evaders = []
    assignments = {}
    for pursuer in [agent for agent in observations if 'adversary' in agent]:
        pursuer_position = observations[pursuer][2:4]
        for evader in [ev for ev in observations if 'agent' in ev]:
            evader_position = observations[evader][2:4]
            distance = np.linalg.norm(pursuer_position - evader_position)
            distances_to_evaders.append((pursuer, evader, distance))
    distances_to_evaders.sort(key=lambda x: x[2])
    for pursuer, evader, _ in distances_to_evaders:
        if evader not in assignments.values():
            assignments[pursuer] = evader
    return assignments

# Extract features at each time step for each agent
def extract_features(observation, agent_name, num_pursuers, num_landmarks, num_evaders):
    self_vel = observation[:2]
    self_pos = observation[2:4]
    landmark_rel_positions = observation[4:4+2*num_landmarks]
    other_agent_rel_positions = observation[4+2*num_landmarks:4+2*num_landmarks+2*(num_pursuers+num_evaders-1)]
    other_agent_velocities = observation[4+2*num_landmarks+2*(num_pursuers+num_evaders-1):]
    num_agents = len(other_agent_rel_positions) // 2
    distances_to_agents = [np.linalg.norm(other_agent_rel_positions[2*i:2*i+2]) for i in range(num_agents)]
    angles_to_agents = [calculate_angle(self_pos, self_pos + other_agent_rel_positions[2*i:2*i+2]) for i in range(num_agents)]
    distances_to_landmarks = [np.linalg.norm(landmark_rel_positions[2*i:2*i+2]) for i in range(num_landmarks)]
    angles_to_landmarks = [calculate_angle(self_pos, self_pos + landmark_rel_positions[2*i:2*i+2]) for i in range(num_landmarks)]
    features = {
        'self_vel': self_vel,
        'self_pos': self_pos,
        'distances_to_agents': distances_to_agents,
        'angles_to_agents': angles_to_agents,
        'distances_to_landmarks': distances_to_landmarks,
        'angles_to_landmarks': angles_to_landmarks,
        'other_agent_velocities': other_agent_velocities,
        'agent_name': agent_name
    }
    return features

# Cooperative strategy for pursuers based on Apollonius circle
def cooperative_strategy_continuous(features, evader_position, speed, speed_ratio=1.5):
    target_position = calculate_apollonius_circle(features['self_pos'], evader_position, speed_ratio)
    optimal_heading = calculate_angle(features['self_pos'], target_position)
    action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    return action

'''
We need to convert the heading to a continuous action that can be used in the environment
'''
def heading_to_continuous_action(heading, max_speed):
    speed = max_speed
    action_x = np.clip(speed * np.cos(heading), -1.0, 1.0)
    action_y = np.clip(speed * np.sin(heading), -1.0, 1.0)
    action = np.array([0.0, action_x, action_y, 0.0, 0.0])
    return np.nan_to_num(action)  # Non NANS!

# Calculate optimal evader heading to escape from pursuers
def optimal_evader_heading(features_e, pursuer_positions, evader_speed=1.5):
    escape_angles = [calculate_angle(features_e['self_pos'], p_pos) for p_pos in pursuer_positions]
    optimal_heading = np.mean(escape_angles) + np.pi  # Escape in the opposite direction
    action_x = evader_speed * np.cos(optimal_heading)
    action_y = evader_speed * np.sin(optimal_heading)
    action = np.clip([0.0, action_x, action_y, 0.0, 0.0], -1.0, 1.0)
    return np.nan_to_num(action)  #NO NAN VALUES

def main_loop():
    for i in range(1, 6):
        seed = random.randint(1, 10000)

        pursuer_speed = 1.5
        evader_speed = 2.5
        num_total_pursuers = 5
        num_evaders = 5  # Adjust based on your scenario
        total_data = []

        env = simple_tag_v3.parallel_env(num_good=num_evaders, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=50, continuous_actions=True, render_mode='human')
        observations, infos = env.reset(seed=seed)

        assignments = initialize_agent_types(observations, num_total_pursuers, num_evaders)
        round_counter = 0

        while env.agents:
            round_counter += 1
            actions = {}
            all_features = {agent: extract_features(observations[agent], agent, num_total_pursuers, 0, num_evaders) for agent in env.agents}

            for agent in env.agents:
                features = all_features[agent]
                if 'adversary' in agent:
                    evader_position = observations[assignments[agent]][2:4]
                    action = cooperative_strategy_continuous(features, evader_position, pursuer_speed)
                else:
                    pursuer_positions = [observations[p][2:4] for p in assignments if assignments[p] == agent]
                    action = optimal_evader_heading(features, pursuer_positions, evader_speed)

                actions[agent] = action

                total_data.append({
                    'game': i,
                    'round': round_counter,
                    'agent': agent,
                    'seed': seed,
                    'observation': observations[agent].tolist(),
                    'action': action.tolist(),
                    'reward': None,
                    'termination': None,
                    'truncation': None,
                    'self_vel': features['self_vel'].tolist(),
                    'self_pos': features['self_pos'].tolist(),
                    'distances_to_agents': features['distances_to_agents'],
                    'angles_to_agents': features['angles_to_agents'],
                    'distances_to_landmarks': features['distances_to_landmarks'],
                    'angles_to_landmarks': features['angles_to_landmarks'],
                    'other_agent_velocities': features['other_agent_velocities'].tolist()
                })

            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
                total_data[-1]['reward'] = rewards[agent]
                total_data[-1]['termination'] = terminations[agent]
                total_data[-1]['truncation'] = truncations[agent]

            env.render()

        env.close()

        df = pd.DataFrame(total_data)
        df.to_csv('simulation_data_with_features_many_vs_many.csv', index=False)
        print("Data saved to simulation_data_with_features.csv")

main_loop()
