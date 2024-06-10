import numpy as np
from pettingzoo.mpe import simple_tag_v3
import pandas as pd
import random

def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

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

def update_positions(positions, velocities, headings, delta_t):
    new_positions = []
    for pos, vel, heading in zip(positions, velocities, headings):
        new_x = pos[0] + vel * np.cos(heading) * delta_t
        new_y = pos[1] + vel * np.sin(heading) * delta_t
        new_positions.append(np.array([new_x, new_y]))
    return new_positions

def initialize_agent_types(observations, num_active_pursuers):
    distances_to_evaders = []
    for agent in observations:
        if 'adversary' in agent:
            pursuer_position = observations[agent][2:4]
            for evader in [ev for ev in observations if 'agent' in ev]:
                evader_position = observations[evader][2:4]
                distance = np.linalg.norm(pursuer_position - evader_position)
                distances_to_evaders.append((agent, evader, distance))
    distances_to_evaders.sort(key=lambda x: x[2])
    assignments = {}
    for agent, evader, _ in distances_to_evaders:
        if agent not in assignments:
            assignments[agent] = evader
    return assignments

def update_agent_types(assignments, observations):
    distances_to_evaders = []
    for agent, evader in assignments.items():
        pursuer_position = observations[agent][2:4]
        evader_position = observations[evader][2:4]
        distance = np.linalg.norm(pursuer_position - evader_position)
        distances_to_evaders.append((agent, evader, distance))
    distances_to_evaders.sort(key=lambda x: x[2])
    new_assignments = {}
    for agent, evader, _ in distances_to_evaders:
        if agent not in new_assignments:
            new_assignments[agent] = evader
    return new_assignments

def extract_features(observation, agent_name, agent_type, num_pursuers, num_landmarks, num_evaders):
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
        'agent_type': agent_type,
        'agent_name': agent_name
    }
    return features

def cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, speed, speed_ratio=1.5):
    if agent_type == 'active':
        target_position = calculate_apollonius_circle(features['self_pos'], evader_position, speed_ratio)
        optimal_heading = calculate_angle(features['self_pos'], target_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    elif agent_type == 'non-active':
        strategic_position = calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active)
        optimal_heading = calculate_angle(features['self_pos'], strategic_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return action

def heading_to_continuous_action(heading, max_speed):
    speed = max_speed
    action_x = np.clip(speed * np.cos(heading), -1.0, 1.0)
    action_y = np.clip(speed * np.sin(heading), -1.0, 1.0)
    if action_x < 0 and action_y < 0:
        return np.array([0.0, abs(action_x), 0.0, abs(action_y), 0.0])
    elif action_x > 0 and action_y > 0:
        return np.array([0.0, 0.0, abs(action_x), 0.0, abs(action_y)])
    elif action_x < 0 and action_y > 0:
        return np.array([0.0, abs(action_x), 0.0, 0.0, abs(action_y)])
    elif action_x > 0 and action_y < 0:
        return np.array([0.0, 0.0, abs(action_x), abs(action_y), 0.0])
    elif action_x < 0:
        return np.array([0.0, abs(action_x), 0.0, 0.0, 0.0])
    elif action_x > 0:
        return np.array([0.0, 0.0, abs(action_x), 0.0, 0.0])
    elif action_y < 0:
        return np.array([0.0, 0.0, 0.0, abs(action_y), 0.0])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, abs(action_y)])

def calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active, radius=2.0):
    active_pursuer_positions = [f['self_pos'] for f in other_agent_features if f['agent_type'] == 'active']
    if len(active_pursuer_positions) < 1:
        angle_increment = 2 * np.pi / total_non_active
        angle_offset = agent_index * angle_increment
        x_offset = radius * np.cos(angle_offset)
        y_offset = radius * np.sin(angle_offset)
        strategic_position = evader_position + np.array([x_offset, y_offset])
    else:
        closest_active_pursuer = min(active_pursuer_positions, key=lambda pos: np.linalg.norm(features['self_pos'] - pos))
        angle_to_evader = calculate_angle(closest_active_pursuer, evader_position)
        support_angle = angle_to_evader + (agent_index * np.pi / total_non_active)
        x_offset = radius * np.cos(support_angle)
        y_offset = radius * np.sin(support_angle)
        strategic_position = closest_active_pursuer + np.array([x_offset, y_offset])
    return strategic_position

def calculate_overlapping_angle(pos_i, pos_j, evader_position, capture_radius, speed_ratio):
    di = np.linalg.norm(pos_i - evader_position)
    dj = np.linalg.norm(pos_j - evader_position)
    if di**2 > 1 and dj**2 > 1:
        phi_i = np.arccos((1 - di ** 2) / (2 * di))
        phi_j = np.arccos((1 - dj ** 2) / (2 * dj))
    else:
        phi_i = 0
        phi_j = 0
    lambda_i = calculate_angle(pos_i, evader_position)
    lambda_j = calculate_angle(pos_j, evader_position)
    theta_ij = phi_i + phi_j - (lambda_i - lambda_j)
    return theta_ij

def optimal_evader_heading(features_e, features_i, features_j, evader_speed=1.5):
    di = features_i['distances_to_agents'][0]
    dj = features_j['distances_to_agents'][0]
    lambda_i = calculate_angle(features_i['self_pos'], features_i['self_pos'] + np.array([di, 0]))
    lambda_j = calculate_angle(features_j['self_pos'], features_j['self_pos'] + np.array([dj, 0]))
    try:
        if di > 0 and dj > 0:
            Psi_Es = (1 / abs(dj) * (np.cos(lambda_j) - np.sin(lambda_j) / np.sqrt(max(dj ** 2 - 2, 1e-6)))) - \
                     (1 / abs(di) * (np.cos(lambda_i) + np.sin(lambda_i) / np.sqrt(max(di ** 2 - 2, 1e-6))))
            Psi_Ec = (1 / abs(di) * (np.sin(lambda_i) - np.cos(lambda_i) / np.sqrt(max(di ** 2 - 2, 1e-6)))) - \
                     (1 / abs(dj) * (np.sin(lambda_j) + np.cos(lambda_j) / np.sqrt(max(dj ** 2 - 2, 1e-6))))

            if Psi_Es == 0 and Psi_Ec == 0:
                sin_psi_E = 0
                cos_psi_E = 1  # Arbitrary direction since there's no meaningful heading
            else:
                sin_psi_E = Psi_Es / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)
                cos_psi_E = Psi_Ec / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)

            action_x = evader_speed * cos_psi_E
            action_y = evader_speed * sin_psi_E

            action = np.clip([0.0, action_x, action_y, 0.0, 0.0], -1.0, 1.0)

        else:
            action = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    except (ValueError, ZeroDivisionError):
        action = np.array([0.0, 0.5, 0.5, 0.0, 0.0])

    return action

def evader_strategy(features, pursuer_features, evader_speed=1.5):
    min_theta = float('inf')
    weakest_link = None
    for i, f_i in enumerate(pursuer_features):
        for j, f_j in enumerate(pursuer_features):
            if i != j:
                theta_ij = calculate_overlapping_angle(f_i['self_pos'], f_j['self_pos'], features['self_pos'], 2.0, 1.5)
                if theta_ij < min_theta or min_theta == float('inf'):
                    min_theta = theta_ij
                    weakest_link = (i, j)
    if weakest_link:
        i, j = weakest_link
        f_i, f_j = pursuer_features[i], pursuer_features[j]
        evader_action = optimal_evader_heading(features, f_i, f_j, evader_speed)
    else:
        evader_action = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    return evader_action

def main_loop():
    for i in range(1, 6):
        seed = random.randint(1, 10000)

        pursuer_speed = 1.5
        evader_speed = 2.5
        num_total_pursuers = 5
        num_evaders = 5 # Adjust based on your scenario
        num_active_pursuers = num_evaders if num_evaders < num_total_pursuers else num_total_pursuers
        num_non_active_pursuers = num_total_pursuers - num_active_pursuers
        total_data = []

        env = simple_tag_v3.parallel_env(num_good=num_evaders, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=50, continuous_actions=True, render_mode='human')
        observations, infos = env.reset(seed=seed)

        assignments = initialize_agent_types(observations, num_active_pursuers)
        round_counter = 0

        while env.agents:
            round_counter += 1
            actions = {}
            all_features = {agent: extract_features(observations[agent], agent, 'adversary' if 'adversary' in agent else 'evader', num_total_pursuers, 0, num_evaders) for agent in env.agents}
            assignments = update_agent_types(assignments, observations)

            for agent in env.agents:
                agent_index = int(agent.split('_')[1])
                agent_type = 'active' if agent in assignments else 'non-active'
                features = all_features[agent]
                other_agent_features = [all_features[other_agent] for other_agent in env.agents if other_agent != agent]

                if 'adversary' in agent:
                    evader_position = observations[assignments[agent]][2:4] if agent in assignments else np.array([0, 0])
                    total_non_active = list(assignments.values()).count('non-active')
                    action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, pursuer_speed)
                else:
                    action = evader_strategy(features, other_agent_features, evader_speed)

                actions[agent] = action

                total_data.append({
                    'game': i,
                    'round': round_counter,
                    'agent': agent,
                    'seed': seed,   
                    'agent_type': agent_type,
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
