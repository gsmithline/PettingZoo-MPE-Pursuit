import numpy as np

#key for paper, this calcualted head angle
def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

def extract_features(observation):
    self_vel = observation[:2]
    self_pos = observation[2:4]
    num_landmarks = (len(observation) - 4) // 4
    landmark_rel_positions = observation[4:4+2*num_landmarks]
    other_agent_rel_positions = observation[4+2*num_landmarks:4+4*num_landmarks]
    other_agent_velocities = observation[4+4*num_landmarks:]

    distances_to_agents = [np.linalg.norm(rel_pos) for rel_pos in np.split(other_agent_rel_positions, len(other_agent_rel_positions) // 2)]
    angles_to_agents = [calculate_angle(self_pos, self_pos + rel_pos) for rel_pos in np.split(other_agent_rel_positions, len(other_agent_rel_positions) // 2)]
    distances_to_landmarks = [np.linalg.norm(rel_pos) for rel_pos in np.split(landmark_rel_positions, len(landmark_rel_positions) // 2)]
    angles_to_landmarks = [calculate_angle(self_pos, self_pos + rel_pos) for rel_pos in np.split(landmark_rel_positions, len(landmark_rel_positions) // 2)]

    features = {
        'self_vel': self_vel,
        'self_pos': self_pos,
        'distances_to_agents': distances_to_agents,
        'angles_to_agents': angles_to_agents,
        'distances_to_landmarks': distances_to_landmarks,
        'angles_to_landmarks': angles_to_landmarks,
        'other_agent_velocities': other_agent_velocities,
    }

    return features

def cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active):
    if agent_type == 'active':
        optimal_heading = calculate_angle(features['self_pos'], evader_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=1.0)
    elif agent_type == 'non-active':
        strategic_position = calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active)
        optimal_heading = calculate_angle(features['self_pos'], strategic_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=0.5)
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return action

def heading_to_continuous_action(heading, max_speed):
    speed = max_speed
    action_x = np.clip(0.5 * (speed * np.cos(heading) + 1.0), 0.0, 1.0)
    action_y = np.clip(0.5 * (speed * np.sin(heading) + 1.0), 0.0, 1.0)
    return np.array([action_x, action_y, 0.0, 0.0, 0.0])

def calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active, radius=2.0):
    angle_increment = 2 * np.pi / total_non_active
    angle_offset = agent_index * angle_increment
    x_offset = radius * np.cos(angle_offset)
    y_offset = radius * np.sin(angle_offset)
    strategic_position = evader_position + np.array([x_offset, y_offset])
    print(f"Non-active pursuer {agent_index} moving to {strategic_position}")
    return strategic_position

def evader_strategy(features, pursuer_features, evader_speed=1.5):
    min_theta = float('inf')
    weakest_link = None
    for i, f_i in enumerate(pursuer_features):
        for j, f_j in enumerate(pursuer_features):
            if i != j:
                theta_ij = calculate_overlapping_angle(f_i, f_j) #we do this to find the weakest link
                if theta_ij < min_theta:
                    min_theta = theta_ij
                    weakest_link = (i, j)
    
    if weakest_link:
        i, j = weakest_link
        f_i, f_j = pursuer_features[i], pursuer_features[j]
        evader_action = optimal_evader_heading(features, f_i, f_j, evader_speed)
    else:
        evader_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    return evader_action

def calculate_overlapping_angle(features_i, features_j):
    di = features_i['distances_to_agents'][0]
    dj = features_j['distances_to_agents'][0]
    if di**2 > 1 and dj**2 > 1:
        phi_i = np.arccos((1 - di ** 2) / (2 * di))
        phi_j = np.arccos((1 - dj ** 2) / (2 * dj))
    else:
        phi_i = 0
        phi_j = 0
    lambda_i = calculate_angle(features_i['self_pos'], features_i['self_pos'] + np.array([di, 0]))
    lambda_j = calculate_angle(features_j['self_pos'], features_j['self_pos'] + np.array([dj, 0]))
    theta_ij = phi_i + phi_j - (lambda_i - lambda_j)
    return theta_ij

def optimal_evader_heading(features_e, features_i, features_j, evader_speed=1.5):
    di = features_i['distances_to_agents'][0]
    dj = features_j['distances_to_agents'][0]
    lambda_i = calculate_angle(features_i['self_pos'], features_i['self_pos'] + np.array([di, 0]))
    lambda_j = calculate_angle(features_j['self_pos'], features_j['self_pos'] + np.array([dj, 0]))

    try:
        if di**2 > 1 and dj**2 > 1:
            Psi_Es = (1 / dj * (np.cos(lambda_j) - np.sin(lambda_j) / np.sqrt(dj ** 2 - 1))) - \
                     (1 / di * (np.cos(lambda_i) + np.sin(lambda_i) / np.sqrt(di ** 2 - 1)))
            Psi_Ec = (1 / di * (np.sin(lambda_i) - np.cos(lambda_i) / np.sqrt(di ** 2 - 1))) - \
                     (1 / dj * (np.sin(lambda_j) + np.cos(lambda_j) / np.sqrt(dj ** 2 - 1)))
        
            sin_psi_E = Psi_Es / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)
            cos_psi_E = Psi_Ec / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)

            action_x = np.clip(0.5 * (evader_speed * cos_psi_E + 1.0), 0.0, 1.0)
            action_y = np.clip(0.5 * (evader_speed * sin_psi_E + 1.0), 0.0, 1.0)
        else:
            action_x, action_y = 0.5, 0.5
    except (ValueError, ZeroDivisionError):
        action_x, action_y = 0.5, 0.5 #incase computation error?

    return np.array([action_x, action_y, 0.0, 0.0, 0.0])


def initialize_agent_types(num_active_pursuers, num_total_pursuers):
    agent_types = ['active'] * num_active_pursuers + ['non-active'] * (num_total_pursuers - num_active_pursuers)
    np.random.shuffle(agent_types)
    return agent_types