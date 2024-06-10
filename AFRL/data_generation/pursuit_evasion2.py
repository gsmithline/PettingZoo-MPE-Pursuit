import numpy as np

def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

def update_positions(positions, velocities, headings, delta_t):
    new_positions = []
    for pos, vel, heading in zip(positions, velocities, headings):
        new_x = pos[0] + vel * np.cos(heading) * delta_t
        new_y = pos[1] + vel * np.sin(heading) * delta_t
        new_positions.append(np.array([new_x, new_y]))
    return new_positions

def calculate_cartesian_oval(position1, position2, capture_radius, speed_ratio):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    if distance <= capture_radius:
        return position1  # Pursuer can capture directly

    oval_x = (speed_ratio * delta_x + capture_radius) / (1 - speed_ratio**2)
    oval_y = (speed_ratio * delta_y + capture_radius) / (1 - speed_ratio**2)
    
    return np.array([position1[0] + oval_x, position1[1] + oval_y])

def initialize_agent_types(observations, num_active_pursuers):
    distances_to_evader = []
    evader_position = observations['agent_0'][2:4]

    for agent in observations:
        if 'adversary' in agent:
            adversary_position = observations[agent][2:4]
            distance = np.linalg.norm(adversary_position - evader_position)
            distances_to_evader.append((agent, distance))

    distances_to_evader.sort(key=lambda x: x[1])
    active_agents = [agent for agent, _ in distances_to_evader[:num_active_pursuers]]
    non_active_agents = [agent for agent, _ in distances_to_evader[num_active_pursuers:]]

    agent_types = {agent: 'active' for agent in active_agents}
    agent_types.update({agent: 'non-active' for agent in non_active_agents})

    return agent_types

def update_agent_types(agent_types, all_features, num_active=2):
    distances_to_evader = []
    for features in all_features:
        if 'adversary' in features['agent_name']:
            distance = features['distances_to_agents'][0]  # Assuming evader is the first agent
            distances_to_evader.append((features['agent_name'], distance))
    
    distances_to_evader.sort(key=lambda x: x[1])
    closest_indices = [agent for agent, _ in distances_to_evader[:num_active]]
    
    new_agent_types = {agent: 'non-active' for agent in agent_types}
    for agent in closest_indices:
        new_agent_types[agent] = 'active'
    
    return new_agent_types

def extract_features(observation, agent_name, agent_type, num_pursuers, num_landmarks, num_evaders):
    self_vel = observation[:2]
    self_pos = observation[2:4]  # Assuming position is at indices 2 and 3
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
    active_pursuer_positions = [f['self_pos'] for f in other_agent_features if f['agent_type'] == 'active']
    
    if agent_type == 'active':
        weakest_link, min_theta = calculate_weakest_link(active_pursuer_positions, evader_position, capture_radius=2.0, speed_ratio=speed_ratio)
        if weakest_link:
            target_position = calculate_cartesian_oval(features['self_pos'], evader_position, capture_radius=2.0, speed_ratio=speed_ratio)
            optimal_heading = calculate_angle(features['self_pos'], target_position)
        else:
            optimal_heading = calculate_angle(features['self_pos'], evader_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    elif agent_type == 'non-active':
        strategic_position = calculate_non_active_position(agent_index, total_non_active, evader_position, radius=3.0)
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
        # Move left-down 
        return np.array([0.0, abs(action_x), 0.0, abs(action_y), 0.0])
    elif action_x > 0 and action_y > 0:
        # Move right-up
        return np.array([0.0, 0.0, abs(action_x), 0.0, abs(action_y)])
    elif action_x < 0 and action_y > 0:
        # Move left-up
        return np.array([0.0, abs(action_x), 0.0, 0.0, abs(action_y)])
    elif action_x > 0 and action_y < 0:
        # Move right-down
        return np.array([0.0, 0.0, abs(action_x), abs(action_y), 0.0])
    elif action_x < 0:
        # Move left
        return np.array([0.0, abs(action_x), 0.0, 0.0, 0.0])
    elif action_x > 0:
        # Move right
        return np.array([0.0, 0.0, abs(action_x), 0.0, 0.0])
    elif action_y < 0:
        # Move down
        return np.array([0.0, 0.0, 0.0, abs(action_y), 0.0])
    else:
        # Move up
        return np.array([0.0, 0.0, 0.0, 0.0, abs(action_y)])

def calculate_weakest_link(active_pursuers_positions, evader_position, capture_radius, speed_ratio):
    min_theta = float('inf')
    weakest_link = None
    
    for i, pos_i in enumerate(active_pursuers_positions):
        for j, pos_j in enumerate(active_pursuers_positions):
            if i != j:
                theta_ij = calculate_overlapping_angle(pos_i, pos_j, evader_position, capture_radius, speed_ratio)
                if theta_ij < min_theta:
                    min_theta = theta_ij
                    weakest_link = (pos_i, pos_j)
    
    return weakest_link, min_theta

def calculate_non_active_position(agent_index, total_non_active, evader_position, radius=2.0):
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
                theta_ij = calculate_overlapping_angle_evader(f_i, f_j)
                print(f"Calculated theta_ij between pursuer {i} and {j}: {theta_ij}")  
                if theta_ij < min_theta:
                    min_theta = theta_ij
                    weakest_link = (i, j)
    
    if weakest_link:
        i, j = weakest_link
        f_i, f_j = pursuer_features[i], pursuer_features[j]
        evader_action = optimal_evader_heading(features, f_i, f_j, evader_speed)
        print(f"Weakest link between pursuer {i} and {j}, evader action: {evader_action}") 
    else:
        evader_action = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # Default action in case error
        print("No weakest link found, default evader action: [0.5, 0.5, 0.0, 0.0, 0.0]")  
    
    return evader_action

def calculate_overlapping_angle_evader(features_i, features_j):
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
    print(f"di: {di}, dj: {dj}, phi_i: {phi_i}, phi_j: {phi_j}, lambda_i: {lambda_i}, lambda_j: {lambda_j}, theta_ij: {theta_ij}")
    return theta_ij

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
        
            sin_psi_E = Psi_Es / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)
            cos_psi_E = Psi_Ec / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)

            action_x = evader_speed * cos_psi_E
            action_y = evader_speed * sin_psi_E
            print(f"Evader optimal heading calculated with action_x: {action_x}, action_y: {action_y}")

            # Determine continuous action based on the heading direction
            if action_x < 0 and action_y < 0:
                return np.array([0.0, -action_x, 0.0, -action_y, 0.0])
            elif action_x > 0 and action_y > 0:
                return np.array([0.0, 0.0, action_x, 0.0, action_y])
            elif action_x < 0 and action_y > 0:
                return np.array([0.0, -action_x, 0.0, 0.0, action_y])
            elif action_x > 0 and action_y < 0:
                return np.array([0.0, 0.0, action_x, -action_y, 0.0])
            elif action_x < 0:
                return np.array([0.0, -action_x, 0.0, 0.0, 0.0])
            elif action_x > 0:
                return np.array([0.0, 0.0, action_x, 0.0, 0.0])
            elif action_y < 0:
                return np.array([0.0, 0.0, 0.0, -action_y, 0.0])
            else:
                return np.array([0.0, 0.0, 0.0, 0.0, action_y])

        else:
            action_x, action_y = 0.5, 0.5
            print("di or dj is zero or negative, defaulting action_x and action_y to 0.5")
    except (ValueError, ZeroDivisionError):
        action_x, action_y = 0.5, 0.5
        print("Exception encountered, defaulting action_x and action_y to 0.5")

    return np.array([0.0, action_x, action_y, 0.0, 0.0])