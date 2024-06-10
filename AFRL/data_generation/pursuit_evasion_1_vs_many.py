import numpy as np


'''
Strategy for Evader:
- He aims to minimize the overlapping angle between the cartesian ovals of active pursers.
- By doing so, the evader can attempt to create a gap through which it can escape.
- The evader's best strategy is to identify and attack the weakest link in the pursuers' formation,
defined by the smallest overlapping angle between the Cartesian Ovals of the active pursuers.
- The evader's heading angle should be chosen to minimize the rate of change of the weakest overlapping angle, 
aiming to escape through the smallest gap.

Strategy for Pursuers:
- Pursuers must cooperate and coordinate their actions to form an encirclement around the evader. 
This cooperation is essential due to the evader's speed advantage.
- Unlike point capture, pursuers are endowed with a positive capture radius. 
This means that a pursuer does not need to touch the evader directly but only needs to come within a certain distance to capture it.
- The paper gives Cartesian Ovals to separate the reachable regions of the pursuers and the evader. 
These ovals consider both the speed ratio and the capture radius, providing a more accurate representation of reachable areas 
compared to Apollonius circles.
- Active Pursuers: A pair of pursuers identified as being closest to the evader's escape route become the active pursuers. 
Their primary role is to capture the evader by moving to the intersection points of their Cartesian Ovals.
- Non-active Pursuers: Other pursuers work to maintain and maximize the overlap of their Cartesian Ovals with their neighbors, 
effectively closing gaps and preventing the evader from escaping.
- Non-active pursuers should maximize the rate of change of the overlapping angle 
between their Cartesian Ovals to ensure no gaps are created for the evader to escape.
- Active pursuers should move to capture the evader at the intersection of their Cartesian Ovals, 
calculated dynamically based on the current positions and movement of the evader.

'''
#This calculates head angle for an agent 
def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle
# This calculates the cartesian oval for the pursuers
#THIS IS CO FROM THE PAPER
def calculate_cartesian_oval(position1, position2, capture_radius, speed_ratio):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    if distance <= capture_radius:
        return position1  # Pursuer can capture directly

    oval_x = (speed_ratio * delta_x + capture_radius) / (1 - speed_ratio**2)
    oval_y = (speed_ratio * delta_y + capture_radius) / (1 - speed_ratio**2)
    
    return np.array([position1[0] + oval_x, position1[1] + oval_y])

def update_positions(positions, velocities, headings, delta_t):
    new_positions = []
    for pos, vel, heading in zip(positions, velocities, headings):
        new_x = pos[0] + vel * np.cos(heading) * delta_t
        new_y = pos[1] + vel * np.sin(heading) * delta_t
        new_positions.append(np.array([new_x, new_y]))
    return new_positions

#creates agents and does the initial classification of agents
def initialize_agent_types(observations, num_active_pursuers):
    distances_to_evader = [] 
    #TODO: Make this plural to evaders
    evader_position = observations['agent_0'][2:4]  # get evader position

    for agent in observations:
        if 'adversary' in agent:
            adversary_position = observations[agent][2:4] 
            distance = np.linalg.norm(adversary_position - evader_position) 
            distances_to_evader.append((agent, distance))

    distances_to_evader.sort(key=lambda x: x[1]) #sort by distance to evader
    active_agents = [agent for agent, _ in distances_to_evader[:num_active_pursuers]] #cloest agents to evader are active
    non_active_agents = [agent for agent, _ in distances_to_evader[num_active_pursuers:]] #rest are non-active

    agent_types = {agent: 'active' for agent in active_agents}
    agent_types.update({agent: 'non-active' for agent in non_active_agents})

    return agent_types

#This updates the agent types based on the distance to the evader
#TODO: update for more evaders, say ensure the closest pursuer to each evader is active
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

import numpy as np


import numpy as np

def extract_features(observation, agent_name, agent_type, num_pursuers, num_landmarks, num_evaders):
    self_vel = observation[:2]
    self_pos = observation[2:4] # position of the agent
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

'''

#This runs the containment strategy for pursuers 
def cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, speed, speed_ratio=1.5):
    if agent_type == 'active':
        target_position = calculate_cartesian_oval(features['self_pos'], evader_position, capture_radius=2.0, speed_ratio=speed_ratio)
        optimal_heading = calculate_angle(features['self_pos'], target_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    elif agent_type == 'non-active':
        strategic_position = calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active)
        optimal_heading = calculate_angle(features['self_pos'], strategic_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return action
'''
'''
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
        strategic_position = calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active)
        optimal_heading = calculate_angle(features['self_pos'], strategic_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return action
'''
def cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, speed, speed_ratio=1.5):
    all_pursuer_positions = [f['self_pos'] for f in other_agent_features if 'adversary' in f['agent_name']]

    if agent_type == 'active':
        weakest_link, min_theta = calculate_weakest_link(all_pursuer_positions, evader_position, capture_radius=2.0, speed_ratio=speed_ratio)
        if weakest_link:
            target_position = calculate_cartesian_oval(features['self_pos'], evader_position, capture_radius=2.0, speed_ratio=speed_ratio)
            optimal_heading = calculate_angle(features['self_pos'], target_position)
        else:
            optimal_heading = calculate_angle(features['self_pos'], evader_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    elif agent_type == 'non-active':
        strategic_position = calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active)
        optimal_heading = calculate_angle(features['self_pos'], strategic_position)
        action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return action




#This calculates the continuous action from the heading angle, following the paper
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

#calculare weakest link between two pursuers
#key for the evader to escape and active pursuers to protect
'''
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
'''
def calculate_weakest_link(all_pursuers_positions, evader_position, capture_radius, speed_ratio):
    min_theta = float('inf')
    weakest_link = None

    for i, pos_i in enumerate(all_pursuers_positions):
        for j, pos_j in enumerate(all_pursuers_positions):
            if i != j:
                theta_ij = calculate_overlapping_angle(pos_i, pos_j, evader_position, capture_radius, speed_ratio)
                if theta_ij < min_theta or min_theta == float('inf'):
                    min_theta = theta_ij
                    weakest_link = (pos_i, pos_j)
    
    return weakest_link, min_theta


#This calculates the strategic position for non-active pursuers
'''
def calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active, radius=2.0):
    angle_increment = 2 * np.pi / total_non_active
    angle_offset = agent_index * angle_increment
    x_offset = radius * np.cos(angle_offset)
    y_offset = radius * np.sin(angle_offset)
    strategic_position = evader_position + np.array([x_offset, y_offset])
    print(f"Non-active pursuer {agent_index} moving to {strategic_position}")
    return strategic_position
'''
#calculate the strategic position for non-active pursuers
'''
def calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active, radius=2.0):
    active_pursuer_positions = [f['self_pos'] for f in other_agent_features if f['agent_type'] == 'active']

    if len(active_pursuer_positions) < 1: # no active pursuers
        angle_increment = 2 * np.pi / total_non_active
        angle_offset = agent_index * angle_increment
        x_offset = radius * np.cos(angle_offset)
        y_offset = radius * np.sin(angle_offset)
        strategic_position = evader_position + np.array([x_offset, y_offset])
    else:
        # Find the closest active pursuer
        closest_active_pursuer = min(active_pursuer_positions, key=lambda pos: np.linalg.norm(features['self_pos'] - pos))
        # Calculate position to support the closest active pursuer
        angle_to_evader = calculate_angle(closest_active_pursuer, evader_position)
        support_angle = angle_to_evader + (agent_index * np.pi / total_non_active)
        x_offset = radius * np.cos(support_angle)
        y_offset = radius * np.sin(support_angle)
        strategic_position = closest_active_pursuer + np.array([x_offset, y_offset])

    print(f"Non-active pursuer {agent_index} moving to {strategic_position}")
    return strategic_position
'''
def calculate_non_active_position(features, other_agent_features, evader_position, agent_index, total_non_active, radius=2.0):
    active_pursuer_positions = [f['self_pos'] for f in other_agent_features if f['agent_type'] == 'active']
    num_active_pursuers = len(active_pursuer_positions)

    if num_active_pursuers < 1:
        angle_increment = 2 * np.pi / total_non_active
        angle_offset = agent_index * angle_increment
        x_offset = radius * np.cos(angle_offset)
        y_offset = radius * np.sin(angle_offset)
        strategic_position = evader_position + np.array([x_offset, y_offset])
    else:
        # Calculate the mean angle to evader for all active pursuers
        mean_angle_to_evader = np.mean([calculate_angle(pos, evader_position) for pos in active_pursuer_positions])
        angle_increment = (2 * np.pi / total_non_active) / num_active_pursuers
        support_angle = mean_angle_to_evader + (agent_index * angle_increment)
        
        x_offset = radius * np.cos(support_angle)
        y_offset = radius * np.sin(support_angle)
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
                if theta_ij < min_theta or min_theta == float('inf'):
                    min_theta = theta_ij
                    weakest_link = (i, j)
                    print(f"New weakest link found between pursuer {i} and {j}, theta_ij: {theta_ij}")
    
    if weakest_link:
        i, j = weakest_link
        f_i, f_j = pursuer_features[i], pursuer_features[j]
        evader_action = optimal_evader_heading(features, f_i, f_j, evader_speed)
        print(f"Weakest link between pursuer {i} and {j}, evader action: {evader_action}") 
    else:
        evader_action = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # Default action in case error
        print("No weakest link found, default evader action: [0.5, 0.5, 0.0, 0.0, 0.0]")  
    
    return evader_action
'''
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
'''

def calculate_overlapping_angle_evader(features_i, features_j):
    di = features_i['distances_to_agents'][0]
    dj = features_j['distances_to_agents'][0]
    
    try:
        if di**2 > 1:
            phi_i = np.arccos((1 - di ** 2) / (2 * di))
        else:
            phi_i = 0
        if dj**2 > 1:
            phi_j = np.arccos((1 - dj ** 2) / (2 * dj))
        else:
            phi_j = 0
    except ValueError:
        phi_i = 0
        phi_j = 0

    lambda_i = calculate_angle(features_i['self_pos'], features_i['self_pos'] + np.array([di, 0]))
    lambda_j = calculate_angle(features_j['self_pos'], features_j['self_pos'] + np.array([dj, 0]))
    
    theta_ij = phi_i + phi_j - (lambda_i - lambda_j)
    
    print(f"di: {di}, dj: {dj}, phi_i: {phi_i}, phi_j: {phi_j}, lambda_i: {lambda_i}, lambda_j: {lambda_j}, theta_ij: {theta_ij}")
    
    return theta_ij

'''
def calculate_overlapping_angle_evader(pos_i, pos_j, evader_position):
    di = np.linalg.norm(evader_position - pos_i)
    dj = np.linalg.norm(evader_position - pos_j)
    if di**2 > 1 and dj**2 > 1:
        phi_i = np.arccos((1 - di ** 2) / (2 * di))
        phi_j = np.arccos((1 - dj ** 2) / (2 * dj))
    else:
        phi_i = 0
        phi_j = 0
    lambda_i = calculate_angle(evader_position, pos_i)
    lambda_j = calculate_angle(evader_position, pos_j)
    theta_ij = phi_i + phi_j - (lambda_i - lambda_j)
    return theta_ij
'''
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

'''
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

            action_x = np.clip(0.5 * (evader_speed * cos_psi_E + 1.0), 0.0, 1.0)
            action_y = np.clip(0.5 * (evader_speed * sin_psi_E + 1.0), 0.0, 1.0)
            print(f"Evader optimal heading calculated with action_x: {action_x}, action_y: {action_y}")
        else:
            action_x, action_y = 0.5, 0.5
            print("di or dj is zero or negative, defaulting action_x and action_y to 0.5")
    except (ValueError, ZeroDivisionError):
        action_x, action_y = 0.5, 0.5
        print("Exception encountered, defaulting action_x and action_y to 0.5")

    return np.array([0.0, action_x, action_y, 0.0, 0.0])
'''

def optimal_evader_heading(features_e, features_i, features_j, evader_speed=1.5):
    di = features_i['distances_to_agents'][0]
    dj = features_j['distances_to_agents'][0]
    lambda_i = calculate_angle(features_i['self_pos'], features_i['self_pos'] + np.array([di, 0]))
    lambda_j = calculate_angle(features_j['self_pos'], features_j['self_pos'] + np.array([dj, 0]))

    print(f"di: {di}, dj: {dj}")
    print(f"lambda_i: {lambda_i}, lambda_j: {lambda_j}")

    try:
        if di > 0 and dj > 0:
            Psi_Es = (1 / abs(dj) * (np.cos(lambda_j) - np.sin(lambda_j) / np.sqrt(max(dj ** 2 - 2, 1e-6)))) - \
                     (1 / abs(di) * (np.cos(lambda_i) + np.sin(lambda_i) / np.sqrt(max(di ** 2 - 2, 1e-6))))
            Psi_Ec = (1 / abs(di) * (np.sin(lambda_i) - np.cos(lambda_i) / np.sqrt(max(di ** 2 - 2, 1e-6)))) - \
                     (1 / abs(dj) * (np.sin(lambda_j) + np.cos(lambda_j) / np.sqrt(max(dj ** 2 - 2, 1e-6))))

            print(f"Psi_Es: {Psi_Es}, Psi_Ec: {Psi_Ec}")

            if Psi_Es == 0 and Psi_Ec == 0:
                sin_psi_E = 0
                cos_psi_E = 1  # Arbitrary direction since there's no meaningful heading
            else:
                sin_psi_E = Psi_Es / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)
                cos_psi_E = Psi_Ec / np.sqrt(Psi_Es ** 2 + Psi_Ec ** 2)

            print(f"sin_psi_E: {sin_psi_E}, cos_psi_E: {cos_psi_E}")

            action_x = evader_speed * cos_psi_E
            action_y = evader_speed * sin_psi_E
            print(f"action_x: {action_x}, action_y: {action_y}")

            if action_x < 0 and action_y == 0:
                print("Moving left")
            #randomly set the action_x and action_y to be between -1 and 1
            #action_x = np.random.uniform(-1, 1) 
            #action_y = np.random.uniform(-1, 1)
            '''
            if action_x < 0 and action_y < 0:
                print("Moving left-down")   
                return np.array([0.0, -action_x, 0.0, -action_y, 0.0])
            elif action_x > 0 and action_y > 0:
                print("Moving right-up")
                return np.array([0.0, 0.0, action_x, 0.0, action_y])
            elif action_x < 0 and action_y > 0:
                print("Moving left-up")
                return np.array([0.0, -action_x, 0.0, 0.0, action_y])
            elif action_x > 0 and action_y < 0:
                print("Moving right-down")
                return np.array([0.0, 0.0, action_x, -action_y, 0.0])
            elif action_x < 0:
                print("Moving left")
                return np.array([0.0, -action_x, 0.0, 0.0, 0.0])
            elif action_x > 0:
                print("Moving right")
                return np.array([0.0, 0.0, action_x, 0.0, 0.0])
            elif action_y < 0:
                print("Moving down")
                return np.array([0.0, 0.0, 0.0, -action_y, 0.0])
            else:
                print("Moving up")
                return np.array([0.0, 0.0, 0.0, 0.0, action_y])
            '''
            return np.array([0.0, action_x, action_y, 0.0, 0.0])

        else:
            action_x, action_y = 0.5, 0.5
            print("di or dj is zero or negative, defaulting action_x and action_y to 0.5")
    except (ValueError, ZeroDivisionError) as e:
        print(f"Exception encountered: {e}")
        action_x, action_y = 0.5, 0.5
        print("Defaulting action_x and action_y to 0.5")

    return np.array([0.0, action_x, action_y, 0.0, 0.0])




