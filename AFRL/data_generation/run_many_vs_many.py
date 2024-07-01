import numpy as np
from pettingzoo.mpe import simple_tag_v3
import pandas as pd
import random
from scipy.optimize import minimize

#compute angle in radiands
def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

# Initialize the Value Function
def initialize_value_function(positions):
    value_function = {}
    for pos in positions:
        value_function[tuple(pos)] = 0 
    return value_function

#compute the hamiltonian
def compute_hamiltonian(pursuer_pos, evader_pos, pursuer_vel, evader_vel, co_states):
    H = pursuer_vel * (co_states[0] * np.cos(pursuer_pos[2]) + co_states[1] * np.sin(pursuer_pos[2])) + \
        evader_vel * (co_states[2] * np.cos(evader_pos[2]) + co_states[3] * np.sin(evader_pos[2]))
    return H

#update control laws numerically
def update_control_laws(pursuer_pos, evader_pos, co_states):
    def pursuer_control(heading):
        return -np.dot(co_states[:2], np.array([np.cos(heading), np.sin(heading)]))
    
    def evader_control(heading):
        return np.dot(co_states[2:], np.array([np.cos(heading), np.sin(heading)]))

    pursuer_heading = minimize(pursuer_control, 0).x[0]
    evader_heading = minimize(evader_control, 0).x[0]
    return pursuer_heading, evader_heading

# Update the Value Function
def update_value_function(value_function, positions, control_laws):
    for pos in positions:
        value_function[tuple(pos)] = max(control_laws) - min(control_laws) 
    return value_function

# Calculate the Apollonius circle for the interception point
def calculate_apollonius_circle(position1, position2, speed_ratio):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    value_inside_sqrt = 1 - speed_ratio**2
    if value_inside_sqrt <= 0:
        value_inside_sqrt = 1e-6  # Small positive value to make we dont divide by zero
    radius = (distance * speed_ratio) / np.sqrt(value_inside_sqrt)
    circle_position = position1 + (radius * np.array([delta_x, delta_y]) / distance)
    return circle_position, radius

# Update positions based on velocity and heading
def update_positions(positions, velocities, headings, delta_t):
    new_positions = []
    for pos, vel, heading in zip(positions, velocities, headings):
        new_x = pos[0] + vel * np.cos(heading) * delta_t
        new_y = pos[1] + vel * np.sin(heading) * delta_t
        new_positions.append(np.array([new_x, new_y, heading]))  # Include heading in the position array
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
    assigned_evaders = set()
    for pursuer, evader, _ in distances_to_evaders:
        if evader not in assigned_evaders:
            assignments[pursuer] = evader
            assigned_evaders.add(evader)
    # Ensure all pursuers have an assignment
    all_pursuers = [agent for agent in observations if 'adversary' in agent]
    unassigned_pursuers = [pursuer for pursuer in all_pursuers if pursuer not in assignments]
    for pursuer in unassigned_pursuers:
        assigned_evader = random.choice(list(assignments.values()))
        assignments[pursuer] = assigned_evader
    return assignments

# Extract features at each time step for each agent
def extract_features(observation, agent_name, all_observations, num_pursuers, num_landmarks, num_evaders):
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
    self_pos.tolist()
    self_vel.tolist()
    #capture evader
    distances_to_evaders = []
    distances_to_pursuers = []
    angles_to_evaders = []
    angles_to_pursuers = []
    evader_velocities = []
    pursuers_velocities = []
    for agent, obs in all_observations.items():
        if 'agent' in agent and 'adversary' in agent_name:
            distances_to_evaders.append(np.linalg.norm(self_pos - obs[2:4]))
            angles_to_evaders.append(calculate_angle(self_pos, obs[2:4]))
            evader_velocities.append(obs[:2])
        if 'adversary' in agent and 'agent' in agent_name or 'adversary' in agent and 'adversary' in agent_name: #if agent is pursuer or evader I want to know the distance to pursuers
            distances_to_pursuers.append(np.linalg.norm(self_pos - obs[2:4]))
            angles_to_pursuers.append(calculate_angle(self_pos, obs[2:4]))
            pursuers_velocities.append(obs[:2])
    captured = []
    intercept_point = None,
    radius = None
    if 'adversary' in agent_name:
        #get radius of capture
        distances_to_evaders = distances_to_evaders.copy() 
        #agent name indices
        #drop agent_ from agent_name
        for agent, obs in all_observations.items():
            if 'agent' in agent:
                evader_position = obs[2:4]
                agent_name_index = int(agent.split('_')[1])
                intercept_point, radius = calculate_apollonius_circle(self_pos, evader_position, speed_ratio= 1.5 / 2.5)
                captured.append(capture_evader(distances_to_evaders[agent_name_index], radius))
    
    else:
        is_evader_captured = None
    

    features = {
        'self_vel': self_vel, #array of velocities ex. [0.1, 0.2]
        'self_pos': self_pos, #array of positions ex. [0.1, 0.2] in x, y coordinates
        'angles_to_pursuers': angles_to_pursuers, #array of angles to pursuers ex. [adversary_0, adversary_1, ...] -> [0.1, 0.2, ...]
        'angles_to_evaders': angles_to_evaders, #array of angles to evaders ex. [agent_0, agent_1, ...] -> [0.1, 0.2, ...],
        'velocities_of_evaders': evader_velocities, #array of velocities of evaders ex. [agent_0, agent_1, ...] -> [[0.1, 0.2], [0.3, 0.4], ...]
        'velocities_of_pursuers': pursuers_velocities, #array of velocities of pursuers ex. [agent_0, agent_1, ...] -> [[0.1, 0.2], [0.3, 0.4], ...]
        'distances_to_landmarks': distances_to_landmarks,
        'angles_to_landmarks': angles_to_landmarks,
        'other_agent_velocities': other_agent_velocities, #array of velocities of other agents ex. [agent_0, agent_1, ...] -> [[0.1, 0.2], [0.3, 0.4], ...]
        'agent_name': agent_name, #string name of agent
        'distances_to_evaders': distances_to_evaders, #array of distances to evaders ex. [adversary_0, adversary_1, ...] -> [0.1, 0.2, ...], NOTE: self is left out if they are a pursuer
        'distances_to_pursuers': distances_to_pursuers, #array of distances to pursuers ex. [agent_0, agent_1, ...] -> [0.1, 0.2, ...]
        'is_evader_captured': captured, #array of booleans in order of evaders ex. [agent_0, agent_1, ...] -> [True, False, ...] indicating capture
        'radius': radius #radius of capture, NONE IF EVADER 
    }
    return features

# Cooperative strategy for pursuers based on Apollonius circle
def cooperative_strategy_continuous(features, evader_position, speed, speed_ratio=1.5):
    target_position, radius = calculate_apollonius_circle(features['self_pos'], evader_position, speed_ratio)
    optimal_heading = calculate_angle(features['self_pos'], target_position)
    action = heading_to_continuous_action(optimal_heading, max_speed=speed)
    return action

def capture_evader(evader_position, capture_radius):
    if evader_position <= capture_radius:
        return True
    else:
        return False

def heading_to_continuous_action(heading, max_speed):
    speed = max_speed
    action_x = np.clip(speed * np.cos(heading), -1.0, 1.0)
    action_y = np.clip(speed * np.sin(heading), -1.0, 1.0)
    action = np.array([0.0, action_x, action_y, 0.0, 0.0])
    return np.nan_to_num(action)

# Calculate optimal evader heading to escape from pursuers
def optimal_evader_heading(features_e, pursuer_positions, evader_speed=1.5):
    if len(pursuer_positions) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    escape_angles = [calculate_angle(features_e['self_pos'], p_pos) for p_pos in pursuer_positions]
    optimal_heading = np.mean(escape_angles) + np.pi  # Escape in the opposite direction
    action_x = evader_speed * np.cos(optimal_heading)
    action_y = evader_speed * np.sin(optimal_heading)
    action = np.clip([0.0, action_x, action_y, 0.0, 0.0], -1.0, 1.0)
    return np.nan_to_num(action)

# Compute co-state variables based on the gradient of the value function
def compute_co_states(value_function, position):
    epsilon = 1e-5  
    co_states = np.zeros(4) #initial costate variables
    pos_array = np.array(position)

    for i in range(2):
        pos_array[i] += epsilon
        value_plus = value_function.get(tuple(pos_array), 0)
        pos_array[i] -= 2 * epsilon
        value_minus = value_function.get(tuple(pos_array), 0)
        pos_array[i] += epsilon
        co_states[i] = (value_plus - value_minus) / (2 * epsilon)

    return co_states
def run_many_vs_many(HumanRender=False):
    pursuer_speed = 1.5
    evader_speed = 2.5
    game_counter = 0
    pursuers_array = [2, 3, 4, 5]
    evaders_array = [2, 3, 4, 5]
    main_loop(pursuer_speed=pursuer_speed, evader_speed=evader_speed, evaders_array=evaders_array, pursuers_array=pursuers_array, game_counter=game_counter, HumanRender=HumanRender)

def main_loop(pursuer_speed=1.5, evader_speed=2.5, evaders_array=[], pursuers_array=[], game_counter=0, HumanRender=False):
    total_data = []
    for evaders in evaders_array:
        num_evaders = evaders
        for pursuer in pursuers_array:
            num_total_pursuers = pursuer
            game_counter += 1
            for i in range(1, 11):
                seed = random.randint(1, 10000)

                if HumanRender:
                    env = simple_tag_v3.parallel_env(num_good=num_evaders, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=15, continuous_actions=True, render_mode='human')
                else:
                    env = simple_tag_v3.parallel_env(num_good=num_evaders, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=15, continuous_actions=True)
                observations, infos = env.reset(seed=seed)

                assignments = initialize_agent_types(observations, num_total_pursuers, num_evaders)
                
                # Initialize Value Function
                initial_positions = [np.concatenate((observations[agent][2:4], [0])) for agent in env.agents]  # Include heading (0) in the position
                value_function = initialize_value_function(initial_positions)
                
                # Debugging: print assignments
                print("Assignments:", assignments)
                
                round_counter = 0

                while env.agents:
                    round_counter += 1
                    actions = {}
                    all_observations = env.observation_spaces
                    all_features = {agent: extract_features(observations[agent], agent, observations, 0, num_total_pursuers, num_evaders) for agent in env.agents}

                    for agent in env.agents:
                        features = all_features[agent]
                        if 'adversary' in agent: #adversaries are pursuers
                            if agent in assignments:
                                evader_position = np.concatenate((observations[assignments[agent]][2:4], [0]))  # Include heading (0) in the position
                                action = cooperative_strategy_continuous(features, evader_position, pursuer_speed)
                            else:
                                print(f"Error: {agent} has no assigned evader.")
                                continue
                        else:
                            pursuer_positions = [np.concatenate((observations[p][2:4], [0])) for p in assignments if assignments[p] == agent]  # Include heading (0) in the position
                            action = optimal_evader_heading(features, pursuer_positions, evader_speed)

                        actions[agent] = action

                        total_data.append({
                            'game': game_counter,
                            'number_of_game_type': i,  
                            'game_type': 'Pursuers: {} vs Evaders: {}'.format(num_total_pursuers, num_evaders),  # '1 vs many' or 'many vs many
                            'round': round_counter,
                            'agent': agent,
                            'seed': seed,
                            'observation': observations[agent],
                            'action': action.tolist(),
                            'reward': None,
                            'termination': None,
                            'truncation': None,
                            'self_vel': features['self_vel'],
                            'self_pos': features['self_pos'],
                            'angles_to_pursuers': features['angles_to_pursuers'],
                            'angles_to_evaders': features['angles_to_evaders'],
                            'velocities_of_evaders': features['velocities_of_evaders'],
                            'velocities_of_pursuers': features['velocities_of_pursuers'],
                            'distances_to_landmarks': features['distances_to_landmarks'],
                            'angles_to_landmarks': features['angles_to_landmarks'],
                            'other_agent_velocities': features['other_agent_velocities'],
                            'distances_to_evaders': features['distances_to_evaders'],
                            'distances_to_pursuers': features['distances_to_pursuers'],
                            'value_function': value_function,
                            'interception_point': None,
                            'distances_to_evaders': features['distances_to_evaders'],
                            'distances_to_pursuers': features['distances_to_pursuers'],
                            'is_evader_capture': features['is_evader_captured'],
                            'radius': features['radius']
                        })


                    for pos in initial_positions:
                        co_states = compute_co_states(value_function, pos)  
                        H = compute_hamiltonian(pos, evader_position, pursuer_speed, evader_speed, co_states)
                        
                        pursuer_heading, evader_heading = update_control_laws(pos, evader_position, co_states)
                        
                        control_laws = [pursuer_heading, evader_heading]  
                        value_function = update_value_function(value_function, initial_positions, control_laws)
                        

                        interception_point, radius = calculate_apollonius_circle(pos[:2], evader_position[:2], pursuer_speed / evader_speed)
                        
                    observations, rewards, terminations, truncations, infos = env.step(actions)

                    for agent in env.agents:
                        total_data[-1]['reward'] = rewards[agent]
                        total_data[-1]['termination'] = terminations[agent]
                        total_data[-1]['truncation'] = truncations[agent]
                        total_data[-1]['interception_point'] = interception_point
                        total_data[-1]['value_function'] = value_function

                    env.render()

                env.close()

            # Save data to dataframe 
            df = pd.DataFrame(total_data)
            df.to_csv(f'simulation_data_with_feature_many_{num_evaders}_vs_{num_total_pursuers}.csv', index=False)
            print("Data saved to simulation_data_with_features_many_vs_many.csv")

  