import numpy as np

# Calculate head angle for an agent
def calculate_angle(position1, position2):
    delta_x = position2[0] - position1[0]
    delta_y = position2[1] - position1[1]
    return np.arctan2(delta_y, delta_x)

# Calculate the interception point using Apollonius circle
def calculate_interception_point(pursuer_pos, evader_pos, pursuer_speed, evader_speed):
    delta_pos = evader_pos - pursuer_pos
    distance = np.linalg.norm(delta_pos)
    ratio = pursuer_speed / evader_speed
    interception_point = pursuer_pos + (delta_pos * ratio / (1 - ratio ** 2))
    return interception_point

# Assign pursuers to evaders based on their distances
def assign_pursuers_to_evaders(pursuers, evaders):
    assignments = []
    for evader in evaders:
        distances = [np.linalg.norm(evader['position'] - pursuer['position']) for pursuer in pursuers]
        closest_pursuer = np.argmin(distances)
        assignments.append((pursuers[closest_pursuer], evader))
    return assignments

# Implement the saddle-point strategy for the pursuer
def pursuer_strategy(pursuer, evader):
    interception_point = calculate_interception_point(pursuer['position'], evader['position'], pursuer['speed'], evader['speed'])
    heading = calculate_angle(pursuer['position'], interception_point)
    return heading

# Implement the optimal evader heading strategy
def evader_strategy(evader, pursuers):
    optimal_pursuer = min(pursuers, key=lambda p: np.linalg.norm(p['position'] - evader['position']))
    interception_point = calculate_interception_point(optimal_pursuer['position'], evader['position'], optimal_pursuer['speed'], evader['speed'])
    heading = calculate_angle(evader['position'], interception_point)
    return heading

# Main loop for the simulation
def simulation_loop(env, num_steps=100):
    for step in range(num_steps):
        actions = {}
        for agent in env.agents:
            if 'adversary' in agent:  # Pursuer
                pursuer_index = int(agent.split('_')[1])
                pursuer = {'position': env.positions[pursuer_index], 'speed': env.speeds[pursuer_index]}
                evader = {'position': env.evader_positions[0], 'speed': env.evader_speeds[0]}
                heading = pursuer_strategy(pursuer, evader)
                actions[agent] = heading
            elif 'agent' in agent:  # Evader
                evader_index = int(agent.split('_')[1])
                evader = {'position': env.evader_positions[evader_index], 'speed': env.evader_speeds[evader_index]}
                pursuers = [{'position': env.positions[p], 'speed': env.speeds[p]} for p in range(len(env.positions))]
                heading = evader_strategy(evader, pursuers)
                actions[agent] = heading
        env.step(actions)
        if env.done:
            break
    return env.get_results()
