# Main loop
from pursuit_evasion2 import extract_features, cooperative_strategy_continuous, evader_strategy, update_agent_types, initialize_agent_types
from pettingzoo.mpe import simple_tag_v3
import pandas as pd
import random
import numpy as np


def initialize_speeds():
    global pursuer_speed, evader_speed
    
    pursuer_speed = 1.5  # Constant speed for all pursuers
    evader_speed = np.random.uniform(2.2, 3.2)

pursuer_speed = 0
evader_speed = 0
game_counter = 0
round_counter = 0

speed_ratio = None

num_active_pursuers = 1
num_total_pursuers = 5
num_evaders = 1
num_non_active_pursuers = num_total_pursuers - num_active_pursuers
total_data = []

initialize_speeds()

for i in range(1, 3): 
    seed = random.randint(1, 10000)

    env = simple_tag_v3.parallel_env(num_good=num_evaders, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=30, continuous_actions=True, render_mode='human')
    observations, infos = env.reset(seed=seed)

    agent_types = initialize_agent_types(observations, num_active_pursuers)

    speed_ratio = pursuer_speed / evader_speed
    round_counter = 0

    while env.agents:
        round_counter += 1

        actions = {}
        all_features = {agent: extract_features(observations[agent], agent, agent_types.get(agent, 'evader'), num_total_pursuers, 0, num_evaders) for agent in env.agents}
        agent_types = update_agent_types(agent_types, list(all_features.values()), num_active_pursuers)

        non_active_counter = 0
        active_counter = 0

        for agent in env.agents:
            agent_index = int(agent.split('_')[1])
            agent_type = agent_types.get(agent, 'evader')
            features = all_features[agent]
            other_agent_features = [all_features[other_agent] for other_agent in env.agents if other_agent != agent]

            if 'adversary' in agent:
                for other_agent in env.agents:
                    if 'agent' in other_agent:
                        evader_observation = observations[other_agent]
                        evader_position = evader_observation[2:4] if evader_observation is not None else np.array([0, 0])
                        total_non_active = list(agent_types.values()).count('non-active')
                        if agent_type == 'active':
                            action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, pursuer_speed, speed_ratio)
                            active_counter += 1
                        else:
                            action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active, pursuer_speed, speed_ratio)
                            non_active_counter += 1
            else:
                action = evader_strategy(features, other_agent_features, evader_speed)

            actions[agent] = action

            total_data.append({
                'game': i,
                'round': round_counter,
                'seed': seed,
                'agent': agent,
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
df.to_csv('simulation_data_with_features.csv', index=False)

print("Data saved to simulation_data_with_features.csv")