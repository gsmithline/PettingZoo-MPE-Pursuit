from pettingzoo.mpe import simple_tag_v3
import numpy as np
import pandas as pd
from pursuit_evasion import extract_features, cooperative_strategy_continuous, evader_strategy, initialize_agent_types

num_active_pursuers = 2
num_total_pursuers = 5
seed = 42

env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=25, continuous_actions=True, render_mode='human')
env.reset(seed=seed)

agent_types = initialize_agent_types(num_active_pursuers, num_total_pursuers)

observations, infos = env.reset(seed=seed)

data = []

while env.agents:
    actions = {}
    step_data = {}
    for agent in env.agents:
        observation = observations[agent]
        features = extract_features(observation)
        other_agent_features = [extract_features(observations[other_agent]) for other_agent in env.agents if other_agent != agent]

        if 'adversary' in agent:
            agent_index = int(agent.split('_')[1])
            agent_type = agent_types[agent_index]
            evader_observation = observations['agent_0']
            if evader_observation is not None:
                evader_position = evader_observation[2:4]  # Assuming 'agent_0' is the evader
            else:
                evader_position = np.array([0, 0])  # Default value if observation is none
            total_non_active = agent_types.count('non-active')
            action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active)
        else:
            action = evader_strategy(features, other_agent_features, evader_speed=1.5)

        actions[agent] = action
        #collected data
        if agent not in step_data:
            step_data[agent] = None
        step_data[agent] = {
            'observation': observation,
            'features': features,
            'action': action,
            'reward': None, 
            'termination': None, 
            'truncation': None
        }

        

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent in env.agents:
        step_data[agent]['reward'] = rewards[agent]
        step_data[agent]['termination'] = terminations[agent]
        step_data[agent]['truncations'] = truncations[agent]
    
    data.append(step_data)
    env.render()

# Close the environment
env.close()
flattened_data = []

for step, step_data in enumerate(data):
    for agent, agent_data in step_data.items():
        flat_features = {f'{k}_{i}': v for k, values in agent_data['features'].items() for i, v in enumerate(values)}
        flattened_data.append({
            'step': step,
            'agent': agent,
            'observation': agent_data['observation'],
            'action': agent_data['action'],
            'reward': agent_data['reward'],
            'termination': agent_data['termination'],
            'truncation': agent_data['truncation'],
            **flat_features
        })

df = pd.DataFrame(flattened_data)

# Save DataFrame to a CSV file for analysis
df.to_csv('simulation_data_with_features.csv', index=False)

print("Data saved to simulation_data_with_features.csv")

