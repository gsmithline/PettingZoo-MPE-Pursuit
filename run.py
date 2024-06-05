from pettingzoo.mpe import simple_tag_v3
import numpy as np
from pursuit_evasion import extract_features, cooperative_strategy_continuous, evader_strategy, initialize_agent_types

num_active_pursuers = 2  # Example: Change this to specify the number of active pursuers
num_total_pursuers = 5 # Example: Total number of pursuers

env = simple_tag_v3.env(num_good=1, num_adversaries=num_total_pursuers, num_obstacles=0, max_cycles=100, continuous_actions=True, render_mode='human')
env.reset(seed=345)

agent_types = initialize_agent_types(num_active_pursuers, num_total_pursuers)

def check_type_pick_action(agent):
    agent_index = int(agent.split('_')[1])
    agent_type = agent_types[agent_index]
    observation = env.observe(agent)
    features = extract_features(observation)
    other_agent_features = [extract_features(env.observe(other_agent)) for other_agent in env.agents if other_agent != agent]

    if agent_type != 'non-active':
        evader_observation = env.observe('agent_0')
        if evader_observation is not None:
            evader_position = evader_observation[2:4]  # Assuming 'agent_0' is the evader
        else:
            evader_position = np.array([0, 0])  # Default value if observation is None
        total_non_active = agent_types.count('non-active')
        action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active)
    else:
        action = evader_strategy(features, other_agent_features)

    # Ensure the action is an array-like object of numbers
    if action is not None:
        action = np.array(action, dtype=float)

    return action

# Main loop
while env.agents:
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None

    else:
        actions = {agent: check_type_pick_action(agent) for agent in env.agents}

    if None not in actions:
        observations, rewards, terminations, truncations, infos = env.step(actions) # type: ignore

    env.render()

env.close()






'''
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        features = extract_features(observation)
        other_agent_features = [extract_features(env.observe(other_agent)) for other_agent in env.agents if other_agent != agent]

        if 'adversary' in agent:
            agent_index = int(agent.split('_')[1])
            agent_type = agent_types[agent_index]
            evader_observation = env.observe('agent_0')
            if evader_observation is not None:
                evader_position = evader_observation[2:4]  # Assuming 'agent_0' is the evader
            else:
                evader_position = np.array([0, 0])  # Default value if observation is None
            total_non_active = agent_types.count('non-active')
            action = cooperative_strategy_continuous(features, agent_type, other_agent_features, evader_position, agent_index, total_non_active)
        else:
            action = evader_strategy(features, other_agent_features)

    env.step(action)
    env.render()

# Close the environment
env.close()
'''

