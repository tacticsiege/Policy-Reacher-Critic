import numpy as np
import time

from unityagents import UnityEnvironment
from agent_utils import load_params, load_weights

from ddpg_agent import DDPG_Agent

def demo_saved_agent_cont(env, agent_name, n_episodes=3, seed=0, train_mode=False):
    # gather scenario information
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment and grab infos
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    # load params for saved agent
    params = load_params(agent_name, verbose=True)
    actor_weights = load_weights(agent_name, 'actor')
    critic_weights = load_weights(agent_name, 'critic')

    # create agent and set weights
    agent = DDPG_Agent(state_size, action_size, brain_name, seed=seed, params=params)
    agent.actor_local.load_state_dict(actor_weights)
    agent.actor_target.load_state_dict(actor_weights)   # only used for training
    agent.critic_local.load_state_dict(critic_weights)  # only used for training
    agent.critic_target.load_state_dict(critic_weights) # only used for training
    print(agent.display_params())

    # run demo
    demo_agent_cont(env, agent, num_agents, 
        n_episodes=n_episodes, 
        seed=seed, train_mode=train_mode)


def demo_agent_cont(env, agent, num_agents, n_episodes=3, seed=0, train_mode=False):
    print(f'\r\nRunning demo of \'{agent.name}\'')
    scores = []
    for i in range(1, n_episodes+1):
        score = 0
        env_info = env.reset(train_mode=train_mode)[agent.brain_name]
        states = env_info.vector_observations
        ep_scores = np.zeros(num_agents)
        agent.reset()

        for t in range(1000):
            actions = agent.act(states)
            env_info = env.step(actions)[agent.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            ep_scores += env_info.rewards
            states = next_states

            if np.any(dones):
                break
        mean_score = np.mean(ep_scores)
        min_score = np.min(ep_scores)
        max_score = np.max(ep_scores)
        scores.append(mean_score)

        print('\rEpisode {}\tMean: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(
            i, mean_score, min_score, max_score
        ))
    
    print('\r\nDemo complete! Avg score: {:.3f}'.format(np.mean(scores)))
    return scores