import numpy as np
import time
import os
from collections import namedtuple, deque

import torch

from agent_utils import save_ddpg

def train_ddpg(env, agent, num_agents,
                n_episodes=2000, max_t=1000, 
                print_every=5, t_learn=5, num_learn=5, 
                goal_score=30, score_window_size=100,
                keep_training=False):
    is_solved = False
    total_scores = []
    total_scores_deque = deque(maxlen=score_window_size)

    # save parameters before starting training
    save_ddpg(agent.name, params=agent.params, verbose=True)
    
    print(f'\r\nTraining started for \'{agent.name}\'...')
    training_start_time = time.time()
    for i_episode in range(1, n_episodes+1):
        # Reset Env and Agent
        env_info = env.reset(train_mode=True)[agent.brain_name]       # reset the environment    
        states = env_info.vector_observations                   # get the current state (for each agent)
        scores = np.zeros(num_agents)                            # initialize the score (for each agent)
        agent.reset()
        
        start_time = time.time()
        
        for t in range(max_t):
            actions = agent.act(states)
            
            env_info = env.step(actions)[agent.brain_name]            # send all actions to the environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done) # send actions to the agent
            
            scores += env_info.rewards                           # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            
            if t % t_learn == 0:
                for _ in range(num_learn):
                    agent.start_learn()
            
            if np.any(dones):                                   # exit loop if episode finished
                break

        # track progress
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores_deque.append(mean_score)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores_deque)
        duration = time.time() - start_time
        
        print(
            '\rEpisode {}\tAvg. Ep. Score: {:.2f}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tTime: {:.2f}'
            .format(i_episode, total_average_score, mean_score, min_score, max_score, duration))

        if i_episode % print_every == 0:
            # todo: save checkpoint
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, i_checkpoint=i_episode)
            
            print('\rEpisode {}\tAverage Score over {} episodes: {:.2f}'.format(
                i_episode, score_window_size, total_average_score))  
            
        if total_average_score >= goal_score and i_episode >= score_window_size and not is_solved:
            is_solved = True
            print(
                '\r\nEnvironment solved in {} episodes! Total Average score: {:.2f}'.format(
                i_episode, total_average_score))
            print('\rTotal Duration: {:.2f}m\n'.format(
                (time.time() - training_start_time)/ 60.0))
            
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, verbose=True)
            
            if keep_training:
                print('\r\nContinuing training...')
            else:
                return total_scores
    
    # finished all episodes
    print('\r\nCompleted training on {} episodes.'.format(n_episodes))
    print('\rAverage Score for last {} episodes: {:.2f}\tGoal: {}'.format(
        score_window_size, np.mean(total_scores_deque), goal_score))
    print('\rTotal Duration: {:.2f}m\n'.format((time.time() - training_start_time)/ 60.0))

    save_ddpg(agent.name, 
        actor_weights=agent.actor_local.state_dict(), 
        critic_weights=agent.critic_local.state_dict(),
        scores=total_scores)

    return total_scores
