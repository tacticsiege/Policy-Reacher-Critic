import numpy as np
import os
import json

import torch

import matplotlib.pyplot as plt

# env helpers
def env_initialize(env, train_mode=True, brain_idx=0, idx=0, verbose=False):
    """ Setup environment and return info  """
    # get the default brain
    brain_name = env.brain_names[brain_idx]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=train_mode)[brain_name]

    # examine the state space and action space
    state = env_info.vector_observations[idx]
    state_size = len(state)    
    action_size = brain.vector_action_space_size
    
    if verbose:
        # number of agents in the environment
        print(f'Number of agents: {len(env_info.agents)}')
        print(f'Number of actions: {action_size}')
        print(f'States have length: {state_size}')
        print(f'States look like: {state}')
        
    return (brain, brain_name, state, action_size, state_size)

def env_reset(env, brain_name, train_mode=True, idx=0):
    """ Reset environment and get initial state """
    
    # reset the environment
    env_info = env.reset(train_mode=train_mode)[brain_name]
    state = env_info.vector_observations[idx]
    
    return state

def state_reward_done_unpack(env_info, idx=0):
    """ Unpacks the state, rewards and done signal from the environment info """
    s1 = env_info.vector_observations[idx] # get the next state
    r1 = env_info.rewards[idx]             # get the reward
    done = env_info.local_done[idx]        # see if episode has finished
    return (s1, r1, done)

# scoring    
def moving_avg(scores, window=100):
    cumsum_vec = np.cumsum(np.insert(scores, 0, 0))
    avg_vec = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    return avg_vec

def filled_moving_avg(scores, window=100):
    """ 
        Moving average with initial averages based on available
        scores but not the full window width.
    """
    initial = [np.sum(scores[:i+1]) / (i+1) for i in range(window-1)]
    tail = moving_avg(scores, window)
    return np.hstack((initial, tail))

# plots
def plot_line(x, y, style=['Label', 'Color', 1, '-'], is_vert=False):
    if is_vert:
        plt.axvline(x, label=style[0], color=style[1],
                linewidth=style[2], linestyle=style[3])        
    else:
        plt.plot(x, y, label=style[0], color=style[1],
                linewidth=style[2], linestyle=style[3])
    
def plot_training_scores(scores, goal_score, window=100,
                        agent_name=None,
                        ylabel='Score'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line styles used for plot
    lines = {
        'score':['Episode score', 'xkcd:sky blue', 1, '-'],
        'avg':['Average score', 'xkcd:pumpkin', 2, '-'],
        'goal':['Goal score', 'xkcd:rose', 3, '--'],
        'start':['First valid score', 'xkcd:black', 2, ':']
    }
    
    x = np.arange(len(scores))  # x-axis of episode numbers
    goals = np.full_like(scores, goal_score) # goal scores
    
    # running avg over window # of episodes
    avg_scores = filled_moving_avg(scores, window)
    
    # plot the episode scores
    plot_line(x, scores, lines['score'])
    plot_line(x, avg_scores, lines['avg'])
    plot_line(x, goals, lines['goal'])
    
    # plot the scoring start frame
    plot_line(window, None, lines['start'], is_vert=True)
    
    # labels and formatting
    plt.xlabel('Episode #')
    plt.ylabel(ylabel)
    title = 'Agent Training Results' if agent_name is None else f'Training Results for \'{agent_name}\''
    plt.title(title)
    ax.legend()
    
    # show it
    plt.show()

# save/load
AGENT_SAVE_DIR = 'saved_agents'
# each /<Agent_Name>/ directory
CHECKPOINT_DIR = '.checkpoints'

PARAMS_FILENAME = 'params.json'
SCORES_FILENAME = 'scores.npy'

def _save_dir(agent_name, i_checkpoint=None):
    if i_checkpoint is None:
        save_dir = f'{AGENT_SAVE_DIR}/{agent_name}'
    else:
        save_dir = f'{AGENT_SAVE_DIR}/{agent_name}/{CHECKPOINT_DIR}'
    return save_dir

def _weight_filename(weight_type, i_checkpoint=None):
    if i_checkpoint is None:
        filename = f'{weight_type}_weights.pyt'
    else:
        filename = f'{i_checkpoint}-ckpt_{weight_type}_weights.pyt'    
    return filename

def _scores_filename(i_checkpoint=None):
    if i_checkpoint:
        filename = f'{i_checkpoint}-{SCORES_FILENAME}'
    else:
        filename = SCORES_FILENAME
    return filename

def save_weights(agent_name, weights, weight_type, i_checkpoint=None, verbose=False):
    save_dir = _save_dir(agent_name, i_checkpoint=i_checkpoint)
    filename = _weight_filename(weight_type, i_checkpoint=i_checkpoint)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(weights, f'{save_dir}/{filename}')
    if verbose:
        print(f'Saved {filename} at {save_dir}/')

def load_weights(agent_name, weight_type, i_checkpoint=None, verbose=False):
    load_dir = _save_dir(agent_name, i_checkpoint=i_checkpoint)
    filename = _weight_filename(weight_type, i_checkpoint=i_checkpoint)

    # todo load
    weights = torch.load(f'{load_dir}/{filename}', map_location='cpu')
    if verbose:
        print(f'Loaded {weight_type} weights from {filename} at {load_dir}')
    
    return weights

def save_params(agent_name, params, verbose=False):
    save_dir = _save_dir(agent_name)
    filename = PARAMS_FILENAME

    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{filename}', 'w') as f:
        json.dump(params, f)
    if verbose:
        print(f'Saved agent parameters at {save_dir}/')

def load_params(agent_name, verbose=False):
    load_dir = _save_dir(agent_name)
    filename = PARAMS_FILENAME

    with open(f'{load_dir}/{filename}', 'r') as f:
        params = json.load(f)
    if verbose:
        print(f'Loaded agent parameters from {load_dir}/')

    return params

def save_scores(agent_name, scores, i_checkpoint=None, verbose=False):
    save_dir = _save_dir(agent_name, i_checkpoint=i_checkpoint)
    filename = _scores_filename(i_checkpoint=i_checkpoint)

    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/{filename}', scores)
    if verbose:
        print(f'Saved scores as {filename} at {save_dir}/')

def load_scores(agent_name, i_checkpoint=None, verbose=False):
    load_dir = _save_dir(agent_name, i_checkpoint=i_checkpoint)
    filename = _scores_filename(i_checkpoint=i_checkpoint)

    scores = np.load(f'{load_dir}/{filename}')
    if verbose:
        print(f'Loaded scores from {load_dir}/')
    return scores

def save_checkpoint(agent_name, i_episode, local_weights, target_weights, verbose=False):
    save_weights(agent_name, local_weights, 
                'local', i_checkpoint=i_episode, verbose=verbose)
    save_weights(agent_name, target_weights, 
                'target', i_checkpoint=i_episode, verbose=verbose)
    if verbose:
        print(f'Saved checkpoint at {_save_dir(agent_name, i_episode)}/')

def save_dqn(agent_name, local_weights=None, target_weights=None, 
            params=None, scores=None, i_checkpoint=None, verbose=False):
    """ 
        Saves the local and target weights, and 
        (optionally) parameters and scores for a DQN agent.
    """
    if local_weights is not None:
        save_weights(agent_name, local_weights, 'local', 
                    i_checkpoint=i_checkpoint, verbose=verbose)
    if target_weights is not None:
        save_weights(agent_name, target_weights, 'target', 
                    i_checkpoint=i_checkpoint, verbose=verbose)

    if params is not None:
        save_params(agent_name, params, verbose=verbose)
    
    if scores is not None:
        save_scores(agent_name, scores, 
                    i_checkpoint=i_checkpoint, verbose=verbose)
    #if verbose:
    #    print(f'Saved DQN agent \'{agent_name}\' at {_save_dir(agent_name, i_checkpoint)}/')

def load_dqn(agent_name, i_checkpoint=None, verbose=False):
    """
        Loads the parameters, local and target 
        weights for a DQN agent.
    """
    params = load_params(agent_name, verbose=verbose)
    local_weights = load_weights(agent_name, 'local', i_checkpoint=i_checkpoint, verbose=verbose)
    target_weights = load_weights(agent_name, 'target', i_checkpoint=i_checkpoint, verbose=verbose)
    if verbose:
        print(f'Loaded DQN agent \'{agent_name}\' from {_save_dir(agent_name, i_checkpoint)}/')
    
    return params, local_weights, target_weights

def save_ddpg(agent_name, 
            params=None, scores=None,
            actor_weights=None, critic_weights=None,
            i_checkpoint=None, verbose=False):
    if params is not None:
        save_params(agent_name, params, verbose=verbose)
    
    if scores is not None:
        save_scores(agent_name, scores,
            i_checkpoint=i_checkpoint, verbose=verbose)
    
    if actor_weights is not None:
        save_weights(agent_name, actor_weights, 'actor', 
            i_checkpoint=i_checkpoint, verbose=verbose)
    
    if critic_weights is not None:
        save_weights(agent_name, critic_weights, 'critic', 
            i_checkpoint=i_checkpoint, verbose=verbose)