# Policy-Reacher-Critic

Deep Deterministic Policy Gradient (DDPG) algorithms used to solve a continuous control problem where a robotic arm reaches for a moving sphere.

![Trained agent in environment](solved_armoboto.gif)

## Environment Overview

The environment contains a 20 agents each with their own task, to reach out and touch a moving sphere target. Each agent can choose how to move at the shoulder and elbow joints, and receives feedback from the environment based on the observed state and reward signal. The agent is rewarded for placing it's hand in the goal location.

### State Space

The state space for each agent contains 33 variables for the position, rotation, velocity, and angular velocities of the arm. All of the values are continuous.

### Action Space

Each agent sends actions to the environment in the form of a vector with four numbers, representing the torque to apply to the two joints, where each value is between -1 and 1.

### Rewards and Scoring

The rewards for this environment focus on the position of the agents hands and the moving goal target.

* ```+ 0.1``` for each step the agent's hand is in the goal position

Scoring for this multi agent episodic task is done by taking the average score of all agent each episode to produce an episode score, and then taking that average episode score over 100 consecutive episodes.

For this project to be considered solved, the agent must achieve an average episode score over 100 consecutive episodes of +30.

## Getting Started

Setup the project with the ```git clone``` -> ```conda create [virtualenv]``` -> ```pip install [dependencies]``` workflow, outlined below in detail.

Additionally you'll need to download the Unity Environment solved in this project, links provided below.

### Installation

1. clone the repository:

    ``` bash
    git clone https://github.com/TacticSiege/Policy-Reacher-Critic
    cd Policy-Reacher-Critic
    ```

2. Create a virtualenv with Anaconda and activate it:

    * Linux or Mac:

    ``` bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

    * Windows:

    ``` bash
    conda create --name drlnd python=3.6
    activate drlnd
    ```

3. Install project dependencies:

    ``` bash
    cd python/
    pip install .
    cd ..
    ```

4. Download the Unity Environment for your OS:

    | Operating System | Link |
    |------------------|------|
    | Windows (32bit) | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) |
    | Windows (64bit) | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) |
    | MacOS | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) |
    | Linux | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) |

    Extract the archive into the project directory, or you may update the ```env_path``` in the notebook(s) to use a custom directory.

5. (Optional) Associate a Jupyter Notebook kernel with our virtualenv:

    ``` bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

    If you forget to activate your virtualenv, you can choose the kernel created above from the Jupyter Notebook UI.

6. Run Jupyter Notebook and explore the repo:

    ``` bash
    jupyter notebook
    ```

### Running the Agent

* Train the agent using ```TrainAgent.ipynb```

* Watch a trained agent by loading saved model weights in the last few cells of the notebook.  This repo contains saved model weights for the solution agent, ```Agent Armoboto```, that are already setup to run.

See ```Report.md``` for more details on implementation and training results.
