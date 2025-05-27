# import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import yaml 
import argparse
import random
from torch import nn
import torch.optim as optim
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "./logs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen


# DEVICE = torch.device("mps")
DEVICE = torch.device("cpu")

class Agent:

    def __init__(self, hyperparameter_set):
        with open('./hyperparameters.yml', 'r') as file:
            all_hyperparameter = yaml.safe_load(file)
            hyperparameters = all_hyperparameter[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters["env_id"]
        self.replay_memory = hyperparameters["replay_memory_size"]
        self.epsilon_init  = hyperparameters["epsilon_init"]
        self.epsilon_decay  = hyperparameters["epsilon_decay"]
        self.epsilon_min   = hyperparameters["epsilon_min"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.stop_on_reward     = hyperparameters['stop_on_reward']# stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag

        # Neural Network
        self.lossFn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        


    def run(self, episodes=10000, is_training = True, render = False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')


        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)CartPole-v1
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)#

        # print(env.action_space.n, env.observation_space.shape[0], env.render_mode)
        input_state = env.observation_space.shape[0]
        output_action = env.action_space.n

        policy_network = DQN(stateDimension=input_state, hiddenDimension=256, outActions=output_action).to(device=DEVICE)
        rewards_per_episode = []
        epsilon_history = []
        if is_training:
            memory = ReplayMemory(self.replay_memory)
            epsilon = self.epsilon_init
            target_network = DQN(stateDimension=input_state, hiddenDimension=256, outActions=output_action).to(device=DEVICE)
            target_network.load_state_dict(policy_network.state_dict())
            self.optimizer = optim.Adam(params=policy_network.parameters(), lr=self.learning_rate)
            step_count = 0
            best_reward = -9999999
        else:
            # Load learned policy
            policy_network.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_network.eval()
        
        
        for each_episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            

            episode_reward = 0.0

            isTerminated = False
            

            while not isTerminated:
                # Next action:
                # (feed the observation to your agent here)

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=DEVICE)
                else:
                    with torch.no_grad():
                        action = policy_network(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)


                # print(obs)
                # print(reward)
                # print(info)
                # print(terminated)
                # break
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                episode_reward += reward
                state = new_state
                
                # Checking if the player is still alive
                isTerminated = terminated
                

            rewards_per_episode.append(episode_reward)
            
            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {each_episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_network.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                # print("just checking: ", episode_reward )


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if(len(memory) > self.mini_batch_size):
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_network, target_network)
                        epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                        epsilon_history.append(epsilon)

                        if(self.network_sync_rate > step_count):
                            target_network.load_state_dict(policy_network.state_dict())
                            step_count = 0

            env.close()
            

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_network, target_network):
        
        # for state, action, newState, reward, terminated in mini_batch:
            
        #     if terminated:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor * target_network(newState).max()
            
        #     currentQ = policy_network(state)

        #     self.lossFn(currentQ, target_q)

        #     self.optimizer.zero_grad()
        #     self.lossFn.backward()
        #     self.optimizer.step()

        # optimized way
        # print(mini_batch)
        states, actions, newStates, rewards, terminated = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        newStates = torch.stack(newStates)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated).float().to(DEVICE)

        
        with torch.no_grad():
            if(self.enable_double_dqn):
                best_action_from_policy = policy_network(newStates).argmax(dim=1)
                target_batch_Q = rewards + (1-terminated) * self.discount_factor * target_network(newStates).gather(dim = 1, index = best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_batch_Q = rewards + (1-terminated) * self.discount_factor * target_network(newStates).max(dim = 1)[0]

        current_batch_Q = policy_network(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze() 

        
        loss = self.lossFn(current_batch_Q, target_batch_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__=='__main__':
    # agent = Agent("caratpole1")
    # agent.run(render=True)

    parser = argparse.ArgumentParser(description='Train or test model.')
    print(parser)
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)