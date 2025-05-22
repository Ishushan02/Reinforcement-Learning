import flappy_bird_gymnasium
import gymnasium
from deep_qlearning.dqn import DQN
import torch
from deep_qlearning.experience_replay import ReplayMemory
import yaml
import random

DEVICE = torch.device("mps")

class Agent:

    def __init__(self, hyperparameter_set):
        with open('./hyperparameters.yml', 'r') as file:
            all_hyperparameter = yaml.safe_load(file)
            hyperparameters = all_hyperparameter[hyperparameter_set]
        
        self.replay_memory = hyperparameters["replay_memory_size"]
        self.epsilon_init  = hyperparameters["epsilon_init"]
        self.epsilon_decay  = hyperparameters["epsilon_decay"]
        self.epsilon_min   = hyperparameters["epsilon_min"]


    def run(self, episodes=10000, is_training = True, render = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)CartPole-v1
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else False)

        # print(env.action_space.n, env.observation_space.shape[0], env.render_mode)
        input_state = env.observation_space.shape[0]
        output_action = env.action_space.n

        policy_network = DQN(stateDimension=input_state, hiddenDimension=256, outActions=output_action).to(device=DEVICE)
        rewards_per_episode = []
        epsilon_history = []
        if is_training:
            memory = ReplayMemory(self.replay_memory)
            epsilon = self.epsilon_init

        
        
        for each_episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0

            isTerminated = False
            

            while not isTerminated:
                # Next action:
                # (feed the observation to your agent here)

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = policy_network(state).argmax()

                new_state, reward, terminated, _, info = env.step(action)
                # print(obs)
                # print(reward)
                # print(info)
                # print(terminated)
                # break
                if is_training:
                    memory.append([state, action, new_state, reward, terminated])

                episode_reward += reward
                state = new_state
                
                # Checking if the player is still alive
                isTerminated = terminated
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)
            
            env.close()
            rewards_per_episode.append(episode_reward)

            