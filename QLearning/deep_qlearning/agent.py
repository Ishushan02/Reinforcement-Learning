import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import yaml
import random
from torch import nn
import torch.optim as optim

DEVICE = torch.device("mps")

class Agent:

    def __init__(self, hyperparameter_set):
        with open('./QLearning/deep_qlearning/hyperparameters.yml', 'r') as file:
            all_hyperparameter = yaml.safe_load(file)
            hyperparameters = all_hyperparameter[hyperparameter_set]
        
        self.replay_memory = hyperparameters["replay_memory_size"]
        self.epsilon_init  = hyperparameters["epsilon_init"]
        self.epsilon_decay  = hyperparameters["epsilon_decay"]
        self.epsilon_min   = hyperparameters["epsilon_min"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.learning_rate = hyperparameters["learning_rate"]
        


    def run(self, episodes=10000, is_training = True, render = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)CartPole-v1
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)#

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
            self.lossFn = nn.MSELoss()
            step_count = 0

        
        
        for each_episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            

            episode_reward = 0.0

            isTerminated = False
            

            while not isTerminated:
                # Next action:
                # (feed the observation to your agent here)

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=DEVICE)
                else:
                    with torch.no_grad():
                        action = policy_network(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, info = env.step(int(action.item()))
                new_state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)

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
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)

                if(len(memory) > self.mini_batch_size):
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_network, target_network)

                    if(self.network_sync_rate > step_count):
                        target_network.load_state_dict(policy_network.state_dict())
                        step_count = 0


            
            env.close()
            rewards_per_episode.append(episode_reward)

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
        terminated = torch.stack(terminated).float().to(DEVICE)

        
        with torch.no_grad():
            target_batch_Q = rewards + (1-terminated) * self.discount_factor * target_network(newStates).max(dim = 1)[0]
            
        current_batch_Q = policy_network(states).gather(dim =  1, index = actions.unsqueeze(dim=1)).squeeze() 

        
        self.lossFn(current_batch_Q, target_batch_Q)
        self.optimizer.zero_grad()
        self.lossFn.backward()
        self.optimizer.step()



if __name__=='__main__':
    agent = Agent("caratpole1")
    agent.run(render=True)