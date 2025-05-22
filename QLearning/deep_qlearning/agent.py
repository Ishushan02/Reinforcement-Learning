import flappy_bird_gymnasium
import gymnasium
from deep_qlearning.dqn import DQN
import torch

DEVICE = torch.device("mps")

class Agent:

    def run(self, is_training = True, render = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)CartPole-v1
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else False)

        # print(env.action_space.n, env.observation_space.shape[0], env.render_mode)
        input_state = env.observation_space.shape[0]
        output_action = env.action_space.n

        policy_network = DQN(stateDimension=input_state, hiddenDimension=256, outActions=output_action).to(device=DEVICE)

        obs, _ = env.reset()
        
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            print(obs)
            print(reward)
            print(info)
            print(terminated)
            break
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()