import argparse
import numpy

import utils
from utils import device

from attrdict import AttrDict

import pickle

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=False, default = None,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--render", action="store_true", default=False,
                    help="render")
parser.add_argument("--manual", action="store_true", default=False,
                    help="manual control")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

if args.model:
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, use_memory=args.memory, use_text=args.text)
assert args.model or args.manual
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

exps = {'obs': [], 'action': [], 'reward': [], 'done': []}
exps_episodes = []

while len(exps_episodes) < args.episodes:
    obs = env.reset()
    done = False
    tot_rew = 0
    exps_episode = {'obs': [], 'action': [], 'reward': [], 'done': []}
    while True:
        if args.render:
            env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
        if not args.manual:
            action = agent.get_action(obs)
        else:
            action = int(input())
        exps_episode['obs'].append(obs)
        exps_episode['action'].append(action)
        exps_episode['done'].append(int(done))
        
        obs, reward, done, info = env.step(action)
        exps_episode['reward'].append(reward)
        tot_rew += reward
        
        if not args.manual:
            agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            if tot_rew > 0.9:
                exps_episodes.append(exps_episode)
            break
    print('tot_rew: ', tot_rew)

    if env.window.closed:
        break

for key in list(exps.keys()):
    for exps_episode in exps_episodes:
        exps[key] = exps[key] + exps_episode[key]


if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")

pickle.dump(exps, open(f'exp_demo_{args.env}.p', 'wb'))