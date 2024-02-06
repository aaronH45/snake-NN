import torch
import gym_snake_game
import gym
import numpy as np
import random

model = torch.load("best_model.pt")


env = gym.make("Snake-v0", render_mode="human")

def run_game(model, turn_limit = 1000) -> float:
    init = env.reset()
    fitness = 0
    obs = init[0]
    turns = 0

    while True:
        input = torch.tensor(obs, dtype=torch.float32)
        out = model(input)
        # move = np.random.choice([0, 1, 2, 3], p=out.detach().numpy())
        move = torch.argmax(out).detach().numpy()

        obs, reward, done, truncated, info = env.step(move)
        # board = info["map"]
        # for i in board:
        #     print(i)
        # break
        # fitness += reward
        turns += 1

        if done or turns > turn_limit:
            break
    
    # print(env.snake.score)
    fitness = env.snake.score*100 + turns*0.1
    return fitness

score = run_game(model, turn_limit=9999)
print(score)

# for param in model.parameters():
#   print(param.data)
