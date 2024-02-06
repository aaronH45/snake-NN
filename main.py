import torch
from pygad import torchga
import pygad
import gym_snake_game
import gym
import numpy as np
import random

env = gym.make("Snake-v0")


def run_game(model, turn_limit = 1_000) -> float:
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
        # fitness += reward
        turns += 1

        if done or turns > turn_limit:
            break
    
    # print(env.snake.score)
    fitness = env.snake.score*100 + turns*0.2

    return fitness




def fitness_func(ga_instance, solution, sol_idx):
    global torch_ga, model, loss_function
    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # random.seed(10)
    sample_fitness = [run_game(model) for _ in range(5)]
    solution_fitness = np.mean(sample_fitness)
    # solution_fitness = run_game(model)

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Build the PyTorch model.
input_layer = torch.nn.Linear(6, 16)
layer_2 = torch.nn.Linear(16, 16)
output_layer = torch.nn.Linear(16, 4)
model = torch.nn.Sequential(input_layer, layer_2, output_layer, torch.nn.Softmax())


# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.CrossEntropyLoss()

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 600 # Number of generations.
num_parents_mating = 3 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 20 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)

env = gym.make("Snake-v0", render_mode="human")
model.load_state_dict(best_solution_weights)

# random.seed(10)
score = run_game(model, turn_limit=10_000)
print(score)

torch.save(model, 'best_model2.pt')