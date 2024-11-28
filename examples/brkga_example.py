import sys
import os
import numpy as np

from sklearn.datasets import load_breast_cancer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaheuristics import brkga, problems


def main():

    # example 1
    # n_samples = 100
    # n_features = 500 # CNN output size
    # X = np.random.rand(n_samples, n_features) # feature matrix: n_samples x n_features (random samples from a uniform distribution over [0, 1))
    # y = np.random.randint(0, 2, size=n_samples) # random labels for classification

    # example 2
    cancer = load_breast_cancer()
    X = cancer["data"]
    y = cancer["target"]
    print(f"Input data: \nX: {X.shape} and y: {y.shape}")

    # BRKGA parameters
    brkga_parameters = {
        "n_elites": 100,
        "n_offsprings": 150,
        "n_mutants": 50,
        "bias": 0.7,
        "eliminate_duplicates": problems.MyElementwiseDuplicateElimination()
    }

    # optimization parameters
    opt_parameters = {
        "n_gen": 4 # max generations for testing
    }

    # define the feature selection problem
    problem = problems.FeatureSelectionProblem(X, y, fitness_function="accuracy")

    res = brkga.run_algorithm(
        problem=problem,
        algorithm_params=brkga_parameters,
        optimization_params=opt_parameters
    )

    # output
    best_solution_fitness = res.F
    best_solution = np.array(res.X)
    selected_features = np.where(best_solution > 0.5)[0] # threshold for binary choice

    print(f"Best solution found: \nX = {best_solution}\nF = {best_solution_fitness}")
    print("Number of selected features:", len(selected_features))
    print("Selected features:", selected_features)


if __name__ == "__main__":
    main()
