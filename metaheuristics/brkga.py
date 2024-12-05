import pandas as pd

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize

from metaheuristics import problems

from pymoo.config import Config
Config.warnings['not_compiled'] = False # disable warning


def run_algorithm(X:pd.DataFrame, y:pd.DataFrame, algorithm_params:dict, optimization_params:dict):
    """Define the problem and run the BRKGA algorithm with the given parameters."""

    # define the feature selection problem
    problem = problems.FeatureSelectionProblem(X, y, optimization_params["fitness_function"])

    algorithm_params["eliminate_duplicates"] = (problems.MyElementwiseDuplicateElimination()
                                            if algorithm_params["eliminate_duplicates"] else None)

    algorithm = BRKGA(**algorithm_params)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", optimization_params["n_gen"]),
        seed=1,
        verbose=False,
        save_history=True
    )

    return res
