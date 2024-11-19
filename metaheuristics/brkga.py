
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize

from pymoo.config import Config
Config.warnings['not_compiled'] = False # disable warning


def run_algorithm(problem:type[ElementwiseProblem], algorithm_params:dict, optimization_params:dict):
    """Runs the BRKGA algorithm with the given problem and parameters."""

    algorithm = BRKGA(**algorithm_params)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", optimization_params["n_gen"]),
        seed=1,
        verbose=False
    )

    return res
