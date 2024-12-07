import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from metaheuristics import fitness_functions

class FeatureSelectionProblem(ElementwiseProblem):

    def __init__(self, X, y, fitness_function, threshold_decoding, classifier=None):
        super().__init__(
            n_var=X.shape[1],   # dimension of the problem
            n_obj=1,            # number of objective functions
            n_constr=0,         # unconstrained optimization problem
            xl=X.min(axis=0),   # lower bound
            xu=X.max(axis=0)    # upper bound
        )
        self.X = X
        self.y = y
        self.fitness_function = fitness_function
        self.threshold_decoding = threshold_decoding
        self.classifier = classifier

    def _evaluate(self, x, out, *args, **kwargs):
        selected_features = np.where(x > self.threshold_decoding)[0] # threshold for binary choice
        out["F"] = fitness_functions.execute(
            self.fitness_function,
            selected_features,
            self.X,
            self.y,
            self.classifier
        )  # objective value


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        solution_a = np.array(a.get("X"))
        solution_b = np.array(b.get("X"))
        return np.array_equal(solution_a, solution_b)
