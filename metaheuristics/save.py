import os
import numpy as np
import pandas as pd

def brkga_history_to_file(filename, res):
    evaluations = np.array([e.evaluator.n_eval for e in res.history])
    fitness_history = np.array([e.opt[0].F for e in res.history])

    data = np.column_stack((evaluations, fitness_history))
    np.savetxt(filename, data, delimiter=",", header="n_evaluations,fitness_history", comments="", fmt="%.6f")

def brkga_best_solution_to_file(filename, column_name, best_solution):
    new_col_df = pd.DataFrame(best_solution, columns=[column_name])

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, new_col_df], axis=1)
    else:
        df = new_col_df

    df.to_csv(filename, index=False)

def experiment_results_to_file(filename, results):
    row_df = pd.DataFrame([results])

    if os.path.exists(filename):
        row_df.to_csv(filename, mode="a", header=False, index=False)
    else:
        row_df.to_csv(filename, mode="w", header=True, index=False)
