import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from Configs.Config import Config
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from typing import Any
import pandas as pd

def do_randomized_search(X_train:np.ndarray, Y_train:np.ndarray, param_grid:dict, model:Any) -> None:   
    RS = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, scoring='f1', n_jobs=-1)
    RS.fit(X_train, Y_train)
    print(RS.best_estimator_)
    print(RS.best_params_)
    print(RS.best_score_)
    RS_df_results = pd.DataFrame(RS.cv_results_)
    RS_df_results = RS_df_results.sort_values(by='f1')
    RS_df_results.to_csv(Definitions.SRC + 'models/tests/RS_results.csv', index=False)


def do_grid_search(X_train:np.ndarray, Y_train:np.ndarray, param_grid:dict, model:Any) -> None:
    GS = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    GS.fit(X_train, Y_train)
    print(GS.best_estimator_)
    print(GS.best_params_)
    print(GS.best_score_)
    GS_df_results = pd.DataFrame(GS.cv_results_)
    GS_df_results = GS_df_results.sort_values(by='score')
    GS_df_results.to_csv(Definitions.SRC + 'models/tests/GS_results.csv', index=False)


if __name__ == '__main__':
    # instancie o objeto do modelo aqui
    # chame a função do_grid_search ou do_randomized_search
    pass
