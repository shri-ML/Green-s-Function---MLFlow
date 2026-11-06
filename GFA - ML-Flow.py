#%%
import pandas as pd
import numpy as np

from scipy.stats import randint
from sklearn.model_selection import ParameterSampler
from MainFunctions.GFAFunctions import Global
from joblib import Parallel, delayed
import mlflow

########################################
#        READ BEFORE RUNNING           #
########################################
# Global (..) -> Trains the chosen model on GFA data, predicts stress for a
# transient profile, and scores the prediction against actual FEA results.
#
# Global(dft, dfop, input_cols, output_cols, model_key, params)
#   dft       : GFA (Green's Function) matrix (25 - 26)
#   dfop      : Transient profile (T1 → T2 → T1) to predict against
#   input_cols: list of input column names
#   output_cols: list of output column names
#   model_key : 'LR' (Linear), 'RF' (Random Forest), 'XGB' (XGBoost), 'MLP' (MLP)
#   params    : dict of hyperparameters to pass to the model
#
# Returns: (DataFrame, alpha, RMSE)
#   DataFrame : T1 → T2 → T1 with predicted stress and error columns
#   alpha     : best scalar from optimization
#   RMSE      : root mean squared error of predictions
#
# 'space' is a dict passed to ParameterSampler. Each key is a
#   hyperparameter name; each value is a range (e.g. uniform(0,1))
#   or a fixed list (e.g. [100, 200, 300]). ParameterSampler draws
#   random combinations from it.
########################################
 
#%%
# Input Files, Column Names for Data
# Uniform Temperature Across geometry # 
dft0 = pd.read_excel("./InputData/50-400-50/vonMises_25s_600s_Shri.xlsx",header=0)
dfop0 = pd.read_excel("./InputData/50-400-50/Transient_Checker - 50-400-50_Shri.xlsx",header=0)

# Non-uniform Temperature Across geometry # 
dft1 = pd.read_excel("./InputData/Non-Uniform-22-400-22/green_non uniform.xlsx", header=0)
dfop1 = pd.read_excel("./InputData/Non-Uniform-22-400-22/transient_non uniform.xlsx", header=0)

input_colx = ['InitialTemperature_C','Time_s']
output_colx = ['VonMises_MPa']

# Sample Parameters  
spaceRF = {'max_depth':randint(10,100),'n_estimators':randint(100,300),'random_state':[42],'verbose':[True]}
spaceMLP = {'regressor__MLP_model__activation':['relu'],'regressor__MLP_model__alpha':[0.01],'regressor__MLP_model__verbose':[True],'regressor__MLP_model__hidden_layer_sizes':[(100,100)]}

# Parameter Sampler: Will give combinations of parameters that you define, based on number of iterations. Choose model and configure parameters above. 
listOfComb = list(ParameterSampler(spaceRF,n_iter=1,random_state=42))
listOfComb1 = list(ParameterSampler(spaceMLP,n_iter=1,random_state=42))

#%% ML-Flow Setup
# For Uniform Temperature #
# Set Experiment Name -> Start Run -> Start Run as a child of Parent 
mlflow.set_experiment('GFA-Runs')
with mlflow.start_run(run_id='d13162da16ea4253b6046f58923bfd2e'):
    for i in listOfComb1:
        with mlflow.start_run(nested=True,run_name='MLP'):
            Df1, alpha1, Rmse1 = Global(dft0,dfop0,input_colx,output_colx,'MLP',i)
            mlflow.log_params(i)
            mlflow.log_metrics({'alpha':alpha1,'rmse':Rmse1})
            mlflow.log_text(Df1.to_csv(index=False),'results.csv')

#%% ML-Flow Setup
# For Non-Uniform Temperature #
# Set Experiment Name -> Start Run -> Start Run as a child of Parent 
mlflow.set_experiment('GFA-Runs')
with mlflow.start_run(run_id='98efa21ec3ca4ae38382f695a2317359'):
    for j in listOfComb1:
        with mlflow.start_run(nested=True,run_name='ML_Non-Uniform (22-400-22)'):
            Df, alpha0, Rmse = Global(dft1,dfop1,input_colx,output_colx,'MLP',j)
            mlflow.log_params(i)
            mlflow.log_metrics({'alpha':alpha0,'rmse':Rmse})
            mlflow.log_text(Df.to_csv(index=False),'results.csv')

#%% Optuna Setup (trial is logged into ML-Flow)
#%%
# Optuna + MLflow: MLP alpha (0.05-0.25) x hidden layer width (100-200)
import optuna
import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('GFA-Runs')

def objective(trial):
    alpha = trial.suggest_float('alpha', 0.05, 0.25)
    width = trial.suggest_int('width', 100, 200)

    params = {
        'regressor__MLP_model__activation': 'relu',
        'regressor__MLP_model__alpha': alpha,
        'regressor__MLP_model__hidden_layer_sizes': (width, width),
        'regressor__MLP_model__random_state': 42,
        'regressor__MLP_model__verbose': False,
    }

    with mlflow.start_run(nested=True, run_name=f'Trial-{trial.number}'):
        Df, alpha0, Rmse = Global(dft1, dfop1, input_colx, output_colx, 'MLP', params)
        mlflow.log_params(params)
        mlflow.log_metrics({'alpha': alpha0, 'rmse': Rmse})
        mlflow.log_text(Df.to_csv(index=False), 'results.csv')

    return Rmse

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)

with mlflow.start_run(run_name='Optuna-MLP_Non-Uniform (22-400-22)'):
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                storage='sqlite:///optuna_gfa.db',
                                study_name='mlp_nonuniform', load_if_exists=True)
    study.optimize(objective, n_trials=15)

    mlflow.log_params({f'best_{k}': v for k, v in study.best_trial.params.items()})
    mlflow.log_metric('best_rmse', study.best_value)

print('Best:', study.best_trial.params, '| RMSE:', study.best_value)