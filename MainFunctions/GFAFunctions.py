#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import uniform,randint
from scipy.optimize import minimize_scalar

from joblib import Parallel, delayed
import mlflow


#%% Classical ML and Neural Networks workflow - Green's Function Interpolation

# Global takes GFA, Transient from T1 -> T2 -> T1 to predict, Data Column Names (Input, Output), model (LR, RF, etc) for Prediction, and model parameter names to vary..
# Global returns dataframe of T1 -> T2 -> T1 with prediction and error, alpha -> best scalar optimization value, and rmse in the same order
# key for model: 'LR' -> Linear, 'RF' -> Random Forests, 'XGB' -> XGradient Boost, 'MLP' -> Multilayer Perceptron
# 'space' variable for what parameters you want to vary in a specific model. It can be a range/set values. Parameter Sampler will handle both.
 
def Global(dftrain:pd.DataFrame,dfopt:pd.DataFrame,input_cols:list,output_cols:list,model:str,params:dict=None): 

    # Prediction Models available #
    modelDict = {'LR':TransformedTargetRegressor(regressor=Pipeline([('LR_Scaler',StandardScaler()),('LR_model',LinearRegression())]),transformer=StandardScaler()),
                'RF':RandomForestRegressor(n_jobs=-1),'XGB':XGBRegressor(n_jobs=-1),
                'MLP':TransformedTargetRegressor(regressor=Pipeline([('MLP_Scaler',StandardScaler()),('MLP_model',MLPRegressor())]),
                                                                         transformer=StandardScaler())}

    temp_col = input_cols[0] # Temperature Header String
    time_col = input_cols[1] # Time Header String

    def convoluteOptimize(df_convolution:pd.DataFrame, ml_model) -> np.ndarray:
        # Time in s -> Time_s
        # Temperature in Celsius -> InitialTemperature_C
        tmatrix = np.zeros(shape=(len(df_convolution)-1,len(df_convolution)))
        for i in range(tmatrix.shape[0]):
            time = (df_convolution[time_col][i:] - df_convolution[time_col][i]).to_numpy()
            temp = df_convolution[temp_col][i]
            temp_diff = df_convolution[temp_col][i+1] - df_convolution[temp_col][i]

            df_temporary = pd.DataFrame({temp_col:temp,time_col:time})
            tmatrix[i][i:] = ml_model.predict(df_temporary)*temp_diff
        conv_stress = tmatrix.sum(axis=0)

        # Finding scalar for prediction error minimization: (error = alpha * Predicted Stress - VonMises_MPa)
        err = lambda a: np.sum(np.abs(a*conv_stress - df_convolution[output_cols[0]])) 
        alpha = minimize_scalar(err,bounds=(0.5,1.5),method='bounded').x
        return alpha*conv_stress,alpha
    
    # Chosen model is fit with data # 
    if model not in modelDict.keys(): 
        return 'Model not in database'

    chosen_model = modelDict[model]

    if params:
        chosen_model.set_params(**params)
    
    chosen_model.fit(dftrain[input_cols],dftrain[output_cols[0]])

    # Stress Prediction # 
    dfopt['Predicted_Stress'], alpha = convoluteOptimize(dfopt,chosen_model)
    dfopt['Squared Error'] = (dfopt['Predicted_Stress'] - dfopt[output_cols[0]])**2
    rmse = ((dfopt['Squared Error']).mean())**0.5
    return dfopt, alpha, rmse






