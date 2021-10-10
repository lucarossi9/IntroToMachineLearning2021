# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:25:33 2021 

Intro to Machine learning for Civil engineers (and one Mathematician,
                        one Physician and one Mechanical Engineer)

@author: LMS, LR, J
"""


#%% NEEDS TO RUN (1 of 3)
#Get necessary packages and Toolboxes and change directory
# for data
import pandas as pd
import numpy as np
import os


# for plotting
import matplotlib.pyplot as plt



## for machine learning

# for Error Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# for Hyperparameter Search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
from skopt import dump

# for Knn
from sklearn.neighbors import KNeighborsRegressor


# for bagged Knn
from sklearn.ensemble import BaggingRegressor


# for random forest 
from sklearn.ensemble import RandomForestRegressor


#for Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

#for gradient boosted random forest
from sklearn.ensemble import GradientBoostingRegressor

#for xtreme Gradient boosted regressor
import xgboost as xgb
#%% NEEDS TO RUN (2 of 3)
# Data preparation 

# Import of data direction needs to be set to a folder which contains the following files: test_set.csv, train_set.csv, val_set.csv
test_set = pd.read_csv('test_set.csv')
train_set = pd.read_csv('train_set.csv')
val_set=test = pd.read_csv('val_set.csv')

# Convert from dataframe to numpy array
train = train_set.to_numpy()
xTest = test_set.to_numpy()
val = val_set.to_numpy()


# Get periods of spectral acceleration
headerAll = list(train_set.columns.values)
periods = np.array(headerAll[:105])

# Find period of maximum spectral acceleration for training set
maxValuesTrain = np.amax(train[:,:-5],axis=1)
maxValuesPerTrain = periods[(np.argmax(train[:,:-5],axis=1))]
maxValuesPerTrain=maxValuesPerTrain.astype(np.float)

# Find period of maximum spectral acceleration for validation set
maxValuesVal = np.amax(val[:,:-5],axis=1)
maxValuesPerVal = periods[(np.argmax(val[:,:-5],axis=1))]
maxValuesPerVal=maxValuesPerVal.astype(np.float)

# Find period of maximum spectral acceleration for test set
maxValuesTest = np.amax(xTest[:,:-5],axis=1)
maxValuesPerTest = periods[(np.argmax(xTest[:,:-5],axis=1))]
maxValuesPerTest = maxValuesPerTest.astype(np.float)


# Append Max_Value and Period_of_Max to dataframe 
train_set.insert(110,"Period_of_Max", maxValuesPerTrain)
del train_set["da5_75"]
val_set.insert(110,"Period_of_Max", maxValuesPerVal)
del val_set["da5_75"]
test_set.insert(110,"Period_of_Max", maxValuesPerTest)
del test_set["da5_75"]


# Convert Dataframe into np.array (this time including the period of the maximum spectral acceleration)
train = train_set.to_numpy()
xTest = test_set.to_numpy()
val = val_set.to_numpy()

# Divide data in x and y values
xTrain = train[:, :-1]
yTrain = train[:, -1]
xVal = val[:, :-1]
yVal = val[:, -1]


# Store all the features after the spectral accelerations in different variables
xTrainE=xTrain[:,105:]
xValE=xVal[:,105:]
xTestE=xTest[:,105:]

""" 
Calculate spectral acceleration at periods which have the same distance between them,
e.g. if k=100, the step size for the periods would be 10s/100=0.1s. So we would have 
a spectral acceleration for 0.1s, 0.2s ,0.3s, ....., 10s
"""

K=80
periods=np.float64(periods)
xvals=np.linspace(0.01, 10, K)

# Create K equaly spaced points using a linear interpolation for the training set
yintTrain=np.zeros([xTrain.shape[0], K])
for i in range(xTrain.shape[0]):
    yintTrain[i,:]=np.float32(np.interp(xvals,periods,xTrain[i,:105]))

# Create K equaly spaced points using a linear interpolation for the training set
yintVal=np.zeros([xVal.shape[0], K])
for i in range(xVal.shape[0]):
    yintVal[i,:]=np.float32(np.interp(xvals,periods,xVal[i,:105]))

# Create K equaly spaced points using a linear interpolation for the training set
yintTest=np.zeros([xTest.shape[0], K])
for i in range(xTest.shape[0]):
    yintTest[i,:]=np.float32(np.interp(xvals,periods,xTest[i,:105]))
    
# Create plot to visualize the effet of what we are doing
fig1 = plt.plot(periods,xTest[264,:105])
fig2 = plt.plot(xvals,yintTest[264,:])
plt.legend(['Original data', 'Data with equidistant points'])
plt.title('Visualization of the difference between the original data and the processed data')
plt.xlabel('Period in seconds')
plt.ylabel('Spectral acceleration in g')


"""
Since we figured, that the impact of the different spectral accelerations is more dependent
on the shape of the plot of the spectral accelerations, rather than the absolute values, we
tried different approaches to make the data comparable
"""


I=80 
for i in range(yintTrain.shape[0]):
    indices = (-yintTrain[i,:]).argsort()[:I] # get indexes for the highest values of yintTrain
    yintTrain[i,:]= np.divide(yintTrain[i,:],(np.sum(yintTrain[i,indices])/I)) #Divide each value of spectral acceleration by the average of the I highest values

    
for i in range(yintVal.shape[0]):
    indices = (-yintVal[i,:]).argsort()[:I] # get indexes for the highest values of yint
    yintVal[i,:]= np.divide(yintVal[i,:],(np.sum(yintVal[i,indices])/I))       #Divide each value of spectral acceleration by the average of the I highest values

    
for i in range(yintTest.shape[0]):
    indices = (-yintTest[i,:]).argsort()[:I] # get indexes for the highest values of yintTrain
    yintTest[i,:]= np.divide(yintTest[i,:],(np.sum(yintTest[i,indices])/I))    #Divide each value of spectral acceleration by the average of the I highest values


xTrain= np.concatenate((yintTrain,xTrainE),axis=1)
xVal= np.concatenate((yintVal,xValE),axis=1)
xTest= np.concatenate((yintTest,xTestE),axis=1)

#%%
"""
Concatenate xTrain and xVal and yTrain and yVal, to to train the final model on more data.But this was not used, since the validation set does not seem
to represent the test set very well, in general the performance on the test set was reduced, when the model was trained on the training and validation set together
"""
xTrain=np.concatenate((xTrain,xVal),axis=0)
yTrain=np.concatenate((yTrain,yVal),axis=0)

#%% NEEDS TO RUN (3 of 3)
# XGBoost 
xgb_reg = xgb.XGBRegressor(max_depth = 8,
                           eta = 0.013184860416758596,
                           reg_lambda = 1.694963215125096,
                           alpha = 0.657268595593835,
                           gamma = 0.6073583569757632,
                           subsample=0.6101061218004574,
                           n_estimators = 619,
                           max_delta_step = 1.1648144144139756,
                           min_child_weight = 2.9528932554016385,
                           verbosity=1)

xgb_reg.fit(xTrain,yTrain)  # fit the XGBRegressor

XGBPred=xgb_reg.predict(xVal) # Predict sat1_col for the validation set
R2XGB = r2_score(yVal,XGBPred)
mseXGB = mean_squared_error(yVal, XGBPred)
XGBPredTest = xgb_reg.predict(xTest)  # Predict sat1_col for the test set
np.savetxt("MLP_LMS.csv", XGBPredTest, delimiter=",",header="sat1_col") # Save XGBPredTest to csv
print ('MSE score for XGB=',mseXGB) # Show MSE for validation set
print ('r2 score for XGB=',R2XGB) # Show r2 score for validation set

#%% Hyperparameter search for the XGBRegressor using forest_minimize (Takes around 6 hours for n_calls=50, dependign on your hardware)

# Define regressor
reg =xgb.XGBRegressor()

# Define parameter space with  min, max, name of feature
space  = [Integer(0, 70, name='max_depth'),
          Real(0, 1, name='eta'),
          Real(1, 3, name='reg_lambda'),
          Real(0, 1, name='alpha'),
          Real(0, 3, name='gamma'),
          Real(0.4, 1, name='subsample'),
          Integer(600, 1000, name='n_estimators'),
          Real(0, 2, name='max_delta_step'),
          Real(0, 3, name='min_child_weight')]

# Define objective, in our case the neg. MSE of a 5-times crossvalidation  
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)
  
    return -np.mean(cross_val_score(reg, xTrain, yTrain, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))

res_gp = forest_minimize(objective,
                     space,
                     n_calls=50,
                     n_initial_points=5,
                     random_state=0,
                     verbose=True)
dump(res_gp,'result.pkl') # Save result


#%% 
"""
After this point, I showed all the different approaches that we were trying out on the problem
but everything that is important to obtain the best result is above this comment 
"""

# Search for Knn

parameters = {'n_neighbors': range(1, 50),
                         'weights': ['uniform', 'distance']}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(xTrain, yTrain)
best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]

#%% Simple Knn-Approach

neigh = KNeighborsRegressor(n_neighbors =70, weights=best_weights)
neigh.fit(xTrain, yTrain)
valPred = neigh.predict(xVal)
R2Knn = r2_score(yVal,valPred)
mseKnn = mean_squared_error(yVal, valPred)
print ('MSE score for KNN=',mseKnn)
print ('r2 score for KNN=',R2Knn)

#%% Bagged Knn-Approach

bagged_knn = KNeighborsRegressor( n_neighbors=100, weights=best_weights)
bagging_model = BaggingRegressor(bagged_knn, n_estimators=20)
bagging_model.fit(xTrain,yTrain)

bagPred = bagging_model.predict(xVal)
R2Bag = r2_score(yVal,bagPred)
mseBag = mean_squared_error(yVal, bagPred)

print ('MSE score for bagKNN=',mseBag)
print ('r2 score for bagKNN=',R2Bag)

#%% Random CV optimization Hyperparameter search for RF (takes a very long time)

n_estimators = [100, 200, 300, 500, 650, 750, 1000]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [5, 10, 20, 30, 50, 80, 100, 130]
max_depth.append(None)
min_samples_split = [2, 5, 10, 40, 50, 100]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid_rf,
                               n_iter = 50,
                               cv = 5,
                               verbose=3,
                               random_state=42,
                               n_jobs = -1,
                               scoring = 'r2',
                               return_train_score = True)

rf_random.fit(xTrain, yTrain)
#%% Random Tree Regression

rf = RandomForestRegressor(n_estimators = 500, 
                           max_depth = 20,
                           min_samples_leaf = 4,
                           min_samples_split = 10,
                           verbose = 1,
                           random_state = 42,
                           oob_score= True,
                           n_jobs = -1)

rf.fit(xTrain, yTrain)
rfPred=rf.predict(xVal)
R2RF = r2_score(yVal,rfPred)
mseRF = mean_squared_error(yVal, rfPred)

print ('MSE score for RF=',mseRF)
print ('r2 score for RF=',R2RF)

#%% MLP Random Hyperparameter Search 

# Scale data 
sc = StandardScaler()
sc.fit(xTrain)
xTrain = sc.transform(xTrain)
xVal = sc.transform(xVal)
xTest = sc.transform(xTest)

hidden_layer = [(randint(10,100),randint(10,100),), (randint.rvs(5,100),)]
alpha = [0.00005, 0.0001, 0.0002, 0.00005]
learning_rate_init = [0.001, 0.002, 0.005, 0.01, 0.0005]


random_grid_MLP = {'hidden_layer_sizes' : hidden_layer,
               'alpha': alpha,
               'learning_rate_init' : learning_rate_init}

MLP = MLPRegressor(random_state=1,
                   max_iter=300,
                   learning_rate = 'adaptive', #if solver is adam, learning rate is automatically 'constant'
                   activation = 'relu',
                   solver = 'adam',
                   early_stopping = True,
                   tol = 1e-5,
                   n_iter_no_change = 15,
                   verbose = True)

MLP_random = RandomizedSearchCV(estimator = MLP,
                               param_distributions = random_grid_MLP,
                               n_iter = 30,
                               verbose=3,
                               cv =5,
                               random_state=42,
                               n_jobs = -1,
                               scoring = 'neg_mean_squared_error',
                               return_train_score = True)

MLP_random.fit(xTrain, yTrain)
#%% Hyperparameter tuning MLP using forest_minimize

# Scale data 
sc = StandardScaler()
sc.fit(xTrain)
xTrain = sc.transform(xTrain)
xVal = sc.transform(xVal)
xTest = sc.transform(xTest)


reg = MLPRegressor(random_state=1,
                   hidden_layer_sizes=(100,10,10),
                   early_stopping = True,
                   tol = 1e-5,
                   verbose = True)

space  = [Categorical({'tanh','relu'},name='activation'),
          Categorical({'adam','lbfgs'},name='solver'),
          Integer(400, 800, name='max_iter'),
          Integer(10, 35, name='n_iter_no_change'),
          Real(0.0000001,0.001, name='alpha'),
          Real(0.0001,0.01, name='learning_rate_init'),
          Real(0.05, 0.3, name='validation_fraction'),
          Real(0.8, 0.95, name='beta_1'),
          Real(0.92, 0.999, name='beta_2'),
          Real(1e-9,1e-7, name='epsilon')]


@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, xTrain, yTrain, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))
    # create the hidden layers as a tuple with length n_layers and n_neurons per layer
   

res_gp = forest_minimize(objective,
                         space,
                         n_calls=200,
                         n_initial_points=50,
                         random_state=0,
                         verbose=True)

"Best score=%.4f" % res_gp.fun

#%% Multi Layer Perceptron regressor

# Scale data 
sc = StandardScaler()
sc.fit(xTrain)
xTrain = sc.transform(xTrain)
xVal = sc.transform(xVal)
xTest = sc.transform(xTest)

#fit MLP
MLP = MLPRegressor(random_state=1,
                   max_iter=697,
                   hidden_layer_sizes = (100,10,10),
                   alpha = 0.0001878,
                   learning_rate = 'adaptive', #if solver is adam, learning rate is automatically 'constant'
                   activation = 'relu',
                   solver = 'adam',
                   early_stopping = True,
                   learning_rate_init = 0.000769,
                   validation_fraction=0.1547,
                   beta_1=0.8122,
                   beta_2=0.9596,
                   epsilon=9.798e-8,
                   tol = 1e-5,
                   n_iter_no_change = 27,
                   verbose = True)

MLP.fit(xTrain, yTrain)
MLPPred=MLP.predict(xVal)
R2MLP = r2_score(yVal,MLPPred)
mseMLP = mean_squared_error(yVal, MLPPred)
MLPPredTest = MLP.predict(xTest)
np.savetxt("MLP_LMS.csv", MLPPredTest, delimiter=",")
print ('MSE score for MLP=',mseMLP)
print ('r2 score for MLP=',R2MLP)


#%% Gradient boosted tree optimization using gp_minimize
n_features = xTrain.shape[1]
reg = GradientBoostingRegressor(verbose=2, tol=1e-5)

space  = [Integer(1, 35, name='max_depth'),
          Real(10**-5, 1, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(5, 200, name='min_samples_leaf'),
          Integer(400, 1200, name='n_estimators'),
          Real(0.6, 1, name='subsample'),
          Real(0, 3, name='min_impurity_decrease'),
          Integer(2, 300, name='max_leaf_nodes'),
          Real(0.05, 0.3, name='validation_fraction'),
          Integer(10, 30, name='n_iter_no_change'),
          Categorical(categories=["ls", "huber", "lad"], name='loss')]


@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, xTrain, yTrain, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))

res_gp = gp_minimize(objective,
                     space,
                     n_calls=10,
                     n_initial_points=10,
                     random_state=0,
                     verbose=True)

"Best score=%.4f" % res_gp.fun

#%% Gradient boosted tree optimization using forest_minimize
n_features = xTrain.shape[1]
reg = GradientBoostingRegressor(verbose=2, tol=1e-5)

space  = [Integer(1, 35, name='max_depth'),
          Real(10**-5, 1, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(5, 200, name='min_samples_leaf'),
          Integer(400, 1200, name='n_estimators'),
          Real(0.6, 1, name='subsample'),
          Real(0, 3, name='min_impurity_decrease'),
          Integer(2, 300, name='max_leaf_nodes'),
          Real(0.05, 0.3, name='validation_fraction'),
          Integer(10, 30, name='n_iter_no_change')]


@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, xTrain, yTrain, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))

res_gp_rf = forest_minimize(objective,
                     space,
                     n_calls=70,
                     n_initial_points=20,
                     random_state=0,
                     verbose=True,
                     n_jobs=-1)



#%% Gradient Boosting regressor

GBR = GradientBoostingRegressor(n_estimators=1000,
                                random_state=0,
                                max_depth=13,
                                learning_rate=0.0058824,
                                max_features=50,
                                min_samples_split=5,
                                min_samples_leaf=38,
                                subsample=0.6,
                                #min_impurity_decrease=0.1764,
                                max_leaf_nodes=150,
                                #validation_fraction=0.11311,
                                verbose=1,
                                n_iter_no_change=10, 
                                tol=1e-5,
                                loss='huber')
GBR.fit(xTrain,yTrain)

# Get numerical feature importances
importances = list(GBR.feature_importances_)
# List of tuples with variable and importance


GBRPred=GBR.predict(xVal)
R2GBR = r2_score(yVal,GBRPred)
mseGBR = mean_squared_error(yVal, GBRPred)
GBRPredTest = GBR.predict(xTest)
np.savetxt("MLP_LMS.csv", GBRPredTest, delimiter=",",header="sat1_col")
print ('MSE score for MLP=',mseGBR)
print ('r2 score for MLP=',R2GBR)

#%% Only keep features with an importance over a certain treshold
xTrain=xTrain[:,(np.greater(importances,0.002))]
xVal=xVal[:, (np.greater(importances,0.002))]
xTest=xTest[:, (np.greater(importances,0.002))]


