# Team T:  
* Luca Rossi SCIPER 331192,
* Lucas Maximilian Schlenger SCIPER 331031
* Jonas Lyng-Jorgensen SCIPER 334127
Id best submission: 142031

# Packages needed:
1. numpy
2. matplotlib
3. pandas
4. sklearn
5. skopt
6. scipy
7. xgboost

# Additional files needed:
test_set.csv, train_set.csv, val_set.csv

# Main file: 
LMS_ML.py

# Remark:
Make sure the directory you are using contains the files: test_set.csv, train_set.csv and val_set.csv.

# Code explanation:

The first cell that is marked with "NEEDS TO RUN" is only the loading of different libraries. The second marked cell is the data preparation, here we start with converting the dataframes into np.arrays. Shortly after we get obtain the period of each observation,
at which the maximum spectral acceleration occurs. Then we add the Period of the max. SA and drop "da5_75" from the datasets, as it increased performance when only using "da5_75" or "da5_95" and result have overall been better with "da5_95". After that happens the magic :D 
As we concluded from looking at various plots of the spectral accelerations, that the sat1_col is more dependend on the shape of the plot, rather than the absolute values, we searched for different ways on how to scale the observations individually, in order to make them
more comparable. Since we wanted to scale each observation by an average of a number of their maximum values, we needed to create equidistant points, in order to make the scaling indepentend from the position of the maximum, e.g. if the maximum of the curve would occur at a period of 0.02s,
than it would be very likely that the values 0.01 and 0.022 would also be very high in respect to the rest of the values of that given observation, if the maximum would occure at a period of 8.0 seconds, the likelyhood that the values of 7.5s and 8.5s seconds are also very high is 
lower than in the first example. For this reason we took the data and created "K" equidistant points. After that we scaled each spectral acceleration with the average of the "I" highest values of the given observation.

In the last cell that needs to be run (3 of 3) we use a XGBRegressor to estimate the sat1_col of the test set. The hyperparameters were obtained using the forest_minimize function of the skopt toolbox, but it takes (depending on the number of calls and your hardware) a couple of hours. We choose the XGB, 
because we saw that the random forest was able to produce quite robust predictions which were then improved by using a gradient boosting random forest. 

The final idea with the scaling came just a couple of days ago, and we didn't have enough time to further look into it, we do not believe that it is the best approach to set "K" and "I" to the same number, however, we obtained the best results with both of them being 80. 

# How to run the code
I personally used Spyder 3.8.

All the cells that need to be run in order to obtain the result are marked with a "NEEDS TO RUN" in the beginning, followed by an indication, in what order they should be run. 