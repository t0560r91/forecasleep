# Ultimate Sleep Guide  


## Introduction  
As much as an ambitious data scientist wants to discover a new insight to better the world during the day, the data scientist needs a full recovery of mind and body over the night in order to access the full focus and endurance given for a day. Our body brutally penalizes us for not giving it enough rest, especially when we need the most out of it. So we make a deal with our body that we will give it the 8 hours of sleep, as recommended by doctors, in return of fully recovered mind and body the next day. However, the body still throws you a tantrum by saying “No! that was not the right amount of rest I requested!”. As it turns out, the 8 hours of awful sleep from yesterday is not the same as the 8 hours of super restful sleep from last week. When the sleep quality is not tangible, how can we know how much to sleep in order to fully recover our body? 

## Objective  
The objective is to help users knowingly decide how much to sleep by constructing a prediction model that user can interact over web application to estimate the sleep quality for each night. Sleep quality can be measured by “minutes of deep sleep”

## Method  
The product builds a model at each use utilizing Lasso Regression (regularization strength = 1.28) to predict “minutes of deep sleep” based on each user’s past sleep data collected through a wearable tracker device. New model is fit at each day upon predict-activation in order to include up-to-date data. Evaluated model performance with cross validated R-Square. 

## Product Description  
Developed an web application where user can authorize the server to fetch sleep data directly from Fitbit Web API and user can input anticipated sleep start and end time to receive back a calculated sleep quality estimation. Currently developing for a more in-depth sleep analysis to be presented to the user which includes a list of most influencing predictors.

## Discussion  
Prediction accuracy can be significantly improved by having a larger dataset and wearing tracker device 24/7 without skipping a day or night, and also including caffeine and alcohol intake data.

Tools Used: Python | Sci-kit Learn | Flask | Brython | HTML | Fitbit Tracker Device | Fitbit Web API 

