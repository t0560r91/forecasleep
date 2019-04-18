## Ultimate Sleep Guide (in progress... )



Prediction model for sleep quality based on features collected on activity/sleep tracker device



Being able to accurately estimate sleep quality each night prior to getting ready for bed and to know how each factor is contributing to the estimation, users can adjust their day time activities, food intake, and overall sleep pattern to optimize long term sleep quality which leads to a better quality day and overall life satisfaction.



More detailed project proposal at: 
https://docs.google.com/document/d/1pLIrq1HPVRkT1hOauKMX99xBhgSEB9NsbXeBk-QTJjU/edit?usp=sharing

Interactive Web Application URL: 


terms of service:
privacy policy:



Ultimate Sleep Guide                                                          


Introduction
As much as an ambitious data scientist wants to discover a new insight to better the world during the day, the data scientist needs a full recovery of mind and body over the night in order to access the full focus and endurance given for a day. Our body brutally penalizes us for not giving it enough rest, especially when we need the most out of it. So we make a deal with our body that we will give it the 8 hours of sleep, as recommended by doctors, in return of fully recovered mind and body the next day. However, the body still throws you a tantrum by saying “No! that was not the right amount of rest I requested!”. As it turns out, the 8 hours of awful sleep from yesterday is not the same as the 8 hours of super restful sleep from last week. When the sleep quality is not tangible, how can we know how much to sleep in order to fully recover our body? 

Objective
The objective is to help users knowingly decide how much to sleep by constructing a prediction model that user can interact over web application to estimate the sleep quality for each night. Sleep quality can be measured by “minutes of deep sleep”

Method
The product builds a model at each use utilizing Lasso Regression (regularization strength = 1.28) to predict “minutes of deep sleep” based on each user’s past sleep data collected through a wearable tracker device. New model is fit at each day upon predict-activation in order to include up-to-date data. Evaluated model performance with cross validated R-Square. 

Model Selection & Parameter Tuning
CV R-Squared
Ridge Regression
Lasso Regression
user_id
data size
reg. strength(a)
train
test
reg. strength(a)
train
test
0
60
30.13
0.34
-0.13
4.34
0.28
0.02
1
409
120.27
0.32
0.23
1.21
0.28
0.27
2
435
44.39
0.50
0.42
1.28
0.47
0.45

Product Description
Developed an web application where user can authorize the server to fetch sleep data directly from Fitbit Web API and user can input anticipated sleep start and end time to receive back a calculated sleep quality estimation. Currently developing for a more in-depth sleep analysis to be presented to the user which includes a list of most influencing predictors.

Discussion
Prediction accuracy can be significantly improved by having a larger dataset and wearing tracker device 24/7 without skipping a day or night, and also including caffeine and alcohol intake data.


Tools Used: 
Python | Sci-kit Learn | Flask | Brython | HTML | Fitbit Tracker Device | Fitbit Web API | 
Github:
https://www.github.com/sehokim88/ultimate_sleep_guide

