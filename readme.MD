# Fighting fire with firepower
## Using machine-learning models to classify wildfires by intensity
<br>
<br>
<strong>General Assembly Data Science Intensive 7:</strong><br>

> Wayne Chan <br>
> Mariam Javed <br>
> Shawn Syms 

<a name="problem-statement"></a>
## Problem statement

Wildfires have a devastating impact on people, wildlife and the environment. At risk are homes, infrastructure and  wildlife habitat, as well as an increase in toxic emissions. In 2019, wildfires in Alaska and California in the summer months accounted for 4.5 billion dollars worth of damage according to the National Centers for Environmental Information. The related costs to fight these fires are also increasing. The US Forest Service and other agencies spent an all-time high of \$2.9B fighting fires in 2018, based on data from the National Interagency Fire Center.

It is therefore crucial to develop a **classification model** that can predict the size of fires so that appropriate resources can be gathered to combat them before they can cause too much damage. We will describe how the size of the fire changes depending on a range of factors. Our guiding accuracy metric will be the accuracy score, a statistical measure of how close the data is to the predictions. 

Through implementing a panoply of machine learning models, we will determine the best possible estimator, and use it to predict fire size to a significant degree of accuracy. As Smokey the Bear, said "Only you can prevent forest fires," but once they are started, our model will be able to predict how intensely they could spread if left unchecked, allowing life-saving resources to be put into play at the proper scale, as quickly as possible.



## Project summary

We initially approached this project as a regression problem, where our target variable was the "fire_size." 

To that end, we enriched the data with historical weather data for each fire occurrence in 2015 from the DarkSky API and developed series of 12 regression models: 


> - ADAboost
> - Bagging regressor
> - ElasticNet
> - Extra Trees regressor
> - Gradient Boosting regressor 
> - LassoCV
> - linearSVR  
> - logistic regression
> - Random Forest regressor
> - RidgeCV
> - Stochastic Gradient Descent (SGR) regressor
> - Support Vector regressor (SVR)


However, our r2 values for these models were almost negligible, and the train and test scores were less than 5 percent.

So we therefore decided to change our approach, and refocus the project as a classification problem. As the target variable, we selected "fire_size_class", where:

 - <strong>A:</strong> greater than 0 but less than or equal to 0.25 acres
 - <strong>B:</strong> 0.26-9.9 acres
 - <strong>C:</strong> 10.0-99.9 acres
 - <strong>D:</strong> 100-299 acres 
 - <strong>E:</strong> 300 to 999 acres
 - <strong>F:</strong> 1000 to 4999 acres 
 - <strong>G:</strong> 5000+ acres

For this new objective, we used the entire dataset (1.8M fire occurrences) and ran the following models:

> - ADAboost
> - Feed-forward Neural Network (NN)
> - kNN
> - Random Forest classifier

As a result of our shift in approach, our accuracy scores improved significantly. The best model (NN) output 61 percent scores for both the training and the test set.



## Project notebooks

We divided our coding efforts into five notebooks:

Part 1: Project overview, initial imports, exploratory data, models: ADAboost, kNN and Random Forest Classifier 

Part 2: Initial feed-forward neural network

Part 3: Feed-forward neural network with hyperparameters tuned

Part 4: Classification metrics

Part 5: API and webscraping



## Conclusion

- Neural Network was the most effective model at predicting the fire-size class for a wildfire—with train and test accuracy of 61 percent and 62 percent, respectively
- This dataset is far better suited to a classification problem than a regression one, with a **significant** boost in model results—from < 0.05 r2 scores to approximately ~60 percent accuracy scores
- Our results are comparable to others that have used this dataset (see Sky B.T. Williams's [article](https://towardsdatascience.com/wildfire-destruction-a-random-forest-classification-of-forest-fires-e08070230276)), which tells us that our accuracy results are the best you can achieve when using the dataset as is



## Recommendations

- The historical weather data from DarkSky API was only collected for 2015 given time and financial constraints. We recommend that this observed weather data is collected for all years (1992—2015) and implemented into the NN classification model to see how results improve
- It's evident from the results that there are significant class-distribution imbalances in this dataset, as class A and class B make up the majority of the dataset, while class G makes up only 0.2% of the total wildfires. For this reason, it would be beneficial to implement some sort of weight classes to compensate for the imbalance in data points



