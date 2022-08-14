# Flight Price Prediction System

##**INTRODUCTION**

The airline industry has evolved into being a highly unpredictable industry with its complex ticket pricing systems. Nowadays, the variation in ticket prices has led to customers traveling with different fares even after being on the same flight. The practice of early booking of tickets by a customer does not seem to be of much importance as in some cases it has been seen that an early purchase of a flight ticket costs the customer more than the booking done just a few days prior to the date of journey.

The fluctuation in the price of a flight ticket is due to the high variability in the factors on which the prices are dependent which include the date of journey, source and destination airports, routes taken, flight duration, arrival and departure times, number of stops, and additional information such as ticket class, in-flight meals, layovers, etc. The other reasons for this fluctuation in prices are insufficient datasets, competition amongst different airline brands, proprietorship in the pricing policies of airlines, etc.


##**PROBLEM STATEMENT**

The variability in the flight fares leads to a number of problems like customer dissatisfaction, flights operating under booked, revenue losses to the airline, budgeting issues for the customers, etc. Amidst this variability of prices, the customer always seeks to avail of the lowest possible fare for their journey. On the other hand, the airline is driven by the sole motive of attaining maximum profit from a trip. This leads to the requirement of a system that can estimate the flight fares with good accuracy prior to the date of the journey of a customer.


##**OBJECTIVE**

The prediction of flight prices at an early stage would help airlines develop suitable strategies and gather the necessary resources impacting a specific market segment level for a route. An efficient system for predicting the flight fare prices should be able to capture the variations in customer demand based on internal factors like off-season demand, seat availability, holiday period, etc, and external factors like emergencies, natural disasters, weather changes, and mass gatherings.

This project aims to develop to propose a model which can capture the variability between different factors affecting the cost of a flight ticket and arrive at the best possible model for estimating the prices of the trip. 

##**RESULTS**

In this study, a machine learning framework was developed to predict the airfare ticket prices between March to June 2019. The dataset used comprises 10683 data points/samples and 10 features with the flight prices being the target variable. Several features were extracted from the datasets and combined together with macroeconomic data, to model the air travel market segment. With the help of the feature selection techniques, the model comprising of the ensemble of the Artificial Neural Networks, XGBoost Regressor, and Light Gradient Boosted Machine Regression models gave the least mean square error and mean absolute error besides having a high R2 score. It is able to predict the average airfare price with a mean absolute error of 1301.4268.


