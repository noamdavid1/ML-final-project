# ML-final-project
Fake profiles are a major problem in the media platform and in particular in the Instagram social network. Using machine learning it is possible to distinguish between fake pages and real pages.
## The database we worked with:
Our database is 2 CSV files - one for the training group and the other as the test group.
The files contain data for a machine learning model. The data consists of various characteristics that describe profiles of users on the Instagram social network so that each line represents a unique profile with its characteristic values.

## How did we do it? 
In order to classify we used several classification algorithms such as:

*SVM:*
An algorithm for classification problems, the main point of which is to find the optimal route between the most different data groups, this is done by creating a "separation plane" that separates the data points in the best way with the help of support vectors.

*AdaBoost:*
Works by combining multiple "weak" classifiers into one "strong" classifier. Each weak classifier is trained on a subset of the data, and the algorithm assigns higher weights to the misclassified data points. The next weakest classifier is then trained on the updated data set, with the weights reflecting the difficulty of classifying each data point.

*KNN:*
It classifies new data points based on the majority class of its K nearest neighbors in the feature space.

*Logistic Regression:*
A classification algorithm whose purpose is to predict the likelihood of a specific outcome using a logistic function. Generally, it is a classification of data into two groups or categories. (fake or real).

## Results
We made several comparisons between these models.
Here are the results:
![WhatsApp Image 2024-03-17 at 15 39 21](https://github.com/noamdavid1/ML-final-project/assets/93923600/72768a62-fd24-48b9-84db-ef3590185268)

### collaborators:                                                                                     
Noam David.                                                                                             
Yogev Ofir.
