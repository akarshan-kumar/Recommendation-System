# Product Recommendation System

## Content Based (Amazon Apparel Recommendation Sytem)

### Data Overview

### Sample Data Points

| Product ID | ASIN        | Brand           | Color        | Medium Image URL                                                                 | Product Type | Title                                     | Price  |
|------------|-------------|-----------------|--------------|----------------------------------------------------------------------------------|--------------|-------------------------------------------|--------|
| 4          | B004GSI2OS  | FeatherLite     | Onyx Black/Stone | ![Image](https://images-na.ssl-images-amazon.com/images/I/41fE2VV4IQL._SX38_SY50_CR,0,0,38,50_.jpg) | SHIRT        | Featherlite Ladies' Long Sleeve Stain Resistant... | $26.26 |
| 6          | B012YX2ZPI  | HX-Kingdom      | White         | ![Image](https://images-na.ssl-images-amazon.com/images/I/41Fj-T0TxyL._SX38_SY50_CR,0,0,38,50_.jpg) | SHIRT        | HX-Kingdom Fashion T-shirts White         |         |
| 11         | B001LOUGE4  | Fitness Etc.    | Black         | ![Image](https://images-na.ssl-images-amazon.com/images/I/41yqTliIVDL._SX38_SY50_CR,0,0,38,50_.jpg) | SHIRT        | Fitness Etc. Black                        |         |
| 15         | B003BSRPB0  | FeatherLite     | White         | ![Image](https://images-na.ssl-images-amazon.com/images/I/41jI-5sMNOL._SX38_SY50_CR,0,0,38,50_.jpg) | SHIRT        | FeatherLite White                         |         |
| 21         | B014ICEDNA  | FNC7C           | Purple        | ![Image](https://images-na.ssl-images-amazon.com/images/I/41f9-T2jMrL._SX38_SY50_CR,0,0,38,50_.jpg) | SHIRT        | FNC7C Purple                              |         |

## Conclusion Table

| Weight Scheme                             | Remarks                                                                                                                                   |
|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Equal weights                             | This scheme of weights does not produce satisfactory results as CNN features being the longest vector overshadows others and recommendations suffer. VGG16 may be a good edge and pattern detector but it did not take care of color and/or semantics.       |
| More weight to IDF featurization          | Mostly recommendations from similar brands but relevant recommendations.                                                                 |
| More weights to Brand+Type+Colour vectorization | Mostly relevant recommendations but no recommendations from other brands.                                                                  |
| More weight to CNN featurization          | No satisfactory results. VGG16 may be a good edge and pattern detector but it did not take care of color and/or semantics.                 |
| Proportional weights                      | Recommendations similar from same brands but irrelevant recommendations from different brands.                                            |
| More weightage to Color                   | Recommendations are pretty relevant in nature.                                                                                           |


## Graph data recommendations (Facebook Freind Recommendation)
# Facebook Friend Recommendation System

## Problem Overview

The Facebook Friend Recommendation System aims to predict potential friendships within a given group of people based on their follower-followee relationships. The data provided includes a 2-column CSV file indicating directed connections among approximately 1.86 million users and 9.4 million connections.

This is essentially a **graph problem**. We employ various graph mining techniques to generate features that are then used to train machine learning models.

## Workflow

1. **Basic Exploratory Data Analysis (EDA):**
    - 99% of users have 40 followers/followees.
    - 14% of people don't follow anyone.
    - 10% of people don't have followers.
    - Intersection of the above two groups is null.
    - Minimum number of followers + followees: 1.
    - 334,291 people have the minimum number of followers + followees.
    - Maximum number of followers + followees: 1,579.
    - Only 1 person has the maximum number of followers + followees.

2. **Creating Negative Samples:**
    - As connections that don't exist are not provided, we generate them to balance the dataset.
    - Negative samples are generated based on the domain knowledge that the probability of a connection with a shortest path length greater than 2 is unlikely.

3. **Feature Engineering:**
    - **NetworkX Python Package**: Used for graph mining functions.
    - **Features Generated:**
        - Jaccard Distance: Ratio of common followers/followees to total followers/followees.
        - Cosine Distance for followers/followees.
        - Page Rank: Ranking users based on the PageRank algorithm.
        - Shortest Path Length: Distance between nodes.
        - Weak Connections, Adar Index, Follow Back, Katz Centrality, Hub/Authority scores.
        - Number of followers/followees.
        - Inter-followers/followees and weighted edges of source and destination.
        - Linear combinations of the above features.
        - Matrix Factorization using SVD: Top six components and SVD dot feature for source and destination.
        - Preferential Matching: Based on the "rich getting richer" theory.

4. **Modeling:**
    - Data split into train and test sets without information leakage.
    - **Models Used:**
        - Random Forest
        - Logistic Regression
    - Hyperparameter tuning was performed for both models.

5. **Evaluation:**
    - Models evaluated using F1 score as both precision and recall are important.
    - Latency was not a major concern.

6. **Feature Importance:**
    - Shown using a bar graph for both models.

## Results

| S.NO. | Model               | Train Score | Test Score |
|-------|---------------------|-------------|------------|
| 1     | Random Forest       | 96.52%      | 92.62%     |
| 2     | Logistic Regression | 88.61%      | 69.25%     |

The Random Forest model performed better than the Logistic Regression model, aligning with the intuition that a more complex model can capture the nuances of the data better.

## Collaborative (Netfilx movie recommendstion)
# Netflix Movie Rating Prediction

## Problem Overview

The objective of this project is to predict user ratings for movies based on historical rating data. The data is structured as follows:
- **CustomerID**: Unique identifier for each customer.
- **MovieID**: Unique identifier for each movie, ranging from 1 to 17770.
- **Rating**: User ratings on a scale of 1 to 5.
- **Date**: The date on which the rating was given, formatted as YYYY-MM-DD.

There are 480,189 users, and the goal is to minimize the Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE) for the predictions.

## Machine Learning Objective and Constraints

1. **Minimize RMSE**: Focus on reducing the Root Mean Square Error for more accurate predictions.
2. **Interpretability**: Aim to provide some level of interpretability in the model's predictions.

## Exploratory Data Analysis (EDA)

Initial EDA involved:
- Loading the data into a dataframe and examining statistical properties.
- Creating a new feature, `weekday`, which did not yield meaningful insights.
- Analyzing the severity of the cold start problem, which was more pronounced in the training data.

## Feature Engineering

Several techniques were attempted to enhance feature engineering:
1. **Direct Matrix Manipulation**: Direct manipulation of the data matrix proved to be time-complex due to the dense nature of the matrix.
2. **Singular Value Decomposition (SVD)**: Tried SVD for dimensionality reduction but found it too time-consuming for this dense matrix.
3. **Similar Users and Movies**: Created features for top five similar users and movies.
4. **Surprise Library**: Utilized various models from the Surprise library including:
    - Baseline Predictor
    - KNN Baseline Model
    - Simple SVD
    - Implicit SVD++

## Modeling

Different models were trained and evaluated using RMSE:
- **Baseline Predictor**
- **KNN Baseline Model**
- **SVD**
- **SVD++**
- **XGBoost Regressor**: Trained with and without additional features.

## Results

The results are summarized in the table below:

| S.NO. | MODEL                        | RMSE   |
|-------|------------------------------|--------|
| 1     | Baseline Predictor (bslpr)   | 1.7140 |
| 2     | KNN Baseline User (knn_bsl_u)| 1.0710 |
| 3     | KNN Baseline Movie (knn_bsl_m)| 1.0710 |
| 4     | SVD                          | 1.0712 |
| 5     | SVD++                        | 1.07135|
| 6     | XGBoost + 13 Features        | 1.0947 |
| 7     | XGBoost + 13 Features + bslpr| 1.1099 |
| 8     | XGBoost + 13 Features + bslpr + knn | 1.0766 |
| 9     | XGBoost + bslpr + knn + mf   | 1.0755 |
| 10    | XGBoost + 13 Features + bslpr + knn + mf | 1.0725 |

## Conclusion

- **KNN Baseline Models** and **SVD** techniques performed the best, achieving an RMSE around 1.071.
- **XGBoost** with a combination of features showed slightly higher RMSE but provides a more interpretable model.

This project demonstrates the effectiveness of combining collaborative filtering techniques with machine learning models to predict movie ratings accurately.

