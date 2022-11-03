# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A model is A binary prediction model that predicts income exceeds $50K/yr or not. The model is trained using [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
## Intended Use
The model will be used as baseline to compare other income prediction models.
## Training Data
[The UCI Machine Learning Repository data set](https://archive.ics.uci.edu/ml/datasets/census+income) is used as the training data. The data is originally from the 1994 Census database.

## Evaluation Data
20% of the training data is assigned as the validation data.
## Metrics
The following metrics are used to evaluate the models predictions:

- precision: 0.7285 
- recall: 0.2699 
- fbeta: 0.3939 

Furthermore, sliced metrics based on sex are calculated. 

### Male
- precision : 0.7731
- recall : 0.2750
- fbeta : 0.4057

### Female
- precision : 0.5283
- recall : 0.2403
- fbeta : 0.3304

## Ethical Considerations
Today, Women's empowerment has been improved dramatically compared to that in 1994, when data was originally created.Therefore predicting today's women's salaries using this model might cause wrong implications.
## Caveats and Recommendations
As the sliced metrics above show, precision score differs significantly among gender. 