# Regression Repository

<img src="https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/201003/20100315144150-1_0.jpg?itok=xksoTT8q">

## Goals

1. **Acquisition-Gather:** Gather structured data from SQL to Pandas

2. **Acquisition-Summarize:** Summarize the data through aggregates, descriptive stats and distribution plots (histograms, density plots, boxplots, e.g.). (pandas: value_counts, head, shape, describe, info, matplotlib.pyplot.hist, seaborn.boxplot)

3. **Preparation-Clean:** We will convert datatypes and handle missing values. In this module we will keep it simple in how we handle missing values. We will introduce other ways to handle missing values as we progress through the course. (pandas: isnull, value_counts, dropna, replace)

4. **Preparation-Split:** We will sample the data so that we are only using part of our available data to analyze and model. We will discuss the reasons for doing this. This is known as "Train, Validate, Test Splitting". (sklearn.model_selection.train_test_split).

5. **Preparation-Scale:** We will discuss the importance of "scaling" data, i.e. putting variables of different units onto the same scale. We will scale data of different units to be on the same scale so they can be compared and modeled. We will discuss different methods for scaling data and why to use one type over another. (sklearn.preprocessing: StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler)

6. **Exploration-Hyptothesize:** We will discuss the meaning of "drivers", variables vs. features, and the target variable. We will disucss the importance of documenting your questions and hypotheses, how to answer them, and documenting takeaways and answers at each step of exploration.

7. **Exploration-Visualize:** We will use visualization techniques (scatterplots, jointplot, pairgrid, heatmap) to identify drivers. When a visualization needs to be followed up with a test, we will do so.

8. **Exploration-Test:** We will analyze the drivers of a continuous variable using appropriate statistical tests (t-tests and correlation tests).

9. **Modeling-Feature Engineering:** We will learn ways to identify, select and create features through feature engineering methods, specifically feature importance. We will discuss the "Curse of Dimensionality." (sklearn.feature_selection.f_regression).

10. **Modeling-Establish Baseline:** We will learn about the importance of a "baseline model" and ways to establish that.

11. **Modeling-Build Models:** We will build linear regression models, i.e. we will use well established algorithms, such as glm (generalized linear model) or a basic linear regression algorithm (e.g. y = mx + b), to extract the patterns the data is demonstrating and return to us a mathematical model or function (e.g. y = 3x + 2) that will predict the target variable, or outcome) we want to predict. We will learn about the differences in the most common regression algorithms. (sklearn.linear_model)

12. **Modeling-Model Evaluation:** We will compare regression models by computing "evaluation metrics", i.e. metrics that measure how well a model did at predicting the target variable. (statsmodels.formula.api.ols, sklearn.metrics, math.sqrt)

13. **Modeling-Model Selection and Testing:** We will learn how to select a model and we will test the model on the unseen data sample (the out-of-sample data in the validate and then test datasets).

14. **Data Science Pipeline and Product Delivery:** We will end with an end to end project practicing steps of the pipeline from planning through model selection and presentation.

## Types of Regression

### Simple Linear Regression

y = b0+b1x+ϵ

In the simple linear case, our feature is x and our target is y. The algorithm finds the parameters that minimize the error between the actual values and the estimated values. The parameters the algorithm will estimate are the slope, β, and the y-intercept, α. ϵ
is the error term, or the residual value. The residual is the difference of the actual value from the predicted value.

### Multiple LInear Regression

y = b0+b1x1+b2x2+...bnxn+ϵ

In a multiple linear regression case with n features, our features are x1 through xn and our target is y. The algorithm finds the parameters that minimize the error between the actual values and the estimated values. The parameters the algorithm will estimate are the coefficients of the features, b1 through bn, and the y-intercept, b0. ϵ is the error term, or the residual value.

### Polynomial Regression

y = b0+b1x+b2x2+...+bnxn+ϵ

In the case we have a polynomial function (), we still have a linear model due to the fact that xi is in fact a feature, and the coefficients/weights associated with that feature is still linear. To convert the original features into their higher order terms we will use the PolynomialFeatures class provided by scikit-learn. Then, we train the model using Linear Regression.