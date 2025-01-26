# **API Documentation**

```python
###################################################################
# University of Toronto
# Faculty of Information
# Master of Information Program
# INF 1340H - Programming for Data Science
# Group Number: 15
# Student Name: Sirui Li, Edward Liu, Hanren Tu
# Student Number: 1004948756, 1002302554, 1011265523
# Instructor: Dr.Maher Elshakankiri
#
#
# Final Project - API Documentation
# Purpose: Comprehensive Data Analysis
# Date Created: 2024-11-10
# Date Modified: 2024-11-21
##################################################################
```

## **Overview**

This documentation provides comprehensive details on the functions used within the final project. Our project aims to help new Airbnb hosts estimate a reasonable nightly rental price for their properties based on listing characteristics such as location, room type, and more. The program defines following functions.

### Table of Contents

1. [Data Cleaning Functions](#data-cleaning-functions)
2. [Descriptive Analytics Functions](#descriptive-analytics-functions)
3. [Diagnostic Analytics Functions](#diagnostic-analytics-functions)
4. [Predictive Analytics Functions](#predictive-analytics-functions)
5. [Utility Functions](#utility-functions)

## **Functions**

### **1. Data Cleaning Functions**

Data cleaning is a crucial step in the data analysis process. The functions in this section are designed to clean and preprocess the dataset, ensuring it is suitable for further analysis and modeling.

#### `read_csv(uri)`

Reads a CSV file from the given URI and returns a pandas DataFrame.

- **Parameters:**

  - `uri` (str): The URI of the CSV file to read.

- **Returns:**
  - DataFrame containing the data from the CSV file.

#### `columns_snakecase(dataFrame)`

Converts column names in a pandas DataFrame to lowercase and replaces all spaces with underscores to ensure uniformity.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose column names need to be converted.

#### `columns_drop(dataFrame, columns)`

Drops the specified columns from a pandas DataFrame, often used to remove irrelevant features.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame from which columns need to be dropped.
  - `columns` (list): A list of column names to be dropped.

#### `columns_drop_by_null_percentage(dataFrame, percentage_threshold)`

Drops columns from a DataFrame where the percentage of null values exceeds a given threshold, reducing noise in the dataset.

- **Parameters:**
  - `dataFrame` (DataFrame): The input DataFrame.
  - `percentage_threshold` (float): The threshold for the percentage of null values beyond which columns will be dropped.

#### `columns_fill_null(dataFrame, columns, value)`

Fills missing values in the specified columns of a pandas DataFrame with a given value, used to handle missing data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame in which missing values need to be filled.
  - `columns` (list): A list of column names whose missing values need to be filled.
  - `value` (any): The value to fill missing values with.

#### `columns_dollarize(dataFrame, columns)`

Converts values in the specified columns of a pandas DataFrame from strings representing monetary values to floats by removing dollar signs and commas.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame in which values need to be converted.
  - `columns` (list): A list of column names whose values need to be converted.

#### `rows_drop_by_condition(dataFrame, condition)`

Drops rows from a pandas DataFrame based on a given condition, typically used for filtering out unwanted data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame from which rows need to be dropped.
  - `condition` (any): A pandas DataFrame condition to filter rows.

#### `rows_drop_by_null(dataFrame, columns)`

Drops rows from a pandas DataFrame that contain missing values in the specified columns, ensuring data completeness.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame from which rows need to be dropped.
  - `columns` (list): A list of column names whose rows need to be dropped.

#### `columns_lowercase(dataFrame, columns)`

Converts the specified columns in a pandas DataFrame to lowercase, typically used for standardizing text data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose columns need to be converted.
  - `columns` (list): A list of column names to convert to lowercase.

#### `columns_categorize(dataFrame, columns)`

Converts the specified columns of a pandas DataFrame to categorical data type, which is useful for optimizing memory usage and improving model performance.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose columns need to be converted.
  - `columns` (list): A list of column names to be converted to categorical data type.

#### `columns_boolize(dataFrame, columns)`

Converts the specified columns of a pandas DataFrame to boolean data type, which is helpful for binary categorical data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose columns need to be converted.
  - `columns` (list): A list of column names to be converted to boolean data type.

#### `columns_intize(dataFrame, columns)`

Converts the specified columns of a pandas DataFrame to integer data type, typically used for numeric categorical data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose columns need to be converted.
  - `columns` (list): A list of column names to be converted to integer data type.

#### `columns_floatize(dataFrame, columns)`

Converts the specified columns of a pandas DataFrame to float data type, useful for numerical data that requires decimal precision.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame whose columns need to be converted.
  - `columns` (list): A list of column names to be converted to float data type.

#### `apply_lambda(dataFrame, columns, fn)`

Applies a lambda function to the specified columns of a pandas DataFrame, enabling custom transformations of data.

- **Parameters:**
  - `dataFrame` (DataFrame): The pandas DataFrame on which the lambda function needs to be applied.
  - `columns` (list): A list of column names whose values need to be transformed.
  - `fn` (any): The lambda function to apply.

### **2. Descriptive Analytics Functions**

Descriptive analytics help summarize the main features of the dataset. The functions in this section are designed to provide insights into the distribution, relationships, and missing values within the data.

#### `calculate_summary_statistics(dataFrame)`

Computes custom summary statistics for all numerical columns in a DataFrame, providing a comprehensive overview of central tendency, dispersion, and shape.

- **Parameters:**

  - `dataFrame` (DataFrame): Input DataFrame.

- **Returns:**
  - DataFrame containing summary statistics for numerical columns, including mean, median, mode, standard deviation, min, max, count, and missing values.

#### `explore_distribution(dataFrame)`

Explores and describes the distribution of numerical data in a DataFrame using skewness and kurtosis.

- **Parameters:**

  - `dataFrame` (DataFrame): The input DataFrame containing numerical data.

- **Returns:**
  - DataFrame containing summary statistics and annotations for the distribution, including skewness (right-skewed, left-skewed, symmetric) and kurtosis (leptokurtic, platykurtic, mesokurtic).

#### `calculate_missing_proportion(dataFrame)`

Identifies and calculates the proportion of missing values for each variable, useful for understanding data quality.

- **Parameters:**

  - `dataFrame` (DataFrame): The input DataFrame.

- **Returns:**
  - DataFrame containing the variable names, missing count, and missing proportion, sorted by the highest proportion of missing values.

#### `generate_frequency_distributions(dataFrame, include_bool=True)`

Generates frequency distributions for all categorical columns in a DataFrame, providing insights into the composition of categorical variables.

- **Parameters:**

  - `dataFrame` (DataFrame): The input DataFrame.
  - `include_bool` (bool): Whether to include boolean columns in the analysis.

- **Returns:**
  - Dictionary containing frequency distributions for each categorical column.

#### `group_price_analysis(dataFrame, group_col)`

Analyzes the relationship between a specified group column and price statistics, visualizing the average price by the group column.

- **Parameters:**

  - `dataFrame` (DataFrame): The input DataFrame containing the specified group column and 'price'.
  - `group_col` (str): The name of the column to group by for analysis.

- **Returns:**
  - DataFrame summarizing average, median, and standard deviation of prices for each group, along with a bar chart visualization.

#### `group_year_price_analysis(dataFrame, group_col)`

Analyzes the relationship between a specified group column and construction year groups in terms of average price. This function segments construction years into 5-year intervals for better analysis.

- **Parameters:**

  - `dataFrame` (DataFrame): The input DataFrame containing columns 'construction_year', the specified group column, and 'price'.
  - `group_col` (str): The column name to group by.

- **Returns:**
  - DataFrame containing average prices for each group and construction year group, with a corresponding visualization.

### **3. Diagnostic Analytics Functions**

Diagnostic analytics helps understand why certain trends and patterns occur in the dataset. The functions in this section are designed to identify relationships and dependencies among variables.

#### `correlation_analysis(data, numerical_columns, method='pearson')`

Performs correlation analysis for a given dataset and visualizes the results using a heatmap. This helps in identifying relationships between numerical features.

- **Parameters:**

  - `data` (DataFrame): The input dataset containing the variables.
  - `numerical_columns` (list): List of numerical column names to include in the correlation analysis.
  - `method` (str): The method to compute the correlation ('pearson', 'spearman', or 'kendall').

- **Returns:**
  - Correlation matrix (DataFrame) displaying the strength of relationships between the numerical features.

#### `cross_tab_analysis(data, group_column, categorical_column)`

Performs cross-tabulation analysis for two specified columns and visualizes the results as a stacked bar chart, providing insights into how categorical variables are distributed across groups.

- **Parameters:**

  - `data` (DataFrame): The input dataset containing the variables.
  - `group_column` (str): The column used for grouping.
  - `categorical_column` (str): The column to count values for within each group.

- **Returns:**
  - Cross-tabulation table (DataFrame) with corresponding stacked bar chart visualization.

#### `regression_analysis(data, independent_vars, dependent_var)`

Performs linear regression analysis to analyze the relationship between independent variables and a dependent variable. Displays regression coefficients, intercept, and summary.

- **Parameters:**

  - `data` (DataFrame): The input dataset containing the variables.
  - `independent_vars` (list): List of independent variable names.
  - `dependent_var` (str): The dependent variable name.

- **Returns:**
  - The fitted linear regression model, including coefficients and intercept.

### **4. Predictive Analytics Functions**

Predictive analytics is used to forecast future outcomes based on historical data. This section provides details regarding the `Dataset` class and the `Model` class, which are used for data preprocessing and predictive modeling respectively.

#### `Dataset`

Defines a wrapper class to preprocess the data for training predictive models.

- **Attributes:**

  - `dataFrame` (DataFrame): The input DataFrame.
  - `target` (str): The name of the target column i.e. value to predict.
  - `features` (list): The list of features to use for prediction. By default, all columns except the target column.
  - `categorical_features` (list): The list of categorical features.
  - `scale_features` (list): The list of features to scale. By default, scale all features columns.
  - `sentiment_features` (list): The list of sentiment features.
  - `labeller_cls` (class): The class to use for label encoding. By default use LabelEncoder class.
  - `scaler_cls` (class): The class to use for scaling. By default use StandardScaler class.
  - `sentiment_analyzer_cls` (class): The class to use for sentiment analysis. By default use SentimentIntensityAnalyzer class.

- **Methods:**
  - `preprocess_train_test(test_size)`: Splits the dataset into the training and testing sets after preprocessing.
    - **Parameters:**
      - `test_size` (float, optional): The proportion of the dataset to include in the test set. Default is 0.2.
    - **Returns:**
      - Tuple of training and testing data (X_train, X_test, y_train, y_test).
  - `preprocess(dataFrame)`: Preprocesses the input DataFrame for prediction.
    - **Parameters:**
      - `dataFrame` (DataFrame): The input DataFrame to preprocess.
    - **Returns:**
      - Preprocessed DataFrame.

#### `Model`

Define a wrapper class for model

- **Attributes:**

  - `id` (str): Name of the model.

- **Methods:**

  - `train(X_train, y_train, **kwargs)`: Trains the model using the given training data.

    - **Parameters:**
      - `X_train` (array-like): Features of the training dataset.
      - `y_train` (array-like): Target values of the training dataset.
      - `**kwargs`: Additional parameters for training the model.

  - `evaluate(X_test, y_test)`: Evaluates the trained model using the test data, providing metrics to assess model performance (e.g. mean squared error, R-squared value)

    - **Parameters:**
      - `X_test` (array-like): Features of the test dataset.
      - `y_test` (array-like): Target values of the test dataset.

  - `predict(X)`: Makes predictions using the trained model.
    - **Parameters:**
      - `X` (array-like): Features to predict target values.
    - **Returns:**
      - Predictions (array-like).

#### `GradientBoostingRegressorModel`, `LinearRegressionModel`, `KNeighborsRegressorModel`, `DecisionTreeRegressorModel`, `NeuralNetworkModel`

These are implementations of the wrapper class for different machine learning models.

### **5. Utility Functions**

Utility functions provide additional tools that are helpful in analyzing and processing data but do not fit into the core data cleaning, analytics, or modeling categories.

#### `get_models()`:

A utility to instantiate all available models defined: Gradient Boosting Regressor, Linear Regression, KNeighbors Regressor, Decision Tree Regressor, and Neural Network.

- **Returns:**
  - A list of available models

#### `train_models(models, dataset)`

A utility to train each model in list.

- **Parameters:**

  - `models` (list): A list of models to train.
  - `dataset` (Dataset): The dataset to use for training.

- **Returns:**
  - A DataFrame of the model evaluations.

#### `inquire(models:, dataset, questions)`

A utility to inquire "questions" per model in list.

- **Parameters:**

  - `models` (list): A list of models to inquire with.
  - `dataset` (Dataset): The dataset used for input preprocessing.
  - `questions` (DataFrame): The questions to inquire.

- **Returns:**
  - A DataFrame of the model predictions.

#### `merge_by_index(left, right)`

A utility to merge the DataFrames, left and right, by index.

- **Parameters:**

  - `left` (DataFrame): The left DataFrame.
  - `right` (DataFrame): The right DataFrame.

- **Returns:**
  - A DataFrame of the merger between left and right.

## **Notes**

- This documentation covers the most relevant functions used for the final project.
- Each function has been designed to facilitate a specific step in the data cleaning, analysis, or modeling workflow, and should be used in combination to achieve optimal results.
- For more detailed explanations of each step and additional functions, refer to the code comments and in-line documentation.
