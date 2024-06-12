# Introduction to Supervised Machine Learning

Supervised machine learning is a type of machine learning where the model is trained on a labeled dataset. This means that each training example is paired with an output label. The goal is to learn a mapping from inputs to outputs that can be used to predict labels for new, unseen data.

## Key Concepts

### Features and Labels

- **Features**: The input variables (also called attributes) used to make predictions.
- **Labels**: The output variable that the model aims to predict.

### Training and Testing

- **Training**: The process of using a dataset to teach a model to make predictions. During training, the model learns the mapping from features to labels.
- **Testing**: Evaluating the performance of the trained model on a separate dataset that the model has not seen during training. This helps to assess the model's generalization ability.

### Model

A mathematical representation of a real-world process. In supervised learning, it maps inputs to outputs based on the training data.

### Loss Function

A function that measures how well the model's predictions match the actual labels. The goal is to minimize this function. Different algorithms use different loss functions.

## Types of Supervised Learning Problems

### Classification

Predicting a discrete label (e.g., spam or not spam). Examples include email spam detection and image recognition.

### Regression

Predicting a continuous value (e.g., house prices). Examples include stock price prediction and temperature forecasting.

## Common Algorithms in Supervised Learning

### Linear Regression

Used for regression tasks. Models the relationship between the dependent variable and one or more independent variables using a linear equation.

#### Mathematical Formula

The linear regression model is represented as:

\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n \]

where:

- \( y \) is the predicted value.
- \( x_1, x_2, \ldots, x_n \) are the input features.
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \ldots, \beta_n \) are the coefficients.

#### Loss Function

The most common loss function for linear regression is the Mean Squared Error (MSE):

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

where:

- \( n \) is the number of data points.
- \( y_i \) is the actual value.
- \( \hat{y}_i \) is the predicted value.

### Logistic Regression

Used for classification tasks. Estimates the probability that a given input belongs to a certain class.

#### Mathematical Formula

The logistic regression model is represented as:

\[ P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)}} \]

where:

- \( P(y=1|x) \) is the probability that the output \( y \) is 1 given input \( x \).
- \( \beta_0, \beta_1, \beta_2, \ldots, \beta_n \) are the coefficients.

#### Loss Function

The most common loss function for logistic regression is the Binary Cross-Entropy Loss (Log Loss):

\[ \text{Log Loss} = - \frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] \]

where:

- \( n \) is the number of data points.
- \( y_i \) is the actual label.
- \( \hat{y}_i \) is the predicted probability.

### Decision Trees

Can be used for both classification and regression. Splits the data into subsets based on the value of input features.

#### Mathematical Formula

A decision tree splits data points recursively based on feature values. At each node, the feature and threshold that result in the highest information gain or lowest impurity are chosen.

### Support Vector Machines (SVM)

Can be used for both classification and regression. Finds the hyperplane that best separates the classes.

#### Mathematical Formula

The decision function for a linear SVM is:

\[ f(x) = w \cdot x + b \]

where:

- \( w \) is the weight vector.
- \( x \) is the input vector.
- \( b \) is the bias.

The optimization objective is:

\[ \min_w \frac{1}{2} ||w||^2 \]

subject to:

\[ y_i (w \cdot x_i + b) \geq 1 \]

for all training examples \( (x_i, y_i) \).

### k-Nearest Neighbors (k-NN)

Used for both classification and regression. Predicts the label of a new input based on the labels of the k nearest training examples.

#### Mathematical Formula

The prediction for k-NN is based on the majority vote for classification or the average for regression of the k nearest neighbors.

### Random Forest

An ensemble method that uses multiple decision trees. Improves the model's accuracy and reduces overfitting.

#### Mathematical Formula

A random forest consists of multiple decision trees. The prediction is made by averaging the predictions (regression) or taking the majority vote (classification) of all the individual trees.

### Neural Networks

Can be used for both classification and regression. Composed of layers of interconnected nodes (neurons).

#### Mathematical Formula

Each neuron in a neural network performs a weighted sum of its inputs and applies a non-linear activation function:

\[ y = f\left( \sum_{i=1}^{n} w_i x_i + b \right) \]

where:

- \( f \) is the activation function (e.g., ReLU, Sigmoid).
- \( w_i \) are the weights.
- \( x_i \) are the inputs.
- \( b \) is the bias.

## Example: Linear Regression

Linear regression is one of the simplest algorithms used for regression tasks. Let's see how it works with an example and code.

### Problem Statement

Predict the price of a house based on its size.

### Dataset

We'll use a simple dataset with two columns: `Size` (in square feet) and `Price` (in dollars).

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
# Size (square feet)
X = np.array([[1500], [1600], [1700], [1800], [1900], [2000], [2100], [2200], [2300], [2400]])
# Price (dollars)
y = np.array([300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Size (square feet)')
plt.ylabel('Price (dollars)')
plt.title('Linear Regression: House Price Prediction')
plt.show()

# Print model parameters
print('Coefficient:', model.coef_)
print('Intercept:', model.intercept_)

# Calculate and print the model's accuracy
accuracy = model.score(X_test, y_test)
print('Model Accuracy:', accuracy)
