# Predoct if a student will pass based on hours of study
# Dataset with two columns: Hours (number of hours studied) and Pass (1 if the student passes, 0 if the student fails).

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
# Hours studied
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# Pass (1 if pass, 0 if fail)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2, marker='o')
plt.xlabel('Hours Studied')
plt.ylabel('Pass/Fail')
plt.title('Logistic Regression: Pass/Fail Prediction')
plt.show()

# Print model parameters
print('Coefficient:', model.coef_)
print('Intercept:', model.intercept_)

# Calculate and print the model's accuracy
accuracy = model.score(X_test, y_test)
print('Model Accuracy:', accuracy)
