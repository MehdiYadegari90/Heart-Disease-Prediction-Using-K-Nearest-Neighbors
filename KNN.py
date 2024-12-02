# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

####################################################################
# Load the dataset from a CSV file
df = pd.read_csv('heart.csv')

# Display the counts of the target variable 'output'
#print(df["output"].value_counts())

# Visualize the distribution of the 'chol' column
#df.hist(column="chol", bins=50)

# Display the columns in the DataFrame
#print(df.columns)

#####################################################################
# Prepare the feature matrix (X) and target vector (y)
# Select relevant features for the model
x = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
         'exng', 'oldpeak', 'slp', 'caa', 'thall']].values

# Target variable
y = df['output'].values

# Normalize the feature matrix
X = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

#####################################################################
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Display the shapes of the training and testing sets
#print("Train set shapes: X_train =", X_train.shape, ", y_train =", y_train.shape)
#print("Test set shapes: X_test =", X_test.shape, ", y_test =", y_test.shape)

#####################################################################
# Define the KNN classifier and fit it to the training data
k = 4  # Number of neighbors
neig = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Make predictions on the test set
y_hat = neig.predict(X_test)

# Display the predicted and actual values of the first 5 instances
#print("Actual values:", y_test[0:5])
#print("Predicted values:", y_hat[0:5])

#####################################################################
# Evaluate the model's performance
# Print the accuracy on the training set
#print("Train set Accuracy =", metrics.accuracy_score(y_train, neig.predict(X_train)))

# Print the accuracy on the test set
print("Test set Accuracy =", metrics.accuracy_score(y_test, y_hat))

#####################################################################
