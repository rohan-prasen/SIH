import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('affidavit_dataset.csv')

# Preprocess the data
X = df['filed_affidavit']
y = df['counter_affidavit']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create a support vector machine classifier
clf = SVC()

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Save the model
clf.save('counter_affidavit_model.pkl')
