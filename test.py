import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the numerical representation of the affidavit dataset
affidavit_dataset = pd.read_csv('affidavit_dataset.csv')
affidavit_dataset_vectorized = np.load('affidavit_dataset_vectorized.npy',allow_pickle=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(affidavit_dataset_vectorized, affidavit_dataset['counter_affidavit'], test_size=0.2)

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
