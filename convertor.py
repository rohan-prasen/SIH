import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the affidavit dataset
affidavit_dataset = pd.read_csv('affidavit_dataset.csv')

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Transform the text data in the affidavit dataset into numerical data
affidavit_dataset_vectorized = vectorizer.fit_transform(affidavit_dataset['filed_affidavit'])

# Save the numerical data to a file
np.save('affidavit_dataset_vectorized.npy', affidavit_dataset_vectorized)