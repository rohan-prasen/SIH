import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(affidavit_dataset_vectorized, affidavit_dataset['counter_affidavit'], test_size=0.2)

# Transform the text data in the X_train dataset into numerical data
X_train_vectorized = vectorizer.fit_transform(X_train)