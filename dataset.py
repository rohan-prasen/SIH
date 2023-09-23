import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Generate a list of random words and phrases related to affidavits
words = ['affidavit', 'deposition', 'testimony', 'declaration', 'statement', 'court', 'judge', 'lawyer', 'plaintiff', 'defendant', 'witness', 'evidence', 'fact', 'truth', 'lie', 'perjury']

# Combine these words and phrases to create random sentences
def generate_sentence(words):
    sentence = ''
    for word in words:
        sentence += word + ' '
    return sentence.strip()

# Generate a dataset of 100 or more filed affidavits
filed_affidavits = []
for i in range(100):
    sentence = generate_sentence(words)
    filed_affidavits.append(sentence)

# Generate counter affidavits for each filed affidavit
counter_affidavits = []
for filed_affidavit in filed_affidavits:
    # Generate a counter affidavit that contradicts the filed affidavit
    counter_affidavit = 'I deny the allegations made in the filed affidavit.'
    counter_affidavits.append(counter_affidavit)

# Save the dataset to a file
with open('affidavit_dataset.csv', 'w') as f:
    f.write('filed_affidavit,counter_affidavit\n')
    for i in range(100):
        f.write(filed_affidavits[i] + ',' + counter_affidavits[i] + '\n')
