import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import wordpunct_tokenize

'''nltk is a widely used pakage in natural language process
   tokenize devide string to a series of substring'''

def getword_mail(email):
    mail = email[0]
    label = email[1]
    wordset = set(wordpunct_tokenize(mail.replace('=\\n', '').lower()))
    return wordset, label

def train(dataset):
    train_likelihood = {}
    for email in dataset:
        mail, label = getword_mail(email)
        for word in mail:
            if word in train_likelihood:
                train_likelihood[word] += 1
            else: train_likelihood[word] = 1
    for word in train_likelihood.keys():
        train_likelihood[word] = float(train_likelihood[word])/len(dataset)
    return train_likelihood

def test(dataset, spam_likelihood, ham_likelihood):
    post_spam = 1
    post_ham = 1
    error = 0
    result = np.zeros((len(dataset),2))
    for i in range(len(dataset)):
        email = dataset[i]
        wordset, label = getword_mail(email)
        for word in wordset:
           if word in spam_likelihood:
               post_spam *= spam_likelihood[word]
           if word in ham_likelihood:
               post_ham *= ham_likelihood[word]
           #else: post_spam *= 0.8
        if post_spam>post_ham:
            cla = 1
        if post_spam<post_ham:
            cla = 0
        if cla!=label:
            error += 1
        result[i,0] = label
        result[i,1] = cla
    error /= len(dataset)
    return result, error


#if __name__ == '__main__':
doc_path = 'C:/Users/Phyorch/Desktop/Learning/Python repository/hello-world/data/lab6_assignment1_data.csv'
data = pd.read_csv(doc_path)
data.as_matrix
dataset = np.array(data)
spam_dataset = dataset[0:1000]
ham_dataset = dataset[2000:5000]
test_dataset = dataset[1000:2000]
spam_likelihood = train(spam_dataset)
ham_likelihood = train(ham_dataset)
result, error = test(test_dataset, spam_likelihood, ham_likelihood)
a = 1

# using list but not set, we can retain the frequence information
