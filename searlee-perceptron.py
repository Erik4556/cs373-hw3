import argparse
import numpy as np
import pandas as pd
import sys
import os

# Helper Functions


def eval_cond(i): # Helper function for determining which predictions were accurate
    return (i != 1)



class Perceptron(object):
    
    def __init__(self, blah=0):
        self.learning_rate = 0.01
        self.weights = np.zeros(7) # 7 features, 1 weight for each
        self.bias = 0

    
    def train(self, data):
        labels = data['survived'].values # Converting to np array for easier indexing
        testData = data.drop('survived', axis=1).values # Get all the data but the actual labels

        for i in range(22): # Loop as many times as we say to refine model
            for j in range(len(testData)): # For each row in test data
#                 print(testData[j])
#                 print('about to hit predict')
                prediction = self.predict(testData[j]) # Get our prediction from helper function
                outcome = (labels[j]-prediction)  
                
#                 print("Labels, prediction, testData",labels[j], prediction, testData[j])
                self.bias += outcome * self.learning_rate # Update bias and weight list
#                 print('outcome stuff', labels[j], prediction)
#                 print("weight stuff", outcome, testData[j])
                self.weights += outcome * testData[j] * self.learning_rate

    def predict(self, row):
#         print('HI')
#         print('predict stuff======', row, self.weights, self.bias)
        if (np.dot(row, self.weights)+self.bias) <= 0: # If the input row is predicted to be zero, return 0
            return 0
        else: # Else return 1
            return 1




# Driver Code

parser = argparse.ArgumentParser(description='CS373 Homework3 Naive Bayes')
parser.add_argument('trainData')
parser.add_argument('trainLabel')
parser.add_argument('testData')
parser.add_argument('testLabel')
args = parser.parse_args()

data_file = args.trainData
data_label = args.trainLabel
test_file = args.testData
test_label = args.testLabel


# data_file = "titanic-train.data"
# data_label = "titanic-train.label"
# test_file = "titanic-test.data"
# test_label = "titanic-test.label"


train_data = pd.read_csv(data_file, delimiter=',', index_col=None, engine='python')
train_label = pd.read_csv(data_label, delimiter=',', index_col=None, engine='python')
train_data["survived"] = train_label["survived"] # Adding the survived column onto the dataframe
train_data = train_data.fillna(train_data.mode().iloc[0]) # Replace NAs with the most frequent value in the column

test_data = pd.read_csv(test_file, delimiter=',', index_col=None, engine='python')
test_label = pd.read_csv(test_label, delimiter=',', index_col=None, engine='python')
test_data["survived"] = test_label["survived"] # Adding the survived column onto the dataframe
test_data = test_data.fillna(test_data.mode().iloc[0]) # Replace NAs with the most frequent value in the column



# ==============================================================================
# ==============================================================================
# ==============================================================================


                
                
# train
# For threshold times:
#  For rows in train_data:
#   call predict on the row, get a prediction out
#   set all but first weight to self.learning_rate * (label - prediction) * inputs
#   first input to self.learning_rate * (label-prediction)

# predict
# dot the inputs with all but the first weight, add the first weight to the sum
# If that comes out to over 0, predict 1, else predict 0. 
# Return

tempInput = train_data.drop('survived', axis=1).values
tempLabels = train_data['survived'].values

perceptron = Perceptron()
perceptron.train(train_data)

testInputData = test_data.drop('survived', axis=1).values
actual = test_data['survived'].values
predictions = []
for i in testInputData:
    predictions.append(perceptron.predict(i))
    
outcome = sum(1 for i in actual+predictions if eval_cond(i))
testAcc = outcome/len(test_data)

hingeLoss = 1-np.mean(1-(predictions*actual))
# print((predictions,actual))
print('Hinge LOSS=' + str(hingeLoss))
print('Test Accuracy='+str(testAcc))
