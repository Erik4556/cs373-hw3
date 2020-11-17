import numpy as np
import pandas as pd
import sys
import os

# 1. Split data into 2 dataframes by survived 1/0
# 2. Find mean/std dev of each column for both DFs 
# 3. Calculate the Gaussian PDF for 

def nbc(train_data):
    # Splitting the data based on survived being 0 or 1
    zero_data = train_data[train_data['survived'] == 0]
    one_data = train_data[train_data['survived'] == 1]

    # Prior Prob Calc
    zeroProb = len(zero_data)/len(train_data)
    oneProb = len(one_data)/len(train_data)
    priorProbs = [zeroProb, oneProb]
#     print("Prior Probabilities: ",zeroProb, oneProb)

    
    # Conditional Prob Calc
    condiProb = {} # Making a dict to store each feature's condi probs
    y = train_data['survived'].unique()
    for i in train_data: # For each feature
        # Calculate the conditional probability for each discrete value in the feature
        # Split by survived or not
        # featureProbs = np.zeros( (len(train_data['survived'].unique()),len(train_data[i].unique()) ) )
#         print(i)
        featureProbs = np.zeros( (2,len(list(train_data[i].unique()))) ) 
        
        
        
        x = list(set(train_data[i]))
#         print(max(y), y)
#         print(max(x), x)
        
        for j in [0,1]: # For survived = 0 and 1
        
            for k in range(0,len(x)): # For each unique value in the feature
#                 print(featureProbs)
#                 print(featureProbs[j,k])
                
                if k+1 in train_data[i]: # If the value is in the feature
#                     try:
#                         targetVals = (train_data['survived']==j)&(train_data[i]==x[k])
#                     except:
#                         print(i,j,k, x)
                    targetVals = (train_data['survived']==j)&(train_data[i]==x[k])
                    featureProbs[j,k] = (train_data.loc[targetVals,].shape[0]+1)/(sum(train_data['survived']==j)+6)
                else:
                    featureProbs[j,k] = 1/(sum(train_data['survived']==j)+6)

                    
                
        featureProbs = pd.DataFrame(featureProbs, index=[0,1], columns=x)
        condiProb[i] = featureProbs
    return priorProbs, condiProb
    # We access a given feature from condiProb, which we can index as 0 or 1 to get the probs
    # for survived = 0 or 1. This is a dictionary which we can query a value from the feature to access a prob
        
    
    
    
    
    
    
    
    
    
# PREDICTION

def predict(pProb, cProb, x):
    # We want to find a probability for both 0 and 1 outcomes for survived
    # Thus, we want to multiply our prior probability, by the conditional probability of every single 
    # feature given for a specific row x.

    
    # We need to consider prediction values outside of the ones we explicitly defined in nbc
    features = ['Pclass','Sex','Age','Fare','Embarked','relatives','IsAlone']
    zeroProb = pProb[0] # Start with prior prob
    oneProb = pProb[1]
    for i in range(len(features)):
#         print(features[i])
#         print(cProb[features[i]])
#         print(x[i])
#         print(cProb[features[i]][x[i]][0])
        try: # If it's in the set, continue as normal
            zeroProb = zeroProb * cProb[features[i]][x[i]][0]
#             print(cProb[features[i]][x[i]][0])
        except: # If not, multiply instead by the general value given for 0
            zeroProb = zeroProb * (1/(sum(train_data['survived']==0)+len(features)))
            
    # Same thing for 1
        
    for i in range(len(features)):
        try: # If it's in the set, continue as normal
            oneProb = oneProb * cProb[features[i]][x[i]][1]
#             print(cProb[features[i]][x[i]][1])
        except: # If not, multiply instead by the general value given for 1
            oneProb = oneProb * (1/(sum(train_data['survived']==1)+len(features)))
    
    
#     zeroProb = pProb[0]*cProb['Pclass'][x[0]][0]*cProb['Sex'][x[1]][0]*cProb['Age'][x[2]][0]*cProb['Fare'][x[3]][0]*cProb['Embarked'][x[4]][0]*cProb['relatives'][x[5]][0]*cProb['IsAlone'][x[6]][0]
#     oneProb = pProb[1]*cProb['Pclass'][x[0]][1]*cProb['Sex'][x[1]][1]*cProb['Age'][x[2]][1]*cProb['Fare'][x[3]][1]*cProb['Embarked'][x[4]][1]*cProb['relatives'][x[5]][1]*cProb['IsAlone'][x[6]][1]
#     print(zeroProb,oneProb)
#     print()
    if oneProb>zeroProb:
#         print("one")
        return 1
    else:
#         print("zero")
        return 0
    
def eval_cond(i): # Helper function for determining which predictions were accurate
    return (i != 1)
    
    
    
    
    
    
    
    
# Driver Code


import argparse

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

#print(train_data.head())

# Shortening these variables for readability, they are Prior, and Conditional Probability respectively
pProb, cProb = nbc(train_data)









# test = [2.0, 1.0, 34.0, 32.0, 0.0, 2.0, 0.0]
# predict(pProb, cProb, test)


# Evaluating test data
# print(test_data.head(50))
predictons = []
for i in range(len(test_data)):
    actual = test_data['survived']
    
    predictons.append(predict(pProb, cProb, list(test_data.iloc[i,:-1]))) # Predict 1 row from the test data
#     print(predict(pProb, cProb, list(test_data.iloc[i,:-1])))
outcome = sum(1 for i in actual+predictons if eval_cond(i))
# print("accurate predictions: ", outcome)
# print("out of: ", len(test_data))
# print()


# print(test_data.head())
# print("actual: ", actual[0:10])
# print("predictons: ", predictons[0:10])


# Calculating zero-one loss
outcome = sum(1 for i in actual+predictons if eval_cond(i))
zeroOneLoss = 1-(outcome/len(test_data))
# From the handout, we calculate zero-one loss by summing a series of 1s that occur whenever our prediction is different
# than reality, then divide it by the number of instances being counted.


# Calculating squared Loss
squaredLoss = np.mean((actual-predictons)*(actual-predictons))
testAcc = outcome/len(test_data)

print("ZERO-ONE LOSS="+str(zeroOneLoss))
print("SQUARED LOSS="+str(squaredLoss)+" Test Accuracy="+str(testAcc))

# Pclass	Sex	Age	Fare	Embarked	relatives	IsAlone	survived
