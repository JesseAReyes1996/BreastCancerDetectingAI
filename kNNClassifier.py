#Written using Python 3.6.1
import numpy as np
import matplotlib.pyplot as plt
import random

#Open file and store in array
with open("breast-cancer-wisconsin.txt", 'r') as data:
    toSplit = data.read().splitlines()
    bCancer = []
    for i in toSplit:
        bCancer.append(i.split(','))

#Missing values will be filled with the attribute mean for all samples belonging
#to the same class

#Cleaning the data
missingData = []
for i in range(len(bCancer)):
    for j in range(len(bCancer[i])):
        #Check each row for missing data and add which columns need to be filled
        #to missingData
        if(bCancer[i][j] == '?' and (j not in missingData)):
            missingData.append(j)

for i in missingData:
    #Number to divide by to obtain average
    benignCnt = 0
    malignCnt = 0

    #Total sum for each attribute
    benignAvg = 0
    malignAvg = 0

    #Obtain data to calculate average for each attribute with missing data
    for j in range(len(bCancer)):
        if(bCancer[j][i] != '?' and bCancer[j][-1] == '2'):
            benignCnt += 1
            benignAvg += int(bCancer[j][i])
        elif(bCancer[j][i] != '?' and bCancer[j][-1] == '4'):
            malignCnt += 1
            malignAvg += int(bCancer[j][i])

    #Attribute average
    benignAvg /= benignCnt
    malignAvg /= malignCnt

    #Replace '?' with average for that attribute
    for j in range(len(bCancer)):
        if(bCancer[j][i] == '?' and bCancer[j][-1] == '2'):
            bCancer[j][i] = benignAvg
        elif(bCancer[j][i] == '?' and bCancer[j][-1] == '4'):
            bCancer[j][i] = malignAvg

#Data has been cleaned

#Determine the distance between two "points," a single point containing every
#attribute of a given element in a data point
def distance(x,y,p):
    toSquare = []
    for i in range(len(y)):
        toSquare.append(np.power(abs(float(x[i]) - float(y[i])),p))
    return np.power(sum(toSquare),1/p)

#Return a list of predicted labels for the test set
def knn_classifier(x_test, x_train, y_train, k, p):
    y_pred = []
    for i in x_test:
        #Obtain the distance from every test point to every training point
        neighborDist = []
        labelVal = 0
        for j in x_train:
            neighborDist.append((distance(i, j, p), y_train[labelVal]))
            labelVal += 1
        neighborDist.sort()

        #Trim the list to only the the K nearest distances
        k_distances = []
        for l in range(k):
            k_distances.append(neighborDist[l])

        #Count how many of the K nearest neigbors are benign/malignant
        benignCnt = 0
        malignCnt = 0
        for m in range(k):
            if(k_distances[m][1] == '2'):
                benignCnt += 1
            elif(k_distances[m][1] == '4'):
                malignCnt += 1

        #Predict the label of the point depending on its neighbor's labels
        if(benignCnt > malignCnt):
            y_pred.append('2')
        elif(malignCnt > benignCnt):
            y_pred.append('4')
        elif(benignCnt == malignCnt):
            y_pred.append('2')

    return y_pred

#(Data set to cross validate, # folds, p for distance function, # neighbors)
def kCrossValidation(testSet, k, p, kNN):
    labelList = []
    accuracyList = []
    sensitivityList = []
    specificityList = []

    points = []
    labels = []

    #Randomly shuffle the data points
    random.shuffle(testSet)
    for i in testSet:
        temp = i[1:-1]
        points.append(temp)
        labels.append(i[-1])

    partionedLists = []
    amountInPartion = int(len(testSet)/k)
    intervalBeg = 0
    intervalEnd = amountInPartion

    #Append attributes/labels to the partioned list
    for i in range(k):
        partionedLists.append((points[intervalBeg:intervalEnd], labels[intervalBeg:intervalEnd]))
        intervalBeg += amountInPartion
        intervalEnd += amountInPartion

    #Begin the cross validation
    for i in range(k):
        testers = partionedLists[i][0]
        trainPoints = []
        trainLabels = []

        for idxJ, j in enumerate(partionedLists):
            if(i == idxJ):
                continue
            else:
                for idxL, l in enumerate(j[0]):
                    trainPoints.append(l)
                    trainLabels.append(j[-1][idxL])

        labelList.append(knn_classifier(testers, trainPoints, trainLabels, kNN, p))

    #Calculate the accuracy for each fold

    #labelList contains the predicted labels
    for idxI, i in enumerate(labelList):
        correct = 0
        for idxJ, j in enumerate(i):
            #partionedLists[idxI][-1] contains a list of correct labels
            if(j == partionedLists[idxI][-1][idxJ]):
                    correct += 1

        accuracyList.append(correct/len(i))

    accuracyList = np.array(accuracyList)

    #Calculate the sensitivity for each fold
    for idxI, i in enumerate(labelList):
        TP = 0
        P = 0
        for idxJ, j in enumerate(i):
            #Calculate correctly predicted positives by comparing correct label(partionedLists) and predicted label(j)
            if(j == partionedLists[idxI][-1][idxJ] and j == '4'):
                    TP += 1
            #Total amount of positives
            if(partionedLists[idxI][-1][idxJ] == '4'):
                    P += 1
        sensitivityList.append(TP/P)
    sensitivityList = np.array(sensitivityList)

    #Calculate the specificity for each fold
    for idxI, i in enumerate(labelList):
        TN = 0
        N = 0
        for idxJ, j in enumerate(i):
            #Calculate correctly predicted negatives by comparing correct label(partionedLists) and predicted label(j)
            if(j == partionedLists[idxI][-1][idxJ] and j == '2'):
                TN += 1
            #Total amount of negatives
            if(partionedLists[idxI][-1][idxJ] == '2'):
                N += 1
        specificityList.append(TN/N)
    specificityList = np.array(specificityList)

    accMean = np.mean(accuracyList)
    accSdDv = np.std(accuracyList)
    senMean = np.mean(sensitivityList)
    senSdDv = np.std(sensitivityList)
    speMean = np.mean(specificityList)
    speSdDv = np.std(specificityList)

    return accMean, accSdDv, senMean, senSdDv, speMean, speSdDv

#Randomly fill the training data
#random.seed(10)#For testing purposes
trainingSet = random.sample(range(0,699), 559)
trainingPoints = []
trainingLabels = []
for i in trainingSet:
    temp = bCancer[i][1:-1]
    trainingPoints.append(temp)
    trainingLabels.append(bCancer[i][10])

checker = []

#Fill the test points
testPoints = []

for i in range(len(bCancer)):
    if i not in trainingSet:
        temp = bCancer[i][1:-1]
        checker.append(bCancer[i][-1])
        testPoints.append(temp)

#For Cross-Validation
measures = []
for i in range(10):
    measures.append(kCrossValidation(bCancer, 10, 2, (i+1)))

numNeighbors = [1,2,3,4,5,6,7,8,9,10]

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]
ax.errorbar(numNeighbors, [i[0] for i in measures], yerr=[i[1] for i in measures], ecolor="green")
ax.set_title("Accuracy")
ax = axs[0,1]
ax.errorbar(numNeighbors, [i[2] for i in measures], yerr=[i[3] for i in measures], ecolor="green")
ax.set_title("Sensitivity")
ax = axs[1,0]
ax.errorbar(numNeighbors, [i[4] for i in measures], yerr=[i[5] for i in measures], ecolor="green")
ax.set_title("Specificity")

fig.text(0.5, 0.05, "k-Neighbors", ha="center", va="center")
fig.text(0.05, 0.5, "Performance", ha="center", va="center", rotation="vertical")

fig.suptitle("Cross-Validation: P=2")

plt.show()
