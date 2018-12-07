
# coding: utf-8

# In[13]:


import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math


# In[41]:


def read_csv(path):
    rReader_all = pd.read_csv(open(path, "r"))
    rReader = rReader_all[[
 'review_count',
 'useful',
 'funny',
 'cool',
 'fans',
 'compliment_hot',
 'compliment_more',
 'compliment_profile',
 'compliment_cute',
 'compliment_list',
 'compliment_note',
 'compliment_plain',
 'compliment_cool',
 'compliment_funny',
 'compliment_writer',
 'compliment_photos']
]
    matrix = np.asarray(rReader.fillna(0).values,dtype=np.float64)
    print(matrix)
    return matrix


# In[42]:


matrix = read_csv('yelp.csv')


# In[4]:


def randomInitialCluster(matrix, size):
    return random.sample(list(matrix),size)


# In[5]:


def getClosestCenter(randomCenter, user):
    distances = np.array(np.linalg.norm(randomCenter - user,axis=1))
    close = np.argmin(distances)
    return close


# In[40]:


def Q2_onlineKmeans(matrix, T , randomCenter, batchSize):
    for t in range(1,T):
        e = 1/t
        randomBatch = random.sample(list(matrix), batchSize)
        newCenter = randomCenter
        for user in randomBatch:
            close = getClosestCenter(randomCenter, user)
            newCenter[close] = np.add(newCenter[close], e * (user - randomCenter[close]))
        randomCenter = newCenter
    return randomCenter



def Q3_kMeansPlusPlus(matrix, k, batch_size):
    # first centroid
    centers = random.sample(list(matrix), 1)
    for i in range(k - 1):
        distance = 0
        sample_distance = []
        sample = random.sample(list(matrix), batch_size)
        for i in range(batch_size):
            distance += getDistance(centers, sample[i])
            sample_distance.append(distance)
        rand = random.random() * distance
        for i in range(batch_size):
            if (i == batch_size - 1) or (sample_distance[i] <= rand and sample_distance[i + 1] >= rand):
                centers.append(sample[i])
                break
    return centers

def getDistance(centers, user):
    distances = np.linalg.norm(centers - user, axis = 1)
    return pow(np.min(distances), 2)


# In[43]:


def Q4_initialCluster(matrix,k):
    dataSum = np.sum(matrix,axis = 1)
    newMatrix = np.c_[matrix,dataSum]
    a_arg = np.argsort(newMatrix[:,newMatrix.shape[1]-1]) 
    newMatrix = newMatrix[a_arg]
    newMatrix = np.delete(newMatrix,-1,axis = 1)
    totalCount = len(dataSum)
    oneCount = int (totalCount/k)
    centers = []
    c = newMatrix.shape[1]
    for i in range(k):
        center = [0] * c
        center = np.array(center)
        if i < k-1:
            for j in range(i*oneCount, (i+1)* oneCount):
                center = np.add(center,newMatrix[j])
            center = np.divide(center,oneCount)
        if i == k-1:
            for j in range(i*oneCount,totalCount):
                center = np.add(center,newMatrix[j])
            center = np.divide(center,totalCount - i*oneCount)
        centers.append(center)
    return centers


# In[21]:


def findClosestDistance(center,user):
    distances = np.array(np.linalg.norm(center - user,axis=1))
    return np.min(distances)


# In[44]:


def Q5_runData(center, sample):
    user = sample[0]
    d = findClosestDistance(center,user)
    sumDist = d
    maxDist = d
    minDist = d
    for s in sample[1:]:
        d = findClosestDistance(center,s)
        sumDist += d
        if d > maxDist:
            maxDist = d
        if d < minDist:
            minDist = d
    return sumDist/len(sample),maxDist,minDist

def tune_batchSize(sizeList):
    meanDistances = []
    sample = randomInitialCluster(matrix, 200)
    for size in sizeList:
        randomCenter = randomInitialCluster(matrix, k)
        center = Q2_onlineKmeans(matrix, 100, randomCenter,size)
        meanDist,maxDist,minDist = Q5_runData(center, sample)
        meanDistances.append(meanDist)
    return meanDistances


# In[45]:


listK = range(5,505,10)
sample = randomInitialCluster(matrix, 200)
means = []
mins = []
maxs = []
for k in listK:
    print('k:'  + str(k) + ' loading......')
    randomCenter = randomInitialCluster(matrix, k)
    center = Q2_onlineKmeans(matrix, 200, randomCenter,200)
    meanDist,maxDist,minDist = Q5_runData(center, sample)
    means.append(meanDist)
    mins.append(minDist)
    maxs.append(maxDist)
plt.figure()
plt.plot(listK, means)
plt.title('mean distance with different K')
plt.xlabel('k')
plt.ylabel('mean distance')
plt.show()

plt.figure()
plt.plot(listK, mins)
plt.title('min distance with different K')
plt.xlabel('k')
plt.ylabel('minimum distance')
plt.show()

plt.figure()
plt.plot(listK, maxs)
plt.title('maximum distance with different K')
plt.xlabel('k')
plt.ylabel('maximum distance')
plt.show()



listK = range(5,505,30)
sample = randomInitialCluster(matrix, 200)
means4 = []
mins4 = []
maxs4 = []
for k in listK:
    print('k:'  + str(k) + ' loading......')
    randomCenter = Q4_initialCluster(matrix,k)
    center = Q2_onlineKmeans(matrix, 200, randomCenter,200)
    meanDist,maxDist,minDist = Q5_runData(center, sample)
    means4.append(meanDist)
    mins4.append(minDist)
    maxs4.append(maxDist)
plt.figure()
plt.plot(listK, means4)
plt.title('mean distance with different K')
plt.xlabel('k')
plt.ylabel('mean distance')
plt.show()

plt.figure()
plt.plot(listK, mins4)
plt.title('min distance with different K')
plt.xlabel('k')
plt.ylabel('minimum distance')
plt.show()

plt.figure()
plt.plot(listK, maxs4)
plt.title('maximum distance with different K')
plt.xlabel('k')
plt.ylabel('maximum distance')
plt.show()



listK = range(5,505,30)
sample = randomInitialCluster(matrix, 200)
means4 = []
mins4 = []
maxs4 = []
for k in listK:
    print('k:'  + str(k) + ' loading......')
    randomCenter = Q3_kMeansPlusPlus(matrix, k, 300)
    center = Q2_onlineKmeans(matrix, 200, randomCenter,200)
    meanDist,maxDist,minDist = Q5_runData(center, sample)
    means4.append(meanDist)
    mins4.append(minDist)
    maxs4.append(maxDist)
plt.figure()
plt.plot(listK, means4)
plt.title('mean distance with different K')
plt.xlabel('k')
plt.ylabel('mean distance')
plt.show()

plt.figure()
plt.plot(listK, mins4)
plt.title('min distance with different K')
plt.xlabel('k')
plt.ylabel('minimum distance')
plt.show()

plt.figure()
plt.plot(listK, maxs4)
plt.title('maximum distance with different K')
plt.xlabel('k')
plt.ylabel('maximum distance')
plt.show()

