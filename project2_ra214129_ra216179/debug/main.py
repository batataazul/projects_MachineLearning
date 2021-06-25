import matplotlib.pyplot as plt
import numpy as np
import math
import json
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


def readData(path):
    try:
        f = open(path)
    except:
        print("Houston we've got a file problem")
        return
    dataset = [[], [], []]
    for i in f:
        stringList = i.split()
        try:
            for j in range(len(dataset)):
                if j < len(stringList):
                    dataset[j].append(float(stringList[j]))
                else:
                    f.close()
                    return
        except ValueError:
            print("You had a Value Error")
            f.close()
            return
        except:
            print("You got another Error")
            f.close()
            return
    f.close()
    return dataset


def splitData(dataset):
    trainset = [0, 1, 2]
    testset = [0, 1, 2]
    trainset[0], testset[0], trainset[1], testset[1], trainset[2], testset[2] = train_test_split(dataset[0], dataset[1],
                                                                                                 dataset[2],
                                                                                                 test_size=0.1,
                                                                                                 random_state=7)
    return trainset, testset


def getPointList(dataset):
    pointList = []
    for i in range(len(dataset[0])):
        point = []
        for j in dataset:
            point.append(j[i])
        pointList.append(point)
    return pointList


def getList(points):
    lists = []
    for i in range(len(points[0])):
        lists.append([])
    for i in points:
        for j in range(len(i)):
            lists[j].append(i[j])
    return lists


def plotPoints(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(0, 2, 2))
    Z = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, shade=False, color="g", alpha=.4)
    ax.plot([-2.5, -2.5, 2.5, 2.5], [0, 1, 1, 0], zdir='z', zs=0, c='g')
    ax.plot([-1.25, -1.25, 1.25, 1.25], [0, 0.05, 0.05, 0], zdir='z', zs=0, c='g')
    ax.scatter(dataset[0], dataset[1], dataset[2])
    return ax


def plotPolynom(dataset, X, Y, Z):
    ax = plotPoints(dataset)
    ax.plot(X, Y, Z)


def gettingX(start, end, step):
    lengh = end - start + step
    numb = int(lengh / step)
    x = np.linspace(start, end, num=numb)
    return x


def j(h, theta, x, y):
    summ = 0
    length = len(x)
    for i in range(length):
        summ += math.pow(h(theta, x[i]) - y[i], 2)
    return summ / (2 * length)


def derivative0(h, theta, x, y):
    summ = 0
    length = len(x)
    for i in range(length):
        summ += h(theta, x[i]) - y[i]
    return summ / length


def generalH(theta, x):
    summ = 0
    for i in range(len(theta)):
        summ += theta[i] * pow(x, i)
    return summ


def derivative(h, theta, x, y, grau):
    summ = 0
    length = len(x)
    for i in range(length):
        summ += (h(theta, x[i]) - y[i]) * pow(x[i], grau)
    return summ / length


def polynomialRegression(a, x, y, theta, err, grau):
    epoch = 0
    difference = 50
    prevJ = j(generalH, theta, x, y)
    while (difference > err):
        if (epoch > limit):
            break
        temp = [0] * (grau + 1)
        temp[0] = theta[0] - a * derivative0(generalH, theta, x, y)
        for i in range(1, grau + 1):
            temp[i] = theta[i] - a * derivative(generalH, theta, x, y, i)
        for i in range(len(temp)):
            theta[i] = temp[i]
        nowJ = j(generalH, theta, x, y)
        dif = abs(nowJ - prevJ)
        difference = dif / prevJ
        epoch += 1
        prevJ = nowJ
    print(epoch)
    print(j(generalH, theta, x, y))
    pass


a = 0.4
err = 0.00000001
limit = 70000
random.seed(7)

dataset = readData("../data/kick1.dat")
dataset2 = readData("../data/kick2.dat")

x = gettingX(1 / 60, 1 / 3, 1 / 60)
print(len(x))

grau = 1
theta2Xg2 = []
for i in range(grau + 1):
    theta2Xg2.append(random.uniform(-1, 1))
polynomialRegression(a, x, dataset2[0], theta2Xg2, err, grau)

grau = 1
theta2Yg2 = []
for i in range(grau + 1):
    theta2Yg2.append(random.uniform(0, 2))
polynomialRegression(a, x, dataset2[1], theta2Yg2, err, grau)

grau = 2
theta2Zg2 = []
for i in range(grau + 1):
    theta2Zg2.append(random.random())
polynomialRegression(a, x, dataset2[2], theta2Zg2, err, grau)

Xline2g2 = []
Yline2g2 = []
Zline2g2 = []
for i in x:
    Xline2g2.append(generalH(theta2Xg2, i))
    Yline2g2.append(generalH(theta2Yg2, i))
    Zline2g2.append(generalH(theta2Zg2, i))

plotPolynom(dataset2, Xline2g2, Yline2g2, Zline2g2)
