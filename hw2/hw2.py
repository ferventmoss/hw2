import numpy
import scipy.optimize
import homework2_starter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def getSets(X, y):
    sz = len(y);
    sets = [[X[:sz/3], y[:sz/3]], [X[sz/3:2*(sz/3)], y[sz/3:2*(sz/3)]], \
    [X[2*(sz/3):sz], y[2*(sz/3):sz]]]
    return sets

def getQ1Answer():
    X, y = homework2_starter.prepData()
    sets = getSets(X, y);
    theta = homework2_starter.train(1.0, sets[0][0], sets[0][1])
    [homework2_starter.validate(set[0], set[1], theta) for set in sets]

def getQ2Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    theta = homework2_starter.train(1.0, sets[0][0], sets[0][1])
    homework2_starter.validate(sets[0][0], sets[0][1], theta)
    homework2_starter.validate(sets[1][0], sets[1][1], theta)
    homework2_starter.validate(sets[2][0], sets[2][1], theta)

def getQ3Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    theta = homework2_starter.train(1.0, sets[0][0], sets[0][1])
    homework2_starter.validate(sets[2][0], sets[2][1], theta) 

def getQ4Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    N = len(sets[0][0])
    y1 = sum(sets[0][1])
    y0 = N-y1;
    a = N / (2*float(y1))
    b = N / (2*float(y0))
    theta = homework2_starter.trainWithWeight(1.0, sets[0][0], sets[0][1], a, b)
    [homework2_starter.validate(set[0], set[1], theta) for set in sets] 

def getQ5Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    N = len(sets[0][0])
    y1 = sum(sets[0][1])
    y0 = N-y1;
    a = N / (2*float(y1))
    b = N / (2*float(y0))
    lams = [0, 0.01, 0.1, 1, 10, 100, 1000] 
    thetas = [homework2_starter.trainWithWeight(lam, sets[0][0], sets[0][1], a, b) for lam in lams]
    
    setIdx = 0;
    setNames = ["Train", "Validation", "Test"]
    for set in sets:
        print setNames[setIdx]
        setIdx += 1
        idx = 0
        for theta in thetas:
            print "lamda = {}".format(lams[idx])
            homework2_starter.validate(set[0], set[1], theta)
            idx += 1

def getQ6Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    pca = PCA(n_components=len(X[0]))
    pca.fit(sets[0][0])
    print pca.components_

def getQ7Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    pca = PCA(n_components=len(X[0]))
    pca.fit(sets[0][0])
    # method 1
    print pca.explained_variance_[2:]
    print sum(pca.explained_variance_[2:])

    # method 2
    x = sets[0][0]
    Y = pca.transform(x)
    y_mean = numpy.mean(Y, axis = 0)

    error = 0
    for i in range(len(Y)):
        for j in range(2, 10):
            error += (Y[i][j] - y_mean[j])**2
    print error / len(Y)

def getQ8Answer():
    X, y = homework2_starter.prepData2()
    sets = getSets(X, y)
    pca = PCA(n_components=2)
    trainset = sets[0][0]
    pca.fit(sets[0][0])
    x = pca.components_[0]
    y = pca.components_[1]
    x_ipa = []
    y_ipa = []
    x_none = []
    y_none = []
    for idx in range(len(trainset)):
        if sets[0][1][idx]:
            x_sum = 0.0
            y_sum = 0.0
            for i in range(len(x)):
                x_sum = 0.0
                y_sum = 0.0
                for val in trainset[i]:
                    x_sum += x[i] * val
                    y_sum += y[i] * val
                x_ipa.append(x_sum)
                y_ipa.append(y_sum)
        else:
            for i in range(len(x)):
                x_sum = 0.0
                y_sum = 0.0
                for val in trainset[i]:
                    x_sum += x[i] * val
                    y_sum += y[i] * val
                x_none.append(x_sum)
                y_none.append(y_sum)

    plt.plot(x_ipa, y_ipa, 'ro', x_none, y_none, 'bs')
    plt.savefig('hw2graph.pdf')


getQ8Answer()

