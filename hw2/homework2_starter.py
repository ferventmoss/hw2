import numpy
import scipy.optimize
import random
from math import exp
from math import log

def parseData(fname):
  for l in open(fname):
    yield eval(l)


def feature(datum):
  feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
  return feat

def prepData():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    X = [feature(d) for d in data]
    y = [d['beer/ABV'] >= 6.5 for d in data]
    return X, y

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam, a = 0, b = 0):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam, a = 0, b = 0):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

# NEGATIVE Log-likelihood
def fWithWeight(theta, X, y, lam, a, b):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    if y[i]:
      loglikelihood -= (a * log(1 + exp(-logit)))
    if not y[i]:
      loglikelihood -= (b * log(1 + exp(-logit)))
      loglikelihood -= (b * logit)
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood with Weight
def fprimeWithWeight(theta, X, y, lam, a, b):
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      if y[i]:
        dl[k] += (a * X[i][k] * (1 - sigmoid(logit)))
      else:
        dl[k] += (b * X[i][k] * (1 - sigmoid(logit)))
        dl[k] -= (b * X[i][k])
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])
##################################################
# Train                                          #
##################################################

def train(lam, X, y):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X, y, lam))
  return theta
def trainWithWeight(lam, X, y, a, b):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(fWithWeight, [0]*len(X[0]), fprimeWithWeight, pgtol = 10, args = (X, y, lam, a, b))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta, X, y):
  scores = [inner(theta,x) for x in X]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y)]
  tp = [(a==b and a == True) for (a,b) in zip(predictions,y)]
  tn = [(a==b and a == False) for (a,b) in zip(predictions,y)]
  fp = [(a!=b and a == True)  for (a,b) in zip(predictions,y)]
  fn = [(a!=b and a == False) for (a,b) in zip(predictions,y)]
  tpr = sum(tp) / float(sum(tp) + sum(fn))
  tnr = sum(tn) / float(sum(tn) + sum(fp))
  print "tp = {}, tn = {}, fp = {}, fn = {}".format(sum(tp), sum(tn), sum(fp), sum(fn))
  print "len(y) = {}, len(X) = {}".format(len(y), len(X))
  ber = 1 - 0.5 * (tpr + tnr)
  print "ber = {}".format(ber)
  acc = sum(correct) * 1.0 / len(correct)
  return acc

##################################################
# Validation pipeline                            #
##################################################
def validate(X, y, theta):
    acc = performance(theta, X, y)
    print("accuracy=" + str(acc))

def wordCounts(sentence, indexer): 
    counts = [0] * 10
    sentence = sentence.lower()
    for word in sentence.split():
        if word in indexer:
            counts[indexer[word]] += 1
            
    return counts

def feature2(datum, indexer):
  return wordCounts(datum['review/text'], indexer)

def prepData2():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    words = ["lactic", "tart", "sour", "citric", "sweet", "acid", "hop", "fruit", "salt", "spicy"]
    indexer = {}
    index = 0
    for word in words:
        indexer[word] = index
        index += 1

    X = [feature2(d, indexer) for d in data]
    y = [d['beer/ABV'] >= 6.5 for d in data]
    return X, y

def prepData3():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    words = ["lactic", "tart", "sour", "citric", "sweet", "acid", "hop", "fruit", "salt", "spicy"]
    indexer = {}
    index = 0
    for word in words:
        indexer[word] = index
        index += 1

    X = [feature2(d, indexer) for d in data]
    y = [d['beer/style'] == 'American IPA' for d in data]
    return X, y
