import statsmodels.formula.api as smf
from sklearn import linear_model
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import numpy
from scipy import stats
from sklearn.svm import SVR


def hat_matrix(X1):#, X2): #Hat Matrix
    hat_mat =  numpy.dot(numpy.dot(X1, numpy.linalg.inv(numpy.dot(X1.T, X1))), X1.T)
    return hat_mat


def pearson(y_true, y_pred):
    err = y_true - y_pred
    SE = err * err
    PRESS = np.sum(SE)
    y_avg = np.mean(y_true)
    n = len(y_true)
    y_mean = [y_avg for i in range(n)]
    err = y_true - y_mean
    SE = err * err
    TSS = np.sum(SE)
    r2 = 1 - PRESS/TSS
    return r2

def training_test(training, test, response,clf):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    from scipy import stats

    y = training.LogBIO
    X = training.drop([response], axis=1)
    remaining = set(X.columns)
    y_train = y
    X_train = X
    #remaining.remove(response)
    n = len(X)
    p = 0
    for i in remaining:
        p = p + 1
    print("length:",p)
    lr1 = clf
    lr1.fit(X, y)
    pred = lr1.predict(X)
    import scipy.stats as stats
    r, p = stats.pearsonr(y, pred)
    q = q2f2(y, pred)
    q2 = max(r*r, q)
    
    print("r2 (tr):", q2)
    #print("BIC (tr):", b_ic)
    #print("AIC_m1.5 (tr):", aic_m15)
    rou = stats.spearmanr(y, pred)
    #print("rou (tr):", rou.correlation)
    #print("slope (training):", slope)
    #print("intercept (training):", intercept)
    qm2(training, 'LogBIO', clf)
    y_LOO = q2_LOO(training, clf)
    AE = abs(y-y_LOO)
    train = pd.DataFrame({'y':y, 'y_pre':y_LOO, 'y_cal':pred, 'AE':AE})
    train.to_csv('Train.csv')
    q2_ven(clf)

    print("----------------------------------")

    y = test.LogBIO
    y_test = y
    X = test.drop([response], axis=1)
    X_test = X
    predicted = lr1.predict(X)
    pred_test = predicted
    data = pd.DataFrame({'y':y, 'y1':predicted})
    data.to_csv('Test.csv')
    #qf2 = q2f2(y_test, predicted)
    r, p = stats.pearsonr(y_test, predicted)
    print("r2(test):", r)
    k1, b1, r_value, p_value, std_err = stats.linregress(y, predicted)
    k1 = abs(k1-1)
    b1 = abs(b1)
    k2, b2, r_value, p_value, std_err = stats.linregress(predicted, y)
    k2 = abs(k2-1)
    b2 = abs(b2)
    k = min(k1, k2)
    b = min(b1, b2)
    print("slope (test):", k)
    print("intercept (test):", b)
    #qf2 = q2f2(y_test, predicted0)

    rou = stats.spearmanr(y, predicted)
    print("rou(test):", rou.correlation)
    ae = abs(y-predicted)
    MAE = np.mean(ae)
    print("MAE(test):", MAE)
    PRESS, TSS = SS(y_train, y_test, pred_test)
    qf3 = q2f3(y_train, y_test, pred_test)
    qf2 = q2f2(y_test, pred_test)
    qf1 = q2f1(y_train, y_test, pred_test)
    cc = ccc(y_train, y_test, pred_test)
    print("PRESS:", PRESS)
    print("TSS:", TSS)
    print("q2f3:", qf3)
    print("q2f2:", qf2)
    print("q2f1:", qf1)
    print("ccc:", cc)
            
def q2_LOO(data, clf):
    n = len(data)
    y = data.LogBIO
    X = data.drop(['LogBIO'], axis=1)
    #lr = clf
    y_pred = cross_val_predict(clf, X, y, cv = n)
    """lr = clf(fit_intercept=False)
    predicted0 = cross_val_predict(lr, X, y, cv = n)"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y0':predicted0})"""
    #r2 = smf.ols('y~y1', cv).fit().rsquared_adj
    """r02 = smf.ols('y~y0',cv).fit().rsquared_adj"""
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
    r, p = stats.pearsonr(y, y_pred)
    q = q2f2(y, y_pred)
    q2 = max(r*r, q)
    print("q2(LOO):", q2)
    return y_pred

def SS(y_train, y_test, pred_test):
  PRESS = numpy.sum(y_test-pred_test)**2
  #SSRes = SSRes/len(y_test)
  TSS = numpy.sum((y_train-numpy.mean(y_train))**2)
  #SStot = SStot/len(y_train)
  #r2 = 1 - (SSRes/SStot)
  return PRESS, TSS

def q2f3(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SSRes = SSRes/len(y_test)
  SStot = numpy.sum((y_train-numpy.mean(y_train))**2)
  SStot = SStot/len(y_train)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f2(y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f1(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_train))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def ccc(y_train, y_test, pred_test):
  SSRes = numpy.sum((y_test-numpy.mean(y_test))*(pred_test-numpy.mean(pred_test)))*2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2) + numpy.sum((pred_test-numpy.mean(pred_test))**2) + len(pred_test) * (numpy.mean(y_test)-numpy.mean(pred_test))**2
  r2 = SSRes/SStot
  return r2

def qm2(data, response, clf):
    n = len(data)
    y = data.LogBIO
    X = data.drop([response], axis=1)
    lr = clf
    predicted = cross_val_predict(lr, X, y, cv = 4)
    #lr = clf(fit_intercept=False)
    #predicted0 = cross_val_predict(lr, X, y, cv = n)
    #cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    r, p = stats.pearsonr(y, predicted)
    q = q2f2(y, predicted)
    q2 = max(r*r, q)
    print("q2(RAN):", q2)
    #r02 = smf.ols('y~y0',cv).fit().rsquared
                
    #import math
    #rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
                        
    #return rm2

def q2_ven(clf):
    cor = pd.read_csv("Correct_2.csv")

    result = pd.DataFrame(columns=['y_exp', 'y_pre'])
    
    group = 0
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 1
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 2
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 3
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)

    y_true = result.y_exp
    y_pred = result.y_pre
    r, p = stats.pearsonr(y_true, y_pred)
    q= q2f2(y_true, y_pred)
    q2 = max(r*r, q)
    #r2 = smf.ols('y_exp~y_pre', result).fit().rsquared
    print("q2(VEN):", q2)

train_file = "Correct_2.csv"
training = pd.read_csv(train_file)
test_file = "Predict_2.csv"
test = pd.read_csv(test_file)

#clf = linear_model.LinearRegression()
clf = SVR(gamma='scale')
training_test(training, test, 'LogBIO', clf)