import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pickle

def fit_predict(X_train, y_train, X_test, y_test, args={}):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    #Dump the model
    pickle_file = 'log_reg_model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    return y_pred

