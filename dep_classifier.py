'''
This file consists of all the classfiers taken into consideration. It is called after training and validation data is complete
'''
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from modname import *

class deploy_classifier:
	
	def knn(self, X_train, Y_train, X_validation, Y_validation):
	        knn = KNeighborsClassifier()	
	        knn.fit(X_train, Y_train)
	        predictions = knn.predict(X_validation)
	        a = accuracy_score(Y_validation, predictions)
	        print('\n\tBest alg is KNN')
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))


	def rf(self, X_train, Y_train, X_validation, Y_validation):
	        rf = RandomForestClassifier()
	        rf.fit(X_train, Y_train)
	        predictions = rf.predict(X_validation)
	        b = accuracy_score(Y_validation, predictions)
	        print('\n\tBest alg is RF')
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

	def lr(self, X_train, Y_train, X_validation, Y_validation):
	        print('\n\tLOGISTIC REGRESSION')
	        lr = LogisticRegression()	
	        lr.fit(X_train, Y_train)
	        predictions = lr.predict(X_validation)
		c = accuracy_score(Y_validation, predictions)
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

	def lda(self, X_train, Y_train, X_validation, Y_validation):
	        print('\n\tLINEAR DISCRIMINANT ANALYSIS')
	        lda = LinearDiscriminantAnalysis()	
	        lda.fit(X_train, Y_train)
	        predictions = lda.predict(X_validation)
		d = accuracy_score(Y_validation, predictions)	        
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

	def svm(self, X_train, Y_train, X_validation, Y_validation):
	        print('\n\tSUPPORT VECTOR MACHINES')
	        svm = SVC()	
	        svm.fit(X_train, Y_train)
	        predictions = svm.predict(X_validation)
		e = accuracy_score(Y_validation, predictions)
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

	def dt(self, X_train, Y_train, X_validation, Y_validation):
	        print('\n\tDECISION TREE')
	        dtc = DecisionTreeClassifier()	
	        dtc.fit(X_train, Y_train)
	        predictions = dtc.predict(X_validation)
		f = accuracy_score(Y_validation, predictions)
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

	def gnb(self, X_train, Y_train, X_validation, Y_validation):
	        print('\n\tGAUSSIAN NB')
	        nb = GaussianNB()
	        nb.fit(X_train, Y_train)
	        predictions = nb.predict(X_validation)
		g = accuracy_score(Y_validation, predictions)
	        print(confusion_matrix(Y_validation, predictions))
	        print(classification_report(Y_validation, predictions))

