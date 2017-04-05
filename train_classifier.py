'''
Training of classifier takes place here. Receives the training data and validation data from application. 
After running each algorithm, the highest accuracy classifier is sent for deployment
'''
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from dep_classifier import deploy_classifier
from modname import *

deploy = deploy_classifier()

class train_classifier:
		
	
	def trainer(self, uuid, X_train, Y_train, X_validation, Y_validation):
		
		knn = KNeighborsClassifier()	
		knn.fit(X_train, Y_train)
		predictions = knn.predict(X_validation)
		a = accuracy_score(Y_validation, predictions)
		print 'Knn accuracy -\t', a
	
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)
		predictions = rf.predict(X_validation)
		b = accuracy_score(Y_validation, predictions)
		print 'rf accuracy -\t', b
		
		lr = LogisticRegression()	
		lr.fit(X_train, Y_train)
		predictions = lr.predict(X_validation)
		c = accuracy_score(Y_validation, predictions)
		print 'lr accuracy -\t', c
	
		lda = LinearDiscriminantAnalysis()	
		lda.fit(X_train, Y_train)
		predictions = lda.predict(X_validation)
		d = accuracy_score(Y_validation, predictions)
		print 'lda accuracy -\t', d
	
		svm = SVC()	
		svm.fit(X_train, Y_train)
		predictions = svm.predict(X_validation)
		e = accuracy_score(Y_validation, predictions)
		print 'svm accuracy -\t', e
	
		dtc = DecisionTreeClassifier()	
		dtc.fit(X_train, Y_train)
		predictions = dtc.predict(X_validation)
		f = accuracy_score(Y_validation, predictions)
		print 'dt accuracy -\t', f
	
		nb = GaussianNB()
		nb.fit(X_train, Y_train)
		predictions = nb.predict(X_validation)
		g = accuracy_score(Y_validation, predictions)
		print 'nb accuracy -\t', g
		
	

		#Choosing best algm
		def alg(a, b, c, d, e, f, g):

			z = max(a,b,c,d,e,f,g)
			print 'Highest accuracy is -\t', z

			if a==(max(a,b,c,d,e,f,g)):
			 deploy.knn(X_train, Y_train, X_validation, Y_validation)
			elif b==(max(a,b,c,d,e,f,g)):
			 deploy.rf(X_train, Y_train, X_validation, Y_validation)
			elif c==(max(a,b,c,d,e,f,g)):
			 deploy.lr(X_train, Y_train, X_validation, Y_validation)
			elif d==(max(a,b,c,d,e,f,g)):
			 deploy.lda(X_train, Y_train, X_validation, Y_validation)
			elif e==(max(a,b,c,d,e,f,g)):
			 deploy.svm(X_train, Y_train, X_validation, Y_validation)
			elif f==(max(a,b,c,d,e,f,g)):
			 deploy.dt(X_train, Y_train, X_validation, Y_validation)
			elif g==(max(a,b,c,d,e,f,g)):
			 deploy.gnb(X_train, Y_train, X_validation, Y_validation)	
			else:
			 print('')


		alg(a,b,c,d,e,f,g)
	
		


	

	
	

