'''
This file retreives sensor data from the Building Depot and then runs a set of classifiers as long as new data is found. Algorithms are compared 
and the classifier with the highest accuracy is chosen and the results for the same is displayed including a report consisting of precision, f1 and suppport
'''
from buildingdepot_helper import BuildingDepotHelper
from train_classifier import train_classifier
from time import time
import os
from os import listdir
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from modname import *



helper = BuildingDepotHelper()
helper.get_oauth_token()
trainer = train_classifier()
'''
This function checks for missing values
'''

def num_missing(x):
  return sum(x.isnull())
'''
Creates labels here. Modify it based on the application. Here brightness is considered for a Lifx
'''

def brightness_to_label(brightness):
	if brightness > 0.400:
		return 0
	elif brightness > 0.300:
		return 1
	elif brightness > 0.200:
		return 2
	else:
		return 3
'''
Mention the uuid of the sensor. Checks the database and passes the stream in here.
Data frames consist each stream of data. 
Values of X,Y marks the respective columns under consideration. Seed is any random number
Validation size can be changed based on the size considered for training.
Data is split into training data and validation data. 
Mean and Standard Deviation for each classifier is calculated and ma
'''


def classify(start_time, end_time):
	#Get time series data
	uuid = '3089fce9-7929-4cc0-b7d9-707230791a3b'
	hue = helper.get_timeseries_data(uuid, start_time, end_time)

	hue = helper.get_timeseries_data('3089fce9-7929-4cc0-b7d9-707230791a3b',start_time,end_time)
	brightness = helper.get_timeseries_data('993b19cd-6989-4966-811f-2e165af6c2a2',start_time,end_time)

	if (len(hue) == 0 or len(brightness) == 0):
		print('No data')
		return False
	print('Found data')

	df_input = {
	  'hue': hue
	}


	df_input1 = {
	 'brightness': brightness
	}


	x = DataFrame(df_input)
	y = DataFrame(df_input1)

	dataset = x.join(y)
	#dataset.hist()
	#plt.show()

	# Split-out validation dataset
	labels = Series([brightness_to_label(x) for x in dataset['brightness']])
	dataset['labels'] = labels
	#print(dataset)
	array = dataset.values
	X = array[:,0:2]
	Y = array[:,2]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	models.append(('RF', RandomForestClassifier()))

	# evaluate each model in turn for mean and standard deviation
	results = []
	names = []
	j = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)


	# Compare Algorithms
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

	trainer.trainer(uuid, X_train, Y_train, X_validation, Y_validation)
	
	return True

#Mention the Start time and End time
start_time = 1488978420
end_time = 1489005000
successful = classify(start_time, end_time)

while successful:
	start_time = end_time
	end_time += 5 * 60 * 60
	successful = classify(start_time, end_time)

