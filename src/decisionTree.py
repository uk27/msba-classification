# Begin Step 1a
# Use Step 1a if you are setting up a .py file to run on Amazon Web Services (AWS)...
# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context needs to be created.
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://project-bucket-group5/data/boston-data"
#output_path = "s3a://msbatutorial/output"

# Boston file
filepath = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(filepath)
bostonRDD.cache()
# End Step 1a


# Begin Step 1b
# Use Step 1b if you are setting up your local machine to run the model
# Import Relevant Libraries
#from pyspark import SparkConf, SparkContext
#from pyspark.ml import Pipeline
#from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
#from pyspark.mllib.util import MLUtils
#from pyspark.mllib.regression import LabeledPoint
#from numpy import array

#Spark context only needs to be called
#sc

# Write the absolute path for the file you wish to use
#filepath = "file:/home/training/training_materials/data/boston50.txt"
#bostonRDD = sc.textFile(filepath)
#bostonRDD.cache
# End Step 1b

# All code below is used for both AWS and your local machine.
# Replace '\t' with ',' to create a comma separated dataset
new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

# Use the decision tree algorithm to fit a model
model = DecisionTree.trainClassifier(training_boston, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

# Use the model we got from last step, we do the perdiction on each records and get their classifications.
prediction = model.predict(training_boston.map(lambda x: x.features))

# Combine the original training data classification labels and the we predicted. 
labelsAndPredictions = training_boston.map(lambda lp: lp.label).zip(prediction)

# Comparing these two labels and geting the error rate.
train_Err = labelsAndPredictions.filter(lambda (v, p): v != p).count()/float(training_boston.count())

print('..........................................................................................................')
# Print Training Error
print('Training error = ' + str(train_Err))
print('..........................................................................................................')

# Tests the predictions from the model
prediction_test = model.predict(test_boston.map(lambda x: x.features))

# Tests labels & predictions
labelsAndPredictions_test = test_boston.map(lambda lp: lp.label).zip(prediction_test)

# Calculating the accuracy of the model on the observed data
accuracy_boston = labelsAndPredictions_test.filter(lambda (v, p): v == p).count()/float(test_boston.count())

print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished Classifitation Tree Model-----') #We got .88
print('..........................................................................................................')