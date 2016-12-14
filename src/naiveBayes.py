# Begin Step 1a
# Use Step 1a if you are setting up a .py file to run on Amazon Web Services (AWS)...
# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context needs to be created.
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://project-bucket-group5/data/boston-data"
#output_path = "s3a://project-bucket-group5/data/boston-data"

# Boston file
filepath = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(filepath)
bostonRDD.cache()
# End Step 1a


# Begin step 1b
# Use Step 1b if you are setting up your local machine to run the model
# Import Relevant Libraries
#from pyspark import SparkConf, SparkContext
#from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
#from pyspark.mllib.util import MLUtils
#from pyspark.mllib.regression import LabeledPoint
#from numpy import array

#Spark context only needs to be called
#sc

# Write the absolute path for the file you wish to use
#filepath = "file:/home/training/training_materials/data/boston50.txt"
#bostonRDD = sc.textFile(filepath)
#bostonRDD.cache
# End step 1b

# All code below is used for both AWS and your local machine.
# Replace '\t' with ',' to create a comma separated dataset
new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

# Split the data by
def parsePoint(line):
    values = [float(x) for x in line.split(",")]
    return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

# Train a naive bayes model
model_boston = NaiveBayes.train(training_boston, 1.0)

# Make prediction and text accuracy
predictionAndLabel_boston = test_boston.map(lambda p: (model_boston.predict(p.features), p.label))
accuracy_boston = 1.0 * predictionAndLabel_boston.filter(lambda(x,v): x == v).count() /test_boston.count()

# Print Model Accuracy
print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished NaiveBayes Model-----')
print('..........................................................................................................')