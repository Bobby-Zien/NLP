# NLP Group 4 - Maximizing the Score for Detecting ARG1 on Partitives

## Team
Zien Yang, zy2236, N17995064  
Jingwei Ye, jy3555, N10236604  
Zeyu Chen, zc2078, N10456612  

## feature.py
contains all the feature selection methods

## machine_learning.py
contains all the machine learning algorithms

## score.py

## How to run java maxent?
  1. javac -cp maxent-3.0.0.jar:trove.jar *.java ### compiling
  2. java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk ### creating the model of the training data
  3. java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk ### creating the system output
