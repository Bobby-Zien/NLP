# NLP Group 4 - Maximizing the Score for Detecting ARG1 on Partitives

## Team
Zien Yang, zy2236, N17995064  
Jingwei Ye, jy3555, N10236604  
Zeyu Chen, zc2078, N10456612  

## How to run our program?
1. make
2. make <gen_path, gen_stem, gen_vec> -- choose one of the three
3. make <run_path, run_stem, run_vec> -- choose one of the three
4. make score

## feature.py
contains all the feature selection methods

## Machine_Learning.py
contains all the machine learning algorithms <br />
run "python Machine_Learning.py [training_file_path] [test_file_path]" <br />
For example, "python Machine_Learning.py ../feature_files/training_path.feature  ../feature_files/test_path.feature" <br />
Outputs predicted by models will be saved as files in /outputs folder <br />

## score.py
run "python score.chunk.py [answerkeysfile] [responsefile]"
Accuracy contains NONE values
Precision contains only ARG1 values

## How to run java maxent?
  1. javac -cp maxent-3.0.0.jar:trove.jar *.java ### compiling
  2. java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk ### creating the model of the training data
  3. java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk ### creating the system output


## Statistics
# Five-Word-Window with Stem <br />
  MaxEnt: 49% Precision, 25% Recall, 0.33 F-measure <br />
  Naive Bayes: 6% Precision, 35% Recall, 0.1 F-measure <br />
  SVM: 4% Precision, 90% Recall, 0.07 F-measure <br />
  Decision Tree: 33% Precision, 34% Recall, 0.34 F-measure <br />
  Random Forest: 64% Precision, 23% Recall, 0.34 F-measure <br />

# Path from pred <br />
  MaxEnt: 66% Precision, 41% Recall, 0.5 F-measure  <br />
  Naive Bayes: 7% Precision, 27% Recall, 0.11 F-measure  <br />
  SVM: 9% Precision, 2% Recall, 0.03 F-measure  <br />
  Decision Tree: 55% Precision, 58% Recall, 0.56 F-measure  <br />
  Random Forest: 90% Precision, 54% Recall, 0.67 F-measure  <br />
 
 # Path from pred + Distance  <br />
   MaxEnt: 74% Precision, 53% Recall, 0.61 F-measure  <br />
   Naive Bayes: 7% Precision, 27% Recall, 0.11 F-measure  <br />
   Decision Tree: 55% Precision, 60% Recall, 0.57 F-measure  <br />
   Random Forest: 90% Precision, 54% Recall, 0.68 F-measure  <br />
  
 # Vector based  <br />
   MaxEnt: 49% Precision, 28% Recall, 0.35 F-measure  <br />
   Naive Bayes: 7% Precision, 8% Recall, 0.08 F-measure  <br />
   Decision Tree: 33% Precision, 31% Recall, 0.32 F-measure  <br />
   Random Forest: 67% Precision, 25% Recall, 0.36 F-measure  <br />
 
 







  
