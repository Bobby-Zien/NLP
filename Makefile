def:
	javac -cp ./maxent/maxent-3.0.0.jar:./maxent/trove.jar ./maxent/*.java

run:
	cd maxent && \
	java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain ../feature_files/training.feature model.chunk && \
	java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtag ../feature_files/test.feature model.chunk ../outputs/output.txt && \
	cd ../

score:
	python3 ./src/score.chunk.py ./answerkeys/anskey_test.txt ./outputs/output.txt

gen_path:
	python3 ./src/path_feature.py

gen_stem:
	python3 ./src/stem_feature.py
	
