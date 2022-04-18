#!/usr/bin/python
#
# scorer for NLP class Spring 2016
# ver.1.0
#
# score a key file against a response file
# both should consist of lines of the form:   token \t tag
# sentences are separated by empty lines
#
import sys
import os

def score (keyFileName, responseFileName):
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()
	if len(key) != len(response):
		print ("length mismatch between key and submitted file")
		return 0
	correct = 0
	incorrect = 0
	keyGroupCount = 0
	responseGroupCount = 0
	correctGroupCount = 0
	for i in range(len(key) - 1):
		key[i] = key[i].rstrip(os.linesep)
		response[i] = response[i].rstrip(os.linesep)
		if key[i] == "":
			if response[i] == "":
				continue
			else:
				print("key string is: " + str(key[i]) + " response string is: " + str(response[i]))
				print("sentence break expected at line " + str(i))
				return 0
		keyFields = key[i].split('\t')
		if len(keyFields) != 2:
			print("format error in key at line " + str(i) + ":" + key[i])
			return 0
		keyToken = keyFields[0].strip()
		keyTag = keyFields[1]
		keyTag = keyTag.rstrip(os.linesep)
		responseFields = response[i].split('\t')
		responseToken = ''
		responseTag = ''
		if len(responseFields) == 2:
			responseToken = responseFields[0].strip()
			responseTag = responseFields[1]
			responseTag = responseTag.rstrip(os.linesep)

		if (responseToken != keyToken) or (responseTag != keyTag):
			print("Token/Tag mismatch at line " + str(i))
			print("Key string is: " + str(key[i]) + "   Response string is: " + str(response[i]))
			incorrect += 1
		else:
			correct = correct + 1
			if keyTag=='ARG1':
				correctGroupCount+=1

		if keyTag=='ARG1':
			keyGroupCount+=1

		if responseTag=='ARG1':
			responseGroupCount+=1

	print(correct, "out of", str(correct + incorrect) + " tags correct")
	accuracy = 100.0 * correct / (correct + incorrect)
	print("  accuracy: %5.2f" % accuracy)
	print(keyGroupCount, "groups in key")
	print(responseGroupCount, "groups in response")
	print(correctGroupCount, "correct groups")
	precision = 100.0 * correctGroupCount / responseGroupCount
	recall = 100.0 * correctGroupCount / keyGroupCount
	F = (2 * precision * recall / (precision + recall)) * .10
	print("  precision: %5.2f" % precision)
	print("  recall:    %5.2f" % recall)
	print("  F1:        %5.2f" % F)
	rndf = int(round(F, 0))
	print("  rounded to: " + str(rndf))


def test(systemout,ans):
	sysn=len(systemout)
	ansn=len(ans)
	count=0
	for sysv in systemout:
		if sysv in ans:
			count+=1

	precision=100.0 * count/sysn
	recall=100.0 * count/ansn
	F = (2 * precision * recall / (precision + recall)) * .10
	print(precision,recall,F)

def generateAns(ansfileName):
	ansfile = open("input_files/"+ansfileName, 'r')
	anskeyfile= open("answerkeys/anskey_dev.txt", 'w')

	Lines = ansfile.readlines()
	for line in Lines:
		if len(line.split()) > 1:
			if len(line.split()) >= 6:
				word=line.split()[0]
				rlabel=line.split()[5]
				# word, pos, tag, token, sentN, rlabel = line.split()
			else:
				rlabel = "NONE"
				word, pos, tag, token, sentN = line.split()

			anskeyfile.write(word + '\t' + rlabel+'\n')
		else:
			anskeyfile.write('\n')


	ansfile.close()
	anskeyfile.close()





def main(args):
	key_file = args[1]
	response_file = args[2]
	# generateAns("part-dev")
	# key_file= "answerkeys/anskey.txt"
	# response_file="output.txt"
	score(key_file,response_file)

if __name__ == '__main__': sys.exit(main(sys.argv))

## python score.chunk.py WSJ_24.pos-chunk response.chunk