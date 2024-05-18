import csv
from nltk import word_tokenize
import string

def prepareMRPCData(path):
	"""
		Makes the sentences lower case, strips leading and trailing whitespace and removes punctuation.
		Returns the sentence pairs with their labels (0 or 1). 
		
		parameters:
			String : path: path of MRPC data
		
		returns:
			List : sentencePairs: 2 lists that contain sentence pairs
			List : labels: integer values 1 or 0 that indicate whether the sentences are similar
	"""
	labels = []
	sentencePairs = [[], []]

	with open(path, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			labels.append(int(row[0]))
			
			sentence1 = row[3].strip().lower()
			sentence2 = row[4].strip().lower()
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			sentence2 = " ".join(word_tokenize(sentence2))
			
			sentencePairs[0].append(sentence1)
			sentencePairs[1].append(sentence2)
	
	return sentencePairs, labels

def prepareSSTData(splitPath, sentencesPath, dictionaryPath, labelsPath):
	"""
		Makes the sentences lower case, strips leading and trailing whitespace and removes punctuation.
		The sentiment "very negative", "negative" etc. is used as the second sentence.
		Returns the sentence pairs with their labels (0 or 1, to be interpreted as false or true).
		
		parameters:
			String : splitPath: file with train, test, dev split information,
			String : sentencesPath: file with all sentences,
			String : dictionaryPath: file with phrases,
			String : labelsPath: file with labels assigned to phrases,
		
		returns:
			List : trainPairs: 2 lists that contain sentence pairs
			List : trainLabels: integer values 1 or 0 that indicate whether the sentences fits the supplied sentiment
			List : devPairs: 2 lists that contain sentence pairs
			List : devLabels: integer values 1 or 0 that indicate whether the sentences fits the supplied sentiment
			List : testPairs: 2 lists that contain sentence pairs
			List : testLabels: integer values 1 or 0 that indicate whether the sentence fits the supplied sentiment

	"""
	labels = {}
	
	sentences = {}
	phrases = {}

	trainIDs = []
	devIDs = []
	testIDs = []

	# split ids: train = 1, dev = 2, test = 3
	id2splitId = {}

	trainPairs = [[], []]
	devPairs = [[], []]
	testPairs = [[], []]

	trainLabels = []
	devLabels = []
	testLabels = []

	with open(splitPath, newline='', encoding='utf8') as splitFile:
		# remove header
		splitFile.readline()
		for line in splitFile:
			id2splitId[int(line[:-3])] = int(line[-2])

			if int(line[-2]) == 1:
				trainIDs.append(int(line[:-3]))
			elif int(line[-2]) == 2:
				testIDs.append(int(line[:-3]))
			elif int(line[-2]) == 3:
				devIDs.append(int(line[:-3]))
			
	with open(sentencesPath, newline='', encoding='utf8') as sentenceFile:
		# remove header
		sentenceFile.readline()
		for line in sentenceFile:
			line = line.replace("-LRB-", "(")
			line = line.replace("-RRB-", ")")
			line = line.replace("Ã¢", "â")
			line = line.replace("Ã ", "à")
			line = line.replace("Ã¡", "á")
			line = line.replace("Ã®", "î")
			line = line.replace("Ã­", "í")
			line = line.replace("Ã¬", "ì")
			line = line.replace("Ãª", "ê")
			line = line.replace("Ã©", "é")
			line = line.replace("Ã¨", "è")
			line = line.replace("Ã´", "ô")
			line = line.replace("Ã³", "ó")
			line = line.replace("Ã²", "ò")
			line = line.replace("Ã»", "û")
			line = line.replace("Ãº", "ú")
			line = line.replace("Ã¹", "ù")
			line = line.replace("Ã¤", "ä")
			line = line.replace("Ã¼", "ü")
			line = line.replace("Ã¶", "ö")
			line = line.replace("Ã±", "ñ")
			line = line.replace("Ã¯", "ï")
			line = line.replace("Ã¦", "æ")
			line = line.replace('Â', '')
			line = line.replace('Ã£', 'ã')
			line = line.replace('Ã§', 'ç')

			whereTab = line.find("\t")
			sentences[int(line[:whereTab])] = line[whereTab+1:-1] #remove \n
	
	with open(dictionaryPath, newline='', encoding='utf8') as dictionaryFile:
		for line in dictionaryFile:
			whereVLine = line.find("|")
			phrases[line[:whereVLine]] = int(line[whereVLine+1:-1]) #remove \n
	
	with open(labelsPath, newline='', encoding='utf8') as labelsFile:
		# remove header
		labelsFile.readline()
		for line in labelsFile:
			whereVLine = line.find("|")
			labels[int(line[:whereVLine])] = float(line[whereVLine+1:-1]) #remove \n
	
	for id in trainIDs:
		sentence = sentences[id]
		label = labels[phrases[sentence]]
		sentiment = ""
		if label <= 0.2:
			sentiment = "very negative"
		elif label <= 0.4:
			sentiment = "negative"
		elif label <= 0.6:
			sentiment = "neutral"
		elif label <= 0.8:
			sentiment = "positive"
		elif label <= 1:
			sentiment = "very positive"

		sentence = sentence.strip().lower()
		
		sentence = sentence.translate(str.maketrans('', '', string.punctuation))
		
		sentence = " ".join(word_tokenize(sentence))
		
		trainPairs[0].append(sentence)
		trainPairs[1].append(sentiment)
		trainLabels.append(label)
	
	for id in devIDs:
		sentence = sentences[id]
		label = labels[phrases[sentence]]
		sentiment = ""
		if label <= 0.2:
			sentiment = "very negative"
		elif label <= 0.4:
			sentiment = "negative"
		elif label <= 0.6:
			sentiment = "neutral"
		elif label <= 0.8:
			sentiment = "positive"
		elif label <= 1:
			sentiment = "very positive"
		
		sentence = sentence.strip().lower()
		
		sentence = sentence.translate(str.maketrans('', '', string.punctuation))
		
		sentence = " ".join(word_tokenize(sentence))
		
		devPairs[0].append(sentence)
		devPairs[1].append(sentiment)
		devLabels.append(label)
	
	for id in testIDs:
		sentence = sentences[id]
		label = labels[phrases[sentence]]
		sentiment = ""
		if label <= 0.2:
			sentiment = "very negative"
		elif label <= 0.4:
			sentiment = "negative"
		elif label <= 0.6:
			sentiment = "neutral"
		elif label <= 0.8:
			sentiment = "positive"
		elif label <= 1:
			sentiment = "very positive"
		
		sentence = sentence.strip().lower()
		
		sentence = sentence.translate(str.maketrans('', '', string.punctuation))
		
		sentence = " ".join(word_tokenize(sentence))

		testPairs[0].append(sentence)
		testPairs[1].append(sentiment)
		testLabels.append(label)
	
	return trainPairs, trainLabels, devPairs, devLabels, testPairs, testLabels

def prepareSSTGlueData(trainPath, devPath, testPath):
	"""
		Makes the sentences lower case, strips leading and trailing whitespace and removes punctuation.
		The sentiment "positive" is used as the second sentence on every sentence.
		Returns the sentence pairs with their labels (0 or 1; interpret as: not positive or positive).
		
		parameters:
			String : splitPath: file with train, test, dev split information,
			String : sentencesPath: file with all sentences,
			String : dictionaryPath: file with phrases,
			String : labelsPath: file with labels assigned to phrases,
		
		returns:
			List : trainPairs: 2 lists that contain sentence pairs
			List : trainLabels: integer values 1 or 0 that indicate whether the sentence is positive
			List : devPairs: 2 lists that contain sentence pairs
			List : devLabels: integer values 1 or 0 that indicate whether the sentence is positive
			List : testPairs: 2 lists that contain sentence pairs
			List : testIndices: integer indices of pairs

	"""
	trainPairs = [[], []]
	devPairs = [[], []]
	testPairs = [[], []]

	trainLabels = []
	devLabels = []
	testIndices = []

	with open(trainPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			trainLabels.append(int(row[1]))
			
			sentence1 = row[0].strip().lower()
			sentence2 = "positive"
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			
			trainPairs[0].append(sentence1)
			trainPairs[1].append(sentence2)
	
	with open(devPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			devLabels.append(int(row[1]))
			
			sentence1 = row[0].strip().lower()
			sentence2 = "positive"
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			
			devPairs[0].append(sentence1)
			devPairs[1].append(sentence2)

	with open(testPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			testIndices.append(int(row[0]))
			
			sentence1 = row[1].strip().lower()
			sentence2 = "positive"
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			
			testPairs[0].append(sentence1)
			testPairs[1].append(sentence2)

	return trainPairs, trainLabels, devPairs, devLabels, testPairs, testIndices

def prepareRTEGlueData(trainPath, devPath, testPath):
	"""
		Makes the sentences lower case, strips leading and trailing whitespace and removes punctuation.
		Returns the sentence pairs with their labels (0 == not_entailment, 1 = entailment).
		
		parameters:
			String : splitPath: file with train, test, dev split information,
			String : sentencesPath: file with all sentences,
			String : dictionaryPath: file with phrases,
			String : labelsPath: file with labels assigned to phrases,
		
		returns:
			List : trainPairs: 2 lists that contain sentence pairs
			List : trainLabels: integer values 1 or 0 that indicate whether the sentences are entailed
			List : devPairs: 2 lists that contain sentence pairs
			List : devLabels: integer values 1 or 0 that indicate whether the sentences are entailed
			List : testPairs: 2 lists that contain sentence pairs
			List : testIndices: integer indices of pairs

	"""
	
	trainPairs = [[], []]
	devPairs = [[], []]
	testPairs = [[], []]

	trainLabels = []
	devLabels = []
	testIndices = []

	with open(trainPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			trainLabels.append(int(row[3] == "entailment"))
			
			sentence1 = row[1].strip().lower()
			sentence2 = row[2].strip().lower()
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			sentence2 = " ".join(word_tokenize(sentence2))
			
			trainPairs[0].append(sentence1)
			trainPairs[1].append(sentence2)
	
	with open(devPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			devLabels.append(int(row[3] == "entailment"))
			
			sentence1 = row[1].strip().lower()
			sentence2 = row[2].strip().lower()
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			sentence2 = " ".join(word_tokenize(sentence2))
			
			devPairs[0].append(sentence1)
			devPairs[1].append(sentence2)

	with open(testPath, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			testIndices.append(int(row[0]))
			
			sentence1 = row[1].strip().lower()
			sentence2 = row[2].strip().lower()
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			sentence2 = " ".join(word_tokenize(sentence2))
			
			testPairs[0].append(sentence1)
			testPairs[1].append(sentence2)

	return trainPairs, trainLabels, devPairs, devLabels, testPairs, testIndices