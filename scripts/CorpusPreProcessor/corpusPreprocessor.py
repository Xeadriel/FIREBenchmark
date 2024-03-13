import csv
from nltk import word_tokenize
import string

def prepareMSRData(path):
	"""
		Makes the sentences lower case, strips leading and trailing whitespace and removes punctuation.
		Returns the sentence pairs with their labels. 
		
		parameters:
			String : path: path of MSR data
		
		returns:
			List : sentencePairs: list of tuples that contain sentence pairs
			List : labels: integer values 1 or 0 that indicate whether the sentences are similar
	"""
	labels = []
	sentencePairs = []

	with open(path, newline='', encoding="utf8") as csvfile:
		# there is a line that's broken by quotation marks. ß is never used in English so there is that
		reader = csv.reader(csvfile, delimiter='\t', quotechar='ß')
		next(reader) # skip header
		
		for row in reader:
			labels.append(row[0])
			
			sentence1 = row[3].strip().lower()
			sentence2 = row[4].strip().lower()
			
			sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
			sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))
			
			sentence1 = " ".join(word_tokenize(sentence1))
			sentence2 = " ".join(word_tokenize(sentence2))
			
			sentencePairs.append((sentence1, sentence2))
	
	return sentencePairs, labels

