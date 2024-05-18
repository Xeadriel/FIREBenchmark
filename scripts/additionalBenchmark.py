from typing import List, Mapping
import argparse
import os
from collections.abc import Callable
from collections import defaultdict, Counter
from corpusit import Vocab
import nltk
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import multiprocessing
import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score

from firelang.models import FireWord, FireTensor
from firelang.utils.log import logger
from firelang.utils.timer import Timer, elapsed
from scripts.sentsim import sentsim_as_weighted_wordsim_cuda
from scripts.corpusPreprocessor import *
from scripts.benchmark import sentence_simmat
import math
import csv

@torch.no_grad()
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--checkpointsMRPC",
		nargs="+",
		default=[
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k1_polysemy",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k10",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l8_k20",
		],
	)

	parser.add_argument(
		"--checkpointsSST",
		nargs="+",
		default=[
			# "checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k1_polysemy",
			# "checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k10",
			# "checkpoints/v1.1/wacky_mlplanardiv_d2_l8_k20",
		],
	)

	parser.add_argument(
		"--checkpointsSSTGlue",
		nargs="+",
		default=[
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k1_polysemy",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k10",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l8_k20",
		],
	)

	parser.add_argument(
		"--checkpointsRTE",
		nargs="+",
		default=[
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k1_polysemy",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k10",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l8_k20",
		],
	)

	args = parser.parse_args()

	device = "cuda"
	torch.set_default_device(device)

	sifA = 0.001
	print("--------------------------------------------------------------------------------------------------------------------------------")
	print("Benchmarks")
	print("--------------------------------------------------------------------------------------------------------------------------------")
	print("\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\tThe Stanford Sentiment Treebank")
	print("\t--------------------------------------------------------------------------------------------------------------------------------")
	# trainPairsSST, trainLabelsSST, devPairsSST, devLabelsSST, testPairsSST, testLabelsSST = prepareSSTData('scripts/tasks/SST/datasetSplit.txt', 
	# 				'scripts/tasks/SST/datasetSentences.txt', 'scripts/tasks/SST/dictionary.txt', 'scripts/tasks/SST/sentiment_labels.txt')
	# for checkpoint in args.checkpointsSST:	
	# 	model = FireWord.from_pretrained(checkpoint).to(device)
	# 	print(f"checkpoint: {checkpoint}")

	# 	accuracy = benchmarkSST(model, testPairsSST, testLabelsSST, devPairsSST, devLabelsSST, sifA)
		
	# 	print(f"accuracy: {accuracy}")
	

	print("\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\tComputing Predictions for GLUE")
	print("\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\t\tMicrosoft Research Paraphrase Corpus")
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	testPairsMRPC, testLabelsMRPC = prepareMRPCData('scripts/tasks/MRPC/msr_paraphrase_test.txt')
	for checkpoint in args.checkpointsMRPC:	
		model = FireWord.from_pretrained(checkpoint).to(device)
		print(f"\t\tcheckpoint: {checkpoint}")

		predsMedianMRPC, predsThresholdMRPC, predsF1ThresholdMRPC, accuracy, f1 = benchmarkMRPC(model, testPairsMRPC, testLabelsMRPC, sifA)
		
		print(f"\t\taccuracy: {accuracy}\n\t\tf1: {f1}\n")

		with open(f'scripts/taskResults/MRPC/median/MRPC-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(range(len(predsMedianMRPC)), predsMedianMRPC):
				writer.writerow([index, pred])
		
		with open(f'scripts/taskResults/MRPC/threshold/MRPC-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(range(len(predsThresholdMRPC)), predsThresholdMRPC):
				writer.writerow([index, pred])
		
		with open(f'scripts/taskResults/MRPC/f1Threshold/MRPC-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(range(len(predsF1ThresholdMRPC)), predsF1ThresholdMRPC):
				writer.writerow([index, pred])

	
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\t\tThe Stanford Sentiment Treebank (GLUE version)")
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	trainPairsSSTGlue, trainLabelsSSTGlue, devPairsSSTGlue, devLabelsSSTGlue, testPairsSSTGlue, testIndicesSSTGlue = prepareSSTGlueData('scripts/tasks/SSTGLUE/train.tsv', 'scripts/tasks/SSTGLUE/dev.tsv', 'scripts/tasks/SSTGLUE/test.tsv')
	for checkpoint in args.checkpointsSSTGlue:
		print(f"\t\tcheckpoint: {checkpoint}")
		model = FireWord.from_pretrained(checkpoint).to(device)
		predsMedianSSTGlue, predsThresholdSSTGlue = predictSSTGlue(model, testPairsSSTGlue, devPairsSSTGlue, devLabelsSSTGlue, sifA)

		with open(f'scripts/taskResults/SSTGLUE/median/SST-2-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(range(len(predsMedianSSTGlue)), predsMedianSSTGlue):
				writer.writerow([index, pred])
		
		with open(f'scripts/taskResults/SSTGLUE/threshold/SST-2-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(testIndicesSSTGlue, predsThresholdSSTGlue):
				writer.writerow([index, pred])
	
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	print("\t\tRecognizing Textual Entailment")
	print("\t\t--------------------------------------------------------------------------------------------------------------------------------")
	trainPairsRTE, trainLabelsRTE, devPairsRTE, devLabelsRTE, testPairsRTE, testIndicesRTE = prepareRTEGlueData('scripts/tasks/RTE/train.tsv', 'scripts/tasks/RTE/dev.tsv', 'scripts/tasks/RTE/test.tsv')
	for checkpoint in args.checkpointsRTE:
		print(f"\t\tcheckpoint: {checkpoint}")
		model = FireWord.from_pretrained(checkpoint).to(device)
		predsMedianRTE, predsThresholdRTE = predictRTE(model, testPairsRTE, devPairsRTE, devLabelsRTE, sifA)

		with open(f'scripts/taskResults/RTE/median/RTE-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
				writer.writerow(["index", "prediction"])
				for index, pred in zip(range(len(predsMedianRTE)), predsMedianRTE):
					writer.writerow([index, pred])
			
		with open(f'scripts/taskResults/RTE/threshold/RTE-{checkpoint[17:]}.tsv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t', quotechar='ß')
			writer.writerow(["index", "prediction"])
			for index, pred in zip(testIndicesRTE, predsThresholdRTE):
				writer.writerow([index, pred])

	
@torch.no_grad()
@Timer(elapsed, "sentsim")
def benchmarkMRPC(
	model: FireWord,
	pairs,
	labels,
	sif_alpha=1e-3,
):
	vocab: Vocab = model.vocab

	counts = pd.Series(vocab.counts_dict())
	probs = counts / counts.sum()
	sif_weights: Mapping[str, float] = {
		w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
	}

	sents1 = pairs[0]
	sents2 = pairs[1]
	allsents = sents1 + sents2
	allsents = [
		[w for w in sent if w in sif_weights and w != vocab.unk]
		for sent in allsents
	]

	""" similarity """
	with Timer(elapsed, "similarity", sync_cuda=True):
		simmat = sentence_simmat(model, allsents, sif_weights)

	""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
	halved bc (9)"""
	with Timer(elapsed, "regularization"):
		diag = np.diag(simmat)
		simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

	""" smoothing by standardization """
	with Timer(elapsed, "smooth"):
		mean1 = np.mean(simmat, axis=1, keepdims=True)
		std1 = np.std(simmat, axis=1, keepdims=True)
		mean0 = np.mean(simmat, axis=0, keepdims=True)
		std0 = np.std(simmat, axis=0, keepdims=True)
		simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

		N = len(pairs[0])
		preds = [simmat[i, i + N] for i in range(N)]
		preds = np.exp(preds)
		preds = np.array(preds)

	medianThreshhold = np.median(preds)
	medianScore = sum([int(preds[i] >= medianThreshhold) == (labels[i]) for i in range(len(preds))]) / len(preds)
	truePosCount = sum([int(preds[i] >= medianThreshhold) == 1 and labels[i] == 1 for i in range(len(preds))])
	falsePosCount = sum([int(preds[i] >= medianThreshhold) == 1 and labels[i] == 0 for i in range(len(preds))])
	falseNegCount = sum([int(preds[i] >= medianThreshhold) == 0 and labels[i] == 1 for i in range(len(preds))])

	medianf1Score = truePosCount / (truePosCount + 0.5 * (falsePosCount + falseNegCount))
	# print(f"\t\tmedianThreshhold: {medianThreshhold} \n\t\tmedianAccuracy: {medianScore}\n\t\tmedianF1: {medianf1Score}")

	bestThreshold = 0
	bestF1Threshold = 0
	bestAccuracy = 0
	bestF1 = 0

	low = min(preds)
	high = max(preds)
	steps = math.ceil((high - low) / 2)*100

	for threshold in np.linspace(low, high, steps):
		truePosCount = sum([int(preds[i] >= threshold) == 1 and labels[i] == 1 for i in range(len(preds))])
		falsePosCount = sum([int(preds[i] >= threshold) == 1 and labels[i] == 0 for i in range(len(preds))])
		falseNegCount = sum([int(preds[i] >= threshold) == 0 and labels[i] == 1 for i in range(len(preds))])

		f1Score = truePosCount / (truePosCount + 0.5 * (falsePosCount + falseNegCount))
		if bestF1 < f1Score: 
			bestF1Threshold = threshold
			bestF1 = f1Score

		accuracy =  sum([int(preds[i] >= threshold) == (labels[i]) for i in range(len(preds))]) / len(preds)
		if bestAccuracy < accuracy: 
			bestThreshold = threshold
			bestAccuracy = accuracy
	
	# print(f"\t\taccuracy threshold = {bestThreshold}")
	# print(f"\t\tF1 threshold = {bestF1Threshold}")

	predsMedian = [int(pred >= medianThreshhold) for pred in preds]
	predsThreshold = [int(pred >= bestThreshold) for pred in preds]
	predsF1Threshold = [int(pred >= bestF1Threshold) for pred in preds]
	return predsMedian, predsThreshold, predsF1Threshold, bestAccuracy, bestF1

@torch.no_grad()
@Timer(elapsed, "sentsim")
def benchmarkSST(
	model: FireWord,
	pairs,
	labels,
	devPairs,
	devLabels,
	sif_alpha=1e-3,
):
	vocab: Vocab = model.vocab

	N = len(pairs[0])

	def computeDevThresholds():
		
		counts = pd.Series(vocab.counts_dict())
		probs = counts / counts.sum()
		sif_weights: Mapping[str, float] = {
			w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
		}

		sents1 = pairs[0]
		sents2 = ["very negative", "negative", "neutral", "positive", "very positive"]
		allsents = sents1 + sents2
		allsents = [
			[w for w in sent if w in sif_weights and w != vocab.unk]
			for sent in allsents
		]

		""" similarity """
		with Timer(elapsed, "similarity", sync_cuda=True):
			simmat = sentence_simmat(model, allsents, sif_weights)

		""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
		halved bc (9)"""
		with Timer(elapsed, "regularization"):
			diag = np.diag(simmat)
			simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

		N = len(pairs[0])
		with Timer(elapsed, "smooth"):
			mean1 = np.mean(simmat, axis=1, keepdims=True)
			std1 = np.std(simmat, axis=1, keepdims=True)
			mean0 = np.mean(simmat, axis=0, keepdims=True)
			std0 = np.std(simmat, axis=0, keepdims=True)
			simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

			preds = np.exp(preds)
			preds = np.array(preds)

		predsVN = np.array([simmat[i, N] for i in range(N)])
		predsN = np.array([simmat[i, N + 1] for i in range(N)])
		predsNeut = np.array([simmat[i, N + 2] for i in range(N)])
		predsP = np.array([simmat[i, N + 3] for i in range(N)])
		predsVP = np.array([simmat[i, N + 4] for i in range(N)])

		bestThresholdVeryNegative, bestThresholdNegative, bestThresholdNeutral, bestThresholdPositive, bestThresholdVeryPositive = (0, 0, 0, 0, 0)
		bestAccuracy = 0
		
		lowVN = min(predsVN)
		highVN = max(predsVN)
		stepsVN = math.ceil((highVN - lowVN) / 2)*100
		lowN = min(predsN)
		highN = max(predsN)
		stepsN = math.ceil((highN - lowN) / 2)*100
		lowNeut = min(predsNeut)
		highNeut = max(predsNeut)
		stepsNeut = math.ceil((highNeut - lowNeut) / 2)*100
		lowP = min(predsP)
		highP = max(predsP)
		stepsP = math.ceil((highP - lowP) / 2)*100
		lowVP = min(predsVP)
		highVP = max(predsVP)
		stepsVP = math.ceil((highVP - lowVP) / 2)*100

		for thresholdVN in np.linspace(lowVN, highVN, stepsVN):
			for thresholdN in np.linspace(lowN, highN, stepsN):
				for thresholdNeut in np.linspace(lowNeut, highNeut, stepsNeut):
					for thresholdP in np.linspace(lowP, highP, stepsP):
						for thresholdVP in np.linspace(lowVP, highVP, stepsVP):
							accuracy =  sum([int( 
								(((predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
									and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] >= 0 and  devLabels[i] <= 0.2)) or # very negative
				   
								(((not predsVN[i] >= thresholdVN) and (predsN[i] >= thresholdN)  and not(predsNeut[i] >= thresholdNeut)
			   						and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.2 and  devLabels[i] <= 0.4)) or # negative
				   
								(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and (predsNeut[i] >= thresholdNeut)
									and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.4 and  devLabels[i] <= 0.6)) or # neutral

								(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
									and (predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.6 and  devLabels[i] <= 0.8)) or # positive

								(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
			   						and not(predsP[i] >= thresholdP) and (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.8 and  devLabels[i] <= 1)) #very positve
							 					) for i in range(N)]) / N
							if bestAccuracy < accuracy: 
								bestThresholdVeryNegative = thresholdVN
								bestThresholdNegative = thresholdN
								bestThresholdNeutral = thresholdNeut
								bestThresholdPositive = thresholdP
								bestThresholdVeryPositive = thresholdVP
								bestAccuracy = accuracy
		

		return bestThresholdVeryNegative, bestThresholdNegative, bestThresholdNeutral, bestThresholdPositive, bestThresholdVeryPositive

	thresholdVN, thresholdN, thresholdNeut, thresholdP, thresholdVP = computeDevThresholds()

	counts = pd.Series(vocab.counts_dict())
	probs = counts / counts.sum()
	sif_weights: Mapping[str, float] = {
		w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
	}

	sents1 = pairs[0]
	sents2 = ["very negative", "negative", "neutral", "positive", "very positive"]
	allsents = sents1 + sents2
	allsents = [
		[w for w in sent if w in sif_weights and w != vocab.unk]
		for sent in allsents
	]

	""" similarity """
	with Timer(elapsed, "similarity", sync_cuda=True):
		simmat = sentence_simmat(model, allsents, sif_weights)

	""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
	halved bc (9)"""
	with Timer(elapsed, "regularization"):
		diag = np.diag(simmat)
		simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

	N = len(pairs[0])
	with Timer(elapsed, "smooth"):
		mean1 = np.mean(simmat, axis=1, keepdims=True)
		std1 = np.std(simmat, axis=1, keepdims=True)
		mean0 = np.mean(simmat, axis=0, keepdims=True)
		std0 = np.std(simmat, axis=0, keepdims=True)
		simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

		preds = np.exp(preds)
		preds = np.array(preds)

	N = len(pairs[0])
	predsVN = np.array([simmat[i, N] for i in range(N)])
	predsN = np.array([simmat[i, N + 1] for i in range(N)])
	predsNeut = np.array([simmat[i, N + 2] for i in range(N)])
	predsP = np.array([simmat[i, N + 3] for i in range(N)])
	predsVP = np.array([simmat[i, N + 4] for i in range(N)])
	
	
	accuracy =  sum([int( 
		(((predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
			and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] >= 0 and  devLabels[i] <= 0.2)) or # very negative

		(((not predsVN[i] >= thresholdVN) and (predsN[i] >= thresholdN)  and not(predsNeut[i] >= thresholdNeut)
			and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.2 and  devLabels[i] <= 0.4)) or # negative

		(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and (predsNeut[i] >= thresholdNeut)
			and not(predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.4 and  devLabels[i] <= 0.6)) or # neutral

		(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
			and (predsP[i] >= thresholdP) and not (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.6 and  devLabels[i] <= 0.8)) or # positive

		(((not predsVN[i] >= thresholdVN) and not(predsN[i] >= thresholdN) and not(predsNeut[i] >= thresholdNeut)
			and not(predsP[i] >= thresholdP) and (predsVP[i] >= thresholdVP)) and (devLabels[i] > 0.8 and  devLabels[i] <= 1)) #very positve
						) for i in range(len(N))]) / len(N)
	# print(
	# 	f"threshhold very negative:\t{thresholdVN}\n"
	# 	f"threshhold negative:\t\t{thresholdN}\n"
	# 	f"threshhold neutral:\t\t {thresholdNeut}\n"
	# 	f"threshhold positive:\t\t {thresholdP}\n"
	# 	f"threshhold very positive:\t{thresholdVP}")

	return accuracy


@torch.no_grad()
@Timer(elapsed, "sentsim")
def predictSSTGlue(
	model: FireWord,
	pairs,
	devPairs,
	devLabels,
	sif_alpha=1e-3,
):
	vocab: Vocab = model.vocab

	def computeThresholdFromDevData():
		counts = pd.Series(vocab.counts_dict())
		probs = counts / counts.sum()
		sif_weights: Mapping[str, float] = {
			w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
		}

		sents1 = devPairs[0]
		sents2 = devPairs[1]
		allsents = sents1 + sents2
		allsents = [
			[w for w in sent if w in sif_weights and w != vocab.unk]
			for sent in allsents
		]

		""" similarity """
		with Timer(elapsed, "similarity", sync_cuda=True):
			simmat = sentence_simmat(model, allsents, sif_weights)

		""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
		halved bc (9)"""
		with Timer(elapsed, "regularization"):
			diag = np.diag(simmat)
			simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

		with Timer(elapsed, "smooth"):
			mean1 = np.mean(simmat, axis=1, keepdims=True)
			std1 = np.std(simmat, axis=1, keepdims=True)
			mean0 = np.mean(simmat, axis=0, keepdims=True)
			std0 = np.std(simmat, axis=0, keepdims=True)
			simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

			N = len(devPairs[0])
			preds = [simmat[i, i + N] for i in range(N)]
			preds = np.exp(preds)
			preds = np.array(preds)

		bestThreshold = 0
		bestAccuracy = 0
		low = min(preds)
		high = max(preds)
		steps = math.ceil((high - low) / 2)*100

		for threshold in np.linspace(low, high, steps):
			accuracy =  sum([int(preds[i] >= threshold) == (devLabels[i]) for i in range(len(preds))]) / len(preds)
			if bestAccuracy < accuracy: 
				bestThreshold = threshold
				bestAccuracy = accuracy
		

		return bestThreshold

	devThreshold = computeThresholdFromDevData()

	counts = pd.Series(vocab.counts_dict())
	probs = counts / counts.sum()
	sif_weights: Mapping[str, float] = {
		w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
	}

	sents1 = pairs[0]
	sents2 = pairs[1]
	allsents = sents1 + sents2
	allsents = [
		[w for w in sent if w in sif_weights and w != vocab.unk]
		for sent in allsents
	]

	""" similarity """
	with Timer(elapsed, "similarity", sync_cuda=True):
		simmat = sentence_simmat(model, allsents, sif_weights)

	""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
	halved bc (9)"""
	with Timer(elapsed, "regularization"):
		diag = np.diag(simmat)
		simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

	with Timer(elapsed, "smooth"):
		mean1 = np.mean(simmat, axis=1, keepdims=True)
		std1 = np.std(simmat, axis=1, keepdims=True)
		mean0 = np.mean(simmat, axis=0, keepdims=True)
		std0 = np.std(simmat, axis=0, keepdims=True)
		simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

		N = len(pairs[0])
		preds = [simmat[i, i + N] for i in range(N)]
		preds = np.exp(preds)
		preds = np.array(preds)
	
	predsMedian = [int(x) for x in preds >= np.median(preds)]
	
	predsDevThreshold = [int(x) for x in preds >= devThreshold]

	return predsMedian, predsDevThreshold

@torch.no_grad()
@Timer(elapsed, "sentsim")
def predictRTE(
	model: FireWord,
	pairs,
	devPairs,
	devLabels,
	sif_alpha=1e-3,
):
	vocab: Vocab = model.vocab

	def computeThresholdFromDevData():
		counts = pd.Series(vocab.counts_dict())
		probs = counts / counts.sum()
		sif_weights: Mapping[str, float] = {
			w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
		}

		sents1 = devPairs[0]
		sents2 = devPairs[1]
		allsents = sents1 + sents2
		allsents = [
			[w for w in sent if w in sif_weights and w != vocab.unk]
			for sent in allsents
		]

		""" similarity """
		with Timer(elapsed, "similarity", sync_cuda=True):
			simmat = sentence_simmat(model, allsents, sif_weights)

		""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
		halved bc (9)"""
		with Timer(elapsed, "regularization"):
			diag = np.diag(simmat)
			simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

		with Timer(elapsed, "smooth"):
			mean1 = np.mean(simmat, axis=1, keepdims=True)
			std1 = np.std(simmat, axis=1, keepdims=True)
			mean0 = np.mean(simmat, axis=0, keepdims=True)
			std0 = np.std(simmat, axis=0, keepdims=True)
			simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

			N = len(devPairs[0])
			preds = [simmat[i, i + N] for i in range(N)]
			preds = np.exp(preds)
			preds = np.array(preds)

		bestThreshold = 0
		bestAccuracy = 0
		low = min(preds)
		high = max(preds)
		steps = math.ceil((high - low) / 2)*100

		for threshold in np.linspace(low, high, steps):
			accuracy =  sum([int(preds[i] >= threshold) == (devLabels[i]) for i in range(len(preds))]) / len(preds)
			if bestAccuracy < accuracy: 
				bestThreshold = threshold
				bestAccuracy = accuracy
		

		return bestThreshold

	devThreshold = computeThresholdFromDevData()

	counts = pd.Series(vocab.counts_dict())
	probs = counts / counts.sum()
	sif_weights: Mapping[str, float] = {
		w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
	}

	sents1 = pairs[0]
	sents2 = pairs[1]
	allsents = sents1 + sents2
	allsents = [
		[w for w in sent if w in sif_weights and w != vocab.unk]
		for sent in allsents
	]

	""" similarity """
	with Timer(elapsed, "similarity", sync_cuda=True):
		simmat = sentence_simmat(model, allsents, sif_weights)

	""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) 
	halved bc (9)"""
	with Timer(elapsed, "regularization"):
		diag = np.diag(simmat)
		simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

	with Timer(elapsed, "smooth"):
		mean1 = np.mean(simmat, axis=1, keepdims=True)
		std1 = np.std(simmat, axis=1, keepdims=True)
		mean0 = np.mean(simmat, axis=0, keepdims=True)
		std0 = np.std(simmat, axis=0, keepdims=True)
		simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

		N = len(pairs[0])
		preds = [simmat[i, i + N] for i in range(N)]
		preds = np.exp(preds)
		preds = np.array(preds)
	
	predsMedian = ["entailment" if x else "not_entailment" for x in preds >= np.median(preds)]
	
	predsDevThreshold = ["entailment" if x else "not_entailment" for x in preds >= devThreshold]

	return predsMedian, predsDevThreshold

if __name__ == "__main__":
	main()
