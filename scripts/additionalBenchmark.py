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

@torch.no_grad()
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--checkpointsMRS",
		nargs="+",
		default=[
			"checkpoints/wacky_mlplanardiv_d2_l4_k1_polysemy",
			"checkpoints/wacky_mlplanardiv_d2_l4_k10",
			"checkpoints/wacky_mlplanardiv_d2_l8_k20",
		],
	)

	args = parser.parse_args()

	sifA = 0.001
	device = "cpu"

	# for checkpoint in args.checkpointsMRS:
	checkpoint = args.checkpointsMRS[0]
	model = FireWord.from_pretrained(checkpoint).to(device)
	
	vocab: Vocab = model.vocab
	
	pairs, labels = prepareMSRData('scripts/benchmarks/MSR/msr_paraphrase_train.csv')
	
	assert len(pairs[0]) == len(labels) == len(pairs[0])

	# print(f"pairs: {len(pairs[0])}")
	score, preds = benchmark_sentence_similarity(model, pairs, labels, sifA)
	
	print(f"score: {score}")
	print(f"(preds, labels):\n{np.array([preds, labels])}")




			



@torch.no_grad()
@Timer(elapsed, "sentsim")
def benchmark_sentence_similarity(
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

	scores = 0

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

		print("simmat")
		print(simmat)
		print(f"max: {max(max(x) for x in simmat)}")
		print(f"min: {min(min(x) for x in simmat)}")

	""" regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) """
	with Timer(elapsed, "regularization"):
		diag = np.diag(simmat)
		simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))
		print("diag")
		print(diag)
		print("diag.reshape(-1, 1)")
		print(diag.reshape(-1, 1))
		print("diag.reshape(1, -1)")
		print(diag.reshape(1, -1))
		print("(diag.reshape(-1, 1) + diag.reshape(1, -1))*0.5")
		print((diag.reshape(-1, 1) + diag.reshape(1, -1))/2)
		print("simmat")
		print(simmat)
		print(f"max: {max(max(x) for x in simmat)}")
		print(f"min: {min(min(x) for x in simmat)}")

	""" rescaling (smoothing) and exponential """

	def _simmat_rescale(simmat) -> np.ndarray:
		scale = np.abs(simmat).mean(axis=1, keepdims=True)
		simmat = simmat / (scale * scale.T) ** 0.5

		print("scale")
		print(scale)
		print("(scale * scale.T)")
		print((scale * scale.T))
		print("(scale * scale.T)** 0.5")
		print((scale * scale.T)** 0.5)
		print("simmat")
		print(simmat)
		print(f"max: {max(max(x) for x in simmat)}")
		print(f"min: {min(min(x) for x in simmat)}")

		return simmat

	with Timer(elapsed, "smooth"):
		simmat = _simmat_rescale(simmat)
		simmat = np.exp(simmat)
		print("exp simmat")
		print(simmat)
		print(f"max: {max(max(x) for x in simmat)}")
		print(f"min: {min(min(x) for x in simmat)}")

	N = len(pairs[0])
	preds = [simmat[i, i + N] for i in range(N)]
	# print(simmat.shape)
	# print(simmat)
	# print(f"max: {max(max(x) for x in simmat)}")
	# print(f"min: {min(min(x) for x in simmat)}")
	# print(sum([x >= 1 for x in simmat]))
	# print()

	score = sum([int(preds[i] >= 0.5) == (labels[i]) for i in range(len(preds))]) / len(preds)

	return score, np.array(preds)





if __name__ == "__main__":
	main()
