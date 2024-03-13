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

	for checkpoint in args.checkpointsMRS:
		model = firelang.modules.FireWord.from_pretrained(checkpoint).to("cuda")
		print(model)

		# 'benchmarks/MSR/msr_paraphrase_train.csv'









if __name__ == "__main__":
	main()