from __future__ import annotations
from typing import List, Mapping
import argparse
import os
import random
import numpy as np
from contextlib import nullcontext
import pandas as pd

import torch
from torch.optim import AdamW, Adam, SGD, Adagrad
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from corpusit import Vocab
from firelang import FireWord, FireTensor
from firelang.utils.optim import DummyScheduler
from firelang.utils.log import logger
from firelang.utils.timer import elapsed, Timer
from scripts.corpusPreprocessor import *
import math
from firelang.utils.optim import Loss
from torch.nn import Module, functional as F
from scripts.sentsim import sentsim_as_weighted_wordsim_cuda
from scripts.additionalBenchmark import *

logger.setLevel(level=os.environ.get("LOGLEVEL", "DEBUG").upper())


total_timer = Timer(elapsed, "total")


@total_timer
def fineTune(args):
	torch.set_num_threads(1)

	device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
	
	set_seed(args.seed)
	
	model = FireWord.from_pretrained(args.pretrainedModel)

	sentencePairs = []
	labels = []
	match args.task:
		case "MRPC":
			sentencePairs, labels = prepareMRPCData('scripts/tasks/MRPC/msr_paraphrase_train.csv')
			evalPairs = sentencePairs[int(len(sentencePairs[0]) / 10):]
			evalLabels = labels[int(len(sentencePairs[0]) / 10):]
		case "SST-2": 
			sentencePairs, labels, evalPairs, evalLabels, _, _ = prepareSSTGlueData('scripts/tasks/SSTGLUE/train.tsv',
																		   'scripts/tasks/SSTGLUE/dev.tsv', 'scripts/tasks/SSTGLUE/test.tsv')
		case "RTE":
			sentencePairs, labels, evalPairs, evalLabels, _, _ = prepareRTEGlueData('scripts/tasks/RTE/train.tsv',
																		   'scripts/tasks/RTE/dev.tsv', 'scripts/tasks/RTE/test.tsv')
	indices = list(range(len(sentencePairs[0])))

	logger.info(model)

	model = model.to(device)
	model.train()

	if args.optimizer == "adamw":
		optimizer = AdamW(
			model.parameters(), lr=args.lr, weight_decay=args.weight_decay
		)
	elif args.optimizer == "adam":
		optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == "adagrad":
		optimizer = Adagrad(
			model.parameters(), lr=args.lr, weight_decay=args.weight_decay
		)
	elif args.optimizer == "sgd":
		optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	if args.lr_scheduler == "OneCycleLR":
		scheduler = OneCycleLR(
			optimizer,
			max_lr=args.lr,
			total_steps=args.n_iters // args.accum_steps + 5,
			div_factor=1.0,
			final_div_factor=20.0,
		)
	else:
		scheduler = DummyScheduler(args.lr)
	logger.info(f"Initialized optimizer and scheduler")
	logger.info(f"	Optimizer: {optimizer}")
	logger.info(f"	Scheduler: {scheduler}")

	if args.amp:
		scaler = GradScaler()
		autocaster = autocast()
	else:
		autocaster = nullcontext()
	if args.profile:
		prof = torch.autograd.profiler.profile(use_cuda=True)
	else:
		prof = nullcontext()

	for i in range(1, args.n_iters + 1):

		with Timer(elapsed, "prepare", sync_cuda=True):
			iterationIndices = random.sample(indices, args.sz_batch)
			iterationPairs = [[sentencePairs[0][x] for x in iterationIndices], [sentencePairs[1][x] for x in iterationIndices]]
			iterationLabels = [labels[x] for x in iterationIndices]
		""" ----------------- forward pass -------------------"""
		with prof, autocaster, Timer(elapsed, "forward", sync_cuda=True):
			model: FireWord

			loss = Loss()

			logits = predictSentencePairs(model, iterationPairs)

			lossSim = F.binary_cross_entropy_with_logits(
				FireTensor(torch.tensor(logits, dtype=torch.float, device=device, requires_grad=True)),
				FireTensor(torch.tensor(iterationLabels, dtype=torch.float, device=device, requires_grad=True)), reduction="none"
			)
			loss.add("sim", lossSim)

			total_loss = loss.reduced_total()
			steploss = total_loss / args.accum_steps
		if args.profile:
			logger.debug("----- forward -----")
			logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
		""" ----------------- backward pass -------------------"""
		with prof, Timer(elapsed, "backward", sync_cuda=True):
			if args.amp:
				scaler.scale(steploss).backward()
			else:
				steploss.backward()

			grad_norm = (
				torch.cat([p.grad.data.reshape(-1) for p in model.parameters()])
				.norm()
				.item()
			)

		if args.profile:
			logger.debug("----- backward -----")
			logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
		""" ----------------- optim -------------------"""
		if i % args.accum_steps == 0:
			with Timer(elapsed, "optim", sync_cuda=True):
				with Timer(elapsed, "step"):
					for name, p in model.named_parameters():
						isnan = p.grad.isnan()
						isinf = p.grad.isinf()
						isinvalid = isnan | isinf
						if isinvalid.any():
							p.grad.masked_fill_(isinvalid, 0)
							p[isinvalid].normal_(0, 0.1)
							print(f"Fixed nan/inf values in grad of {name}")
							print(f"  grad = {p.grad}")

					if args.amp:
						scaler.step(optimizer)
						scaler.update()
					else:
						optimizer.step()

				with Timer(elapsed, "lrstep", sync_cuda=True):
					scheduler.step()

				with Timer(elapsed, "zerograd", sync_cuda=True):
					model.zero_grad()

		if i % args.eval_interval == 0:

			os.makedirs(args.savedir, exist_ok=True)
			model.eval()

			"""--------------- similarity benchmark ---------------"""
			with Timer(elapsed, "benchmark on evaluation data", sync_cuda=True):
					match args.task:
						case "MRPC":
							#F1 score
							simscore = benchmarkMRPC(model, evalPairs, evalLabels)[4]
						case "SST-2":
							#threshold search based predictions 
							predictions = predictSSTGlue(model, evalPairs, evalPairs, evalLabels)[1]
							#accuracy
							simscore = sum([predictions[x] == evalLabels[x] for x in range(len(predictions))]) / len(predictions)
						case "RTE":
							#threshold search based predictions
							predictions = predictRTE(model, evalPairs, evalPairs, evalLabels)[1]
							#accuracy
							simscore = sum([predictions[x] == evalLabels[x] for x in range(len(predictions))]) / len(predictions)
					
			if simscore > best_simscore:
				best_iter = i
				best_simscore = simscore
				best_loss = total_loss.item()
				model.save(args.savedir)

			logger.info(
				f"Iter {i}. Loss={loss}; grad={grad_norm:.3g}; "
				f"lr={scheduler.get_last_lr()[0]:.3g}; "
				f"sim={simscore:.3f}%"
				f"loss={total_loss}"
				f"best iteration: {best_iter}"
				f"best sim={best_simscore}"
				f"best loss={best_loss}"
			)
			total_timer.update()
			logger.debug("-- Elapsed --\n" + elapsed.format(thresh=0.8))


			model.train()

	model.eval()

def batched_cross_selfsim(model, words, col_batch_size=100):
    x: FireTensor = model[words]
    wordsim = np.zeros((len(words), len(words)), dtype=np.float32)
    for i in range(0, len(words), col_batch_size):
        xbatch = model[words[i : i + col_batch_size]]
        _wordsim = x.measures.integral(xbatch.funcs, cross=True).data.cpu().numpy()
        wordsim[i : i + col_batch_size, :] += _wordsim
        wordsim[:, i : i + col_batch_size] += _wordsim.T
    return wordsim


def sentence_simmat(model, sents: List[List[str]], sif_weights: Mapping[str, float]):
    allwords = sorted(list(set([w for sent in sents for w in sent])))
    weights = np.array([sif_weights[w] for w in allwords], dtype=np.float32)
    s2i = dict(zip(allwords, range(len(allwords))))

    wordsim = batched_cross_selfsim(model, allwords)

    idseqs = [[s2i[w] for w in sent] for sent in sents]

    sentsim = sentsim_as_weighted_wordsim_cuda(
        wordsim, weights, idseqs, device=model.detect_device().index
    )
    return sentsim

def predictSentencePairs(
	model: FireWord,
	pairs,
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
	print("#######################################################################################")
	print(type(simmat))
	print("#######################################################################################")
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

	return preds

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_arguments():
	parser = argparse.ArgumentParser()

	# ----- optimize -----
	parser.add_argument("--n_iters", type=int, default=100000)
	parser.add_argument(
		"--sz_batch",
		type=int,
		default=8192,
		help="set it large to put all assets in one batch, or make it small to reduce memory usage",
	)
	parser.add_argument("--optimizer", type=str, default="adamw")
	parser.add_argument("--lr", type=float, default=0.005)
	parser.add_argument("--lr_scheduler", type=str, default="OneCycleLR")
	parser.add_argument(
		"--accum_steps",
		type=int,
		default=10,
		help="Update parameters every several iterations.",
	)
	parser.add_argument("--weight_decay", type=float, default=1e-6)
	parser.add_argument(
		"--sinkhorn_weight",
		type=float,
		default=0.0,
		help="Weight of the Sinkhorn distance term in the total loss.",
	)
	parser.add_argument(
		"--sinkhorn_reg",
		type=float,
		default=1.0,
		help="Weight on the regularization term in the Sinkhorn distance",
	)
	parser.add_argument(
		"--sinkhorn_max_iter",
		type=int,
		default=50,
		help="A parameter of the Sinkhorn distance term that limits the number of estimating iterations.",
	)
	parser.add_argument(
		"--sinkhorn_p",
		type=float,
		default=2.0,
		help="Norm dimension of the Sinkhorn distance.",
	)
	parser.add_argument(
		"--sinkhorn_tau",
		type=float,
		default=1e3,
		help="Used for stablization of the Sinkhorn computation.",
	)
	parser.add_argument(
		"--sinkhorn_stop_threshold",
		type=float,
		default=1e-2,
		help="Controlling stop of the Sinkhorn iteration.",
	)

	# ----- miscellaneous -----
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--cpu", action="store_true", help="Use cpu rather than CUDA.")
	parser.add_argument("--savedir", type=str, default="./results/fineTuningResults/")

	parser.add_argument(
		"--amp", action="store_true", help="Use half precision for accelleration."
	)
	parser.add_argument("--profile", action="store_true", help="CPU/GPU profiling.")

	parser.add_argument("--tag", type=str, default=None)

	parser.add_argument("--pretrainedModel",
		type=str,
		default="",
		choices=[
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k1_polysemy", 
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l4_k10",
			"checkpoints/v1.1/wacky_mlplanardiv_d2_l8_k20"],
		help = "Choose the pretrained model to fine-tune."
	)

	parser.add_argument(
			"--eval_interval",
			type=int,
			default=1000,
			help="model is evaluated every eval_interval iterations and the snapshot is saved if it's better than the current one",
		)

	parser.add_argument("--task",
		type=str,
		default="",
		choices=["MRPC", "RTE", "SST-2"],
		help = "Choose the benchmark task to fine-tune for."
	)

	args = parser.parse_args()

	return args


if __name__ == "__main__":
	args = parse_arguments()

	fineTune(args)
