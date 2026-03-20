"""
Bonito Basecaller
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model
from bonito.io import PreprocessFileReader
from bonito.my_utils import *

import os
import json
import torch
import distance
import numpy as np
from pathlib import Path

from .local_aligner import local_aligner, align_top
from fast_ctc_decode import beam_search
from tqdm import tqdm

def load_json_metadata(json_file):
    """
    Load cropping metadata from JSON file.

    Expected JSON format:
    [
        {
            "read_id": "read_id_1",
            "start_pos": 1000,
            "primer1": "...",
            "primer2": "..."
        },
        ...
    ]

    Returns:
        dict: Mapping of read_id -> metadata dict
    """
    with open(json_file, 'r') as f:
        entries = json.load(f)

    return {entry['read_id']: entry for entry in entries}


def main(args):

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

    samples = 0
    num_reads = 0
    max_read_size = 4e6
    dtype = np.float16 if args.half else np.float32
    reader = PreprocessFileReader(args.fast5)

    # beam search params
    f_basename = os.path.basename(args.sim_targets[:-5])
    complexity_log_file = os.path.join(os.path.dirname(args.sim_targets), 'complexity_bs.txt')
    avg_num_computations = 0
    num_iter = 0

    print(f"Loading inputs...", file=sys.stderr)

    processed_count = 0
    alphabet = "NACGT"
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    int_dict = {0:'A', 1:'C', 2:'G', 3:'T'}

    reads_entry_dict = load_json_metadata(args.sim_targets)
    # ==== for alignment ========
    aligner = local_aligner()
    aligner.mode = 'global'
    # ===========================

    t0 = time.perf_counter()
    #sys.stderr.write("> calling\n")
    pbar = tqdm(total=len(reads_entry_dict))

    with reader, torch.no_grad():
        while True:
            read = reader.queue.get()
            if read is None:
                break
            target_read_id, raw_data = read
            if target_read_id not in reads_entry_dict:
                continue
            sim_entry = reads_entry_dict[target_read_id]

            num_reads += 1
            samples += len(raw_data)

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()
            seq, path, num_computations = beam_search(posteriors, alphabet, args.beam, args.thresh, True)
            avg_num_computations += num_computations / posteriors.shape[0]
            num_iter += 1

            pbar.update(1)
    duration = time.perf_counter() - t0

    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
    sys.stderr.write("> done\n")

    # log complexity file
    avg_num_computations = avg_num_computations / num_iter
    with open(complexity_log_file, 'a') as f:
        f.write(f'{f_basename} {args.beam} {args.thresh} {avg_num_computations}\n')

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("fast5", help="Fast5 file containing reads")
    parser.add_argument("--sim_targets", required=True, type=str,
                       help="JSON file from which simulation entries must be read. each entry must contain the raw signal ID, forward primers (and reverse primer?) and the estimated starting position of the primer in the raw signal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beam", default=5, type=int)
    parser.add_argument("--thresh", default=0.0, type=float)
    parser.add_argument("--half", action="store_true", default=False)
    return parser
