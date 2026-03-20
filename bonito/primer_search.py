"""
Bonito Basecaller
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model
from bonito.io import DecoderWriter, PreprocessFileReader
from bonito.my_utils import *

import os
import json
import torch
import distance
import numpy as np
from pathlib import Path

from .local_aligner import local_aligner, align_top
from fast_ctc_decode import primer_beam_search_brute, primer_beam_search_opt,  primer_beam_search_ss, beam_search
from tqdm import tqdm
r"""
def check_args(args):
    if not Path(args.sim_targets).exists():
        print(f"Error: sim_targets file not found: {args.sim_targets}", file=sys.stderr)
        sys.exit(1)
    if args.conv_code and not Path(args.conv_code).exists():
        print(f"Error: conv_code file not found: {args.conv_code}", file=sys.stderr)
        sys.exit(1)
    if args.bam_file and not Path(args.bam_file).exists():
        print(f"Error: BAM file not found: {args.bam_file}", file=sys.stderr)
        sys.exit(1)
    if args.ref_file and not Path(args.ref_file).exists():
        print(f"Error: REF file not found: {args.ref_file}", file=sys.stderr)
        sys.exit(1)
    if args.primer_search != "dynamic" and args.primer_search != "guppy":
        print(f"Only dynamic and guppy primer_search arguments are available.", file=sys.stderr)
        sys.exit(1)
    if args.code != 'conv' and args.code != 'conv+marker':
        print(f"Only conv and conv+marker code arguments are available.", file=sys.stderr)
        sys.exit(1)
    return
"""

def get_reference_sequence(args, target_read_id, reference, target_fwd_primer, payload_length):

    primer_length = len(target_fwd_primer)
    alignment = get_reference_alignment(args.bam_file, target_read_id)
    prim_start = find_primer_start(args.bam_file, target_read_id, reference, target_fwd_primer)
    extracted_ref_sequence = reference.fetch(alignment["reference_name"], prim_start, prim_start + payload_length + 2 * primer_length,)

    return extracted_ref_sequence


def return_ratio(deltas, target_delta=30):

    unique_deltas, unique_counts = np.unique(deltas, return_counts=True)

    unique_abs_deltas = np.sort(np.unique(np.abs(unique_deltas)))
    cum_delta_counts = []
    for abs_delta_val in unique_abs_deltas:
        cum_delta_count = 0
        for unique_delta, unique_count in zip(unique_deltas, unique_counts):
            if abs(unique_delta) <= abs_delta_val:
                cum_delta_count += unique_count
        cum_delta_counts.append(cum_delta_count)


    total = len(deltas)
    cum_delta_counts_norm = np.array(cum_delta_counts) / total
    #print(unique_abs_deltas)
    return cum_delta_counts_norm[np.where(unique_abs_deltas == target_delta)]

def get_primer_start_deltas(json_data, ind=0):
    method1 = "ctc_primer_position"
    method2 = "ctc_basecall_lev_primer_position"
    primer_start_deltas = []
    for entry in json_data:
        if isinstance(entry, dict) == False:
            continue
        rc = entry.get('rc', False)
        if rc == True:
            continue
        #print(type(float(entry["lokatt_primer_search_score"])))
        #score = entry.get("lokatt_score")#entry.get("lokatt_primer_search_score")
        pos1 = entry.get(method1)
        if method1 == "guppy_position":
            if len(pos1) >= 4:
                pos1 = pos1[4]
            else:
                pos1 = None
        elif method1 == "ctc_primer_position":
            if isinstance(pos1,dict):
                pos1 = pos1[ind]
        pos2 = entry.get(method2)#entry.get("lokatt_estimate_primer_start")
        if method2 == "guppy_position":
            if len(pos2) >= 4:
                pos2 = pos2[4]
            else:
                pos2 = None
        elif method2 == "ctc_basecall_lev_primer_position":
            pos2 = pos2[ind]
        if pos1 is not None and pos2 is not None and pos2 != -1:
            primer_start_deltas.append(pos2 - pos1)

    return primer_start_deltas


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

    # primer search params
    subsample = args.subsample
    ps_thresh = args.thresh
    complexity_log_file = os.path.join(os.path.dirname(args.results_file), 'complexity_ps.txt')
    avg_num_computations = 0
    num_iter = 0

    # load data from json_file_out, if any
    json_file_out = args.results_file
    data_out = []

    processed_count = 0
    alphabet = "NACGT"
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    int_dict = {0:'A', 1:'C', 2:'G', 3:'T'}

    print(f"Loading inputs...", file=sys.stderr)
    reads_entry_dict = load_json_metadata(args.sim_targets)
    # ==== for alignment ========
    aligner = local_aligner()
    aligner.mode = 'global'
    # ===========================
    if args.primer_search_method == 'ctc' or args.primer_search_method == 'both':
        shift = args.shift // model.stride
        max_sample_depth = args.max_sample_depth // model.stride
    t0 = time.perf_counter()

    sys.stderr.write("> calling\n")
    pbar = tqdm(total=len(reads_entry_dict))
    avg_sig_len = 0
    with reader, torch.no_grad():

        while True:

            read = reader.queue.get()
            if read is None:
                break
            i, target_read_id, raw_data = read
            if target_read_id not in reads_entry_dict:
                continue
            sim_entry = reads_entry_dict[target_read_id]
            rc = sim_entry.get('rc', False)
            if rc == True:
                continue

            num_reads += 1
            samples += len(raw_data)

            target_fwd_primer = sim_entry.get("fwd_primer")
            primer_length = len(target_fwd_primer)

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()
            avg_sig_len += posteriors.shape[0]
            # =========================== primer search ==================================
            if args.primer_search_method == 'ctc' or args.primer_search_method == 'both':
                if args.version == 'brute':
                    primer_scores, total_comp = primer_beam_search_brute(posteriors, args.beam, ps_thresh, target_fwd_primer[args.start_offset:], max_sample_depth, shift, subsample)
                elif args.version == 'opt':
                    primer_scores, total_comp = primer_beam_search_opt(posteriors, args.beam, args.examine_fraction, ps_thresh, args.conc_thresh, target_fwd_primer[args.start_offset:], max_sample_depth, shift, subsample)
                elif args.version == 'ss':
                    primer_scores, total_comp = primer_beam_search_ss(posteriors, args.beam, args.min_beam, ps_thresh, args.conc_thresh, args.rel_thresh, target_fwd_primer[args.start_offset:], max_sample_depth, args.trunc1, args.trunc2, shift, subsample)
                top_5_scores = sorted(range(len(primer_scores)), key=lambda i: primer_scores[i], reverse=True)[:5]
                sim_entry["ctc_primer_position"] = {args.start_offset : top_5_scores[0] * model.stride}
                sim_entry["ctc_primer_search_score"] = { args.start_offset : primer_scores[top_5_scores[0]]}
                avg_num_computations += total_comp / (posteriors.shape[0] - primer_length + 1)
                num_iter += 1

            if args.primer_search_method == 'basecall_align' or args.primer_search_method == 'both':
                # beam search & basecall
                sequence, path, total_scores_computed = beam_search(posteriors, alphabet, args.beam, 0.1)
                start_pos, min_edit_distance, minimizing_LD_index = get_primer_position_basecall_align(sequence, path, target_fwd_primer)
                sim_entry["ctc_basecall_align_primer_position"] = start_pos * model.stride
                sim_entry["ctc_basecall_align_primer_LD"] = min_edit_distance
                #print(minimizing_LD_index, len(sequence))
                if minimizing_LD_index >= 0:
                    al0 = align_top(target_fwd_primer, sequence[minimizing_LD_index:min(minimizing_LD_index+primer_length, len(sequence))], aligner=aligner)
                    score0, aligned_seq0, pattern, aligned_seq1, _alignment = al0
                    #print(f"{aligned_seq0}\n{pattern}\n{aligned_seq1}")
                    sim_entry["ctc_basecall_primer_align0"] = aligned_seq0
                    sim_entry["ctc_basecall_primer_align_pattern"] = pattern
                    sim_entry["ctc_basecall_primer_align1"] = aligned_seq1
                else:
                    sim_entry["ctc_basecall_primer_align0"] = ""
                    sim_entry["ctc_basecall_primer_align_pattern"] = ""
                    sim_entry["ctc_basecall_primer_align1"] = ""
            data_out.append(sim_entry)

            with open(json_file_out, 'w') as f:
                json.dump(data_out, f, indent=4)
            pbar.update(1)
    duration = time.perf_counter() - t0

    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
    sys.stderr.write(f"> avg CTC dimension = {avg_sig_len / num_reads}\n")
    sys.stderr.write("> done\n")


    # log complexity file
    f_basename = os.path.basename(json_file_out[:-5])
    avg_num_computations = avg_num_computations / num_iter
    if args.primer_search_method == 'ctc' or args.primer_search_method == 'both':
        ratios = [0, 0, 0]
        i = 0
        for ind in [0,1,2]:
            primer_search_deltas = get_primer_start_deltas(data_out, args.start_offset)
            ratio = return_ratio(primer_search_deltas, target_delta=51)
            ratios[i] = ratio[0]
            i += 1
        print(args.beam, subsample, ratios, avg_num_computations)

        if args.version == 'brute':
            with open(complexity_log_file, 'a') as f:
                f.write(f'BR {f_basename} {len(reads_entry_dict)} {subsample} {args.beam} {args.max_sample_depth} {ps_thresh} {avg_num_computations} {args.start_offset} [{ratios[0]} {ratios[1]} {ratios[2]}]\n')
        elif args.version == 'opt':
            with open(complexity_log_file, 'a') as f:
                f.write(f'TR {f_basename} {len(reads_entry_dict)} {subsample} {args.beam} {args.min_beam} {args.max_sample_depth} {args.trunc1} {args.trunc2} {ps_thresh} {args.conc_thresh} {args.rel_thresh} {avg_num_computations} {args.start_offset} [{ratios[0]} {ratios[1]} {ratios[2]}]\n')
        elif args.version == 'ss':
            with open(complexity_log_file, 'a') as f:
                f.write(f'SS {f_basename} {len(reads_entry_dict)} {subsample} {args.beam} {args.min_beam} {args.max_sample_depth} {args.trunc1} {args.trunc2} {ps_thresh} {args.conc_thresh} {args.rel_thresh} {avg_num_computations} {args.start_offset} [{ratios[0]} {ratios[1]} {ratios[2]}]\n')
def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("fast5", help="Fast5 file containing reads")
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--sim_targets", required=True, type=str,
                       help="JSON file from which simulation entries must be read. each entry must contain the raw signal ID, forward primers (and reverse primer?) and the estimated starting position of the primer in the raw signal")
    parser.add_argument("--results_file", type=str, help="Output json file")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)

    parser.add_argument("--primer_search_method", type=str, default='ctc', help="Method to localize primer")
    parser.add_argument("--version", default='opt', type=str, help="which version of PrimerSeeker to run")
    # ps params
    parser.add_argument("--beam", default=8, type=int)
    parser.add_argument("--shift", default=100, type=int)
    parser.add_argument("--subsample", default=6, type=int)
    parser.add_argument("--max_sample_depth", default=600, type=int)
    parser.add_argument("--examine_fraction", default=1.0, type=float, help="only the first part of the raw signal will be examined for the starting primer")
    parser.add_argument("--conc_thresh", default=0.995, type=float)
    parser.add_argument("--thresh", default=1e-20, type=float) # beam_cut_threshold

    # additional ps params (unnecessary for _opt version of PrimerSeeker)
    parser.add_argument("--min_beam", default=4, type=int)
    parser.add_argument("--trunc1", default=200, type=int)
    parser.add_argument("--trunc2", default=100, type=int)
    parser.add_argument("--rel_thresh", default=1e-7, type=float)
    parser.add_argument("--start_offset", default=0, type=int)
    return parser
