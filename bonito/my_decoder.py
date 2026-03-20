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
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

from .local_aligner import local_aligner, align_top
from fast_ctc_decode import vanilla_beam_search_log, convolutional_beam_search_log, marker_convolutional_beam_search_log, beam_search, primer_beam_search_brute

def check_args(args):
    if not Path(args.sim_targets).exists():
        print(f"Error: sim_targets file not found: {args.sim_targets}", file=sys.stderr)
        sys.exit(1)
    if args.code != 'conv' and args.code != 'conv+marker':
        print(f"Only conv and conv+marker code arguments are available.", file=sys.stderr)
        sys.exit(1)
    if args.conv_code and not Path(args.conv_code).exists():
        print(f"Error: conv_code file not found: {args.conv_code}", file=sys.stderr)
        sys.exit(1)
    if args.ref_file and not Path(args.ref_file).exists():
        print(f"Error: REF file not found: {args.ref_file}", file=sys.stderr)
        sys.exit(1)

    raw_file = args.fast5
    if raw_file.endswith('fast5'):
        if args.bam_file and not Path(args.bam_file).exists():
            print(f"Error: BAM file not found: {args.bam_file}", file=sys.stderr)
            sys.exit(1)
    elif raw_file.endswith('hdf5'):
        if args.ref_file and not Path(args.ref_file).exists():
            print(f"Error: REF file not found: {args.ref_file}", file=sys.stderr)
            sys.exit(1)
    if args.primer_search != "dynamic" and args.primer_search != "guppy":
        print(f"Only dynamic and guppy primer_search arguments are available.", file=sys.stderr)
        sys.exit(1)
    return

def get_reference_sequence(args, target_read_id, reference, target_fwd_primer, payload_length):

    primer_length = len(target_fwd_primer)
    alignment = get_reference_alignment(args.bam_file, target_read_id)
    prim_start = find_primer_start(args.bam_file, target_read_id, reference, target_fwd_primer)
    extracted_ref_sequence = reference.fetch(alignment["reference_name"], prim_start, prim_start + payload_length + 2 * primer_length,)

    return extracted_ref_sequence

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
    entries_trimmed = []
    for i, entry in enumerate(entries):
        if isinstance(entry, dict):
            tr = entry.get('read_id')
            if tr is not None:
                entries_trimmed.append(entry)
    
    return {entry['read_id']: entry for entry in entries_trimmed}

def validate_word(code_type, cc_code, mark_code, received):
    if code_type == "conv":
        return cc_code.is_codeword(received)
    elif code_type == "conv+marker":
        if not mark_code.validate_markers(received):
            return False
        cc_cw_est = mark_code.remove_markers(received)
        return cc_code.is_codeword(cc_cw_est)

def get_primer_start(sim_entry, primer_search, primer_offset, stride):
    if primer_search == 'guppy':
        start_pos = sim_entry['guppy_position'][0]
        start_pos = start_pos // stride
    elif primer_search == 'ctc':
        start_pos = sim_entry.get("ctc_primer_position")
        if isinstance(start_pos, dict):
            primer_offset_ = next(iter(start_pos))
            assert int(primer_offset_) == primer_offset, "wrong primer_offset stored in PS results"
            start_pos = start_pos[primer_offset_]
            start_pos = start_pos // stride
        elif start_pos is None:
            print("Primer starting pos estimate not available. Aborting..")
            quit()
        else:
            start_pos = start_pos // stride
    elif primer_search == 'stanford' or primer_search == 'basecall_lev':
        start_pos = sim_entry.get("ctc_basecall_lev_primer_position")
        if start_pos is None:
            print("Primer starting pos estimate not available. Aborting..")
            quit()
        elif isinstance(start_pos, list):
            start_pos = start_pos[primer_offset]
            start_pos = start_pos // stride
        else:
            start_pos = start_pos // stride
    elif primer_search == 'ctc_basecall_align' or primer_search == 'basecall_align':
        start_pos = sim_entry.get("ctc_basecall_align_primer_position")
        if start_pos is None:
            print("Primer starting pos estimate not available. Aborting..")
            quit()
        else:
            start_pos = start_pos // stride
    return start_pos


def main(args):

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

    samples = 0
    num_reads = 0
    max_read_size = 4e6
    dtype = np.float16 if args.half else np.float32
    reader = PreprocessFileReader(args.fast5)

    print(f"Loading inputs...", file=sys.stderr)
    conv_code_dict = load_code(args.conv_code)

    json_file_out = args.results_file
    json_file_out = os.path.expanduser(json_file_out)
    if os.path.exists(json_file_out):
        with open(json_file_out, 'r') as f:
            data_out = json.load(f)
    else:
        data_out = []

    print(f"Loading reference...", file=sys.stderr)
    if args.fast5.endswith('fast5'):
        reference = pysam.FastaFile(args.ref_file)
    elif args.fast5.endswith('hdf5'):
        f5 = h5py.File(args.fast5,'r')
        fasta_dict = read_fasta_to_dict(args.ref_file)

    processed_count = 0
    conv_codeword_length, num_marker_symbols = 0, 0
    cc_code, mark_code = None, None
    alphabet = "NACGT"
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    int_dict = {0:'A', 1:'C', 2:'G', 3:'T'}

    ## initialize code
    print(args.code)
    if args.code == 'conv' or args.code == 'conv+marker':
        cc_code = ConvCode(args.conv_code)
    if args.code == 'marker' or args.code=='conv+marker':
        marker_int = [base_dict[base] for base in args.marker]
        mark_code = MarkerCode(marker_int, args.marker_interval)

    if args.code == 'conv':
        code_params = f"{cc_code.c}_{cc_code.b}_{cc_code.K}"
    elif args.code == 'marker':
        code_params = f"{args.marker_interval}_{args.marker}"
    elif args.code == 'conv+marker':
        code_params = f"cc:{cc_code.c}_{cc_code.b}_{cc_code.K}_marker:{args.marker_interval}_{args.marker}"
    elif args.code == 'base':
        code_params = "base"

    ## set up coding things
    payload_length, conv_codeword_length, num_marker_symbols, rate, code_params = get_code_params(args, base_dict)
    assert args.seg_length > payload_length * 10 * 1.2, f"seg_length must be larger: not enough to accommodate 10 samples for each of {payload_length} bases."
    sim_results = get_init_entry(args, code_params, rate, payload_length, conv_codeword_length, num_marker_symbols)
    data_out.append(sim_results)
    entry_ind = len(data_out) - 1
    reads_entry_dict = load_json_metadata(args.sim_targets)

    # ==== for alignment ========
    aligner = local_aligner()
    aligner.mode = 'global'
    # ===========================
    max_sample_depth = 600 // model.stride
    t0 = time.perf_counter()
    pbar = tqdm(total=len(reads_entry_dict))
    total_scores_computations_track = [0, 0]
    with reader, torch.no_grad():

        while True:
            read = reader.queue.get()
            if read is None:
                break

            i, read_id, raw_data = read
            if read_id not in reads_entry_dict:
                continue
            sim_entry = reads_entry_dict[read_id]
            rc = sim_entry.get('rc', False)
            if rc == True:
                continue

            num_reads += 1
            samples += len(raw_data)
            target_read_id = sim_entry.get("read_id")
            target_fwd_primer = sim_entry.get("fwd_primer")
            primer_length = len(target_fwd_primer)
            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            #print(raw_data)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()
            primer_offset = args.primer_offset
            start_pos = get_primer_start(sim_entry, args.primer_search, primer_offset, model.stride)
            if start_pos < 0 or start_pos > posteriors.shape[0]:
                print("start_pos invalid")
                continue
            elif posteriors.shape[0] - start_pos < payload_length + 2*primer_length:
                print("too short post matrix")
                continue
            #assert start_pos >= 0 and start_pos < posteriors.shape[0], "start_pos invalid"
            #assert posteriors.shape[0] - start_pos > payload_length + primer_length, "too short post matrix"
            # crop posteriors
            end_col = min(posteriors.shape[0], start_pos + ((payload_length + primer_length*2)*20) // model.stride)
            cropped_posteriors = posteriors[start_pos : end_col]
            #print("="*40)
            #np.save('test.post', cropped_posteriors)
            # get reference payload
            if args.fast5.endswith('fast5'):
                ref_sequence = get_reference_sequence(args, target_read_id, reference, target_fwd_primer, payload_length)
            elif args.fast5.endswith('hdf5'):
                ref_id = f5[read_id].attrs['ref'].decode("utf-8")
                ref_sequence = fasta_dict[ref_id]

            if len(ref_sequence) < payload_length + 2*primer_length:
                print("Ref sequence too short. Continuing...")
                continue

            payload = ref_sequence[primer_length : payload_length + primer_length]
            target_back_primer = ref_sequence[payload_length + primer_length : payload_length + 2 * primer_length]
            if len(payload) != payload_length:
                print('Payload length mismatch')
                continue
            elif len(target_fwd_primer) != len(target_back_primer):
                print('Primer length mismatch')
                continue

            payload_int = [base_dict[o] for o in payload]
            fwd_primer_int = [base_dict[o] for o in target_fwd_primer]
            back_primer_int = [base_dict[o] for o in target_back_primer]

            # generate random codeword and get corresponding offset
            codeword = get_random_codeword(args, payload, cc_code, mark_code, conv_codeword_length, base_dict)
            assert len(codeword) == payload_length
            offset_int = [(4-codeword[x] + payload_int[x]) % 4 for x in range(payload_length)]
            offset_sequence = ''.join([int_dict[el] for el in offset_int])

            # Run integrated basecaller+decoder
            if args.code == "conv":
                try:
                    #print("entering beam search")
                    seq, _starts, score, total_score_computations = convolutional_beam_search_log(
                                cropped_posteriors,
                                alphabet,
                                args.beam,
                                1e-80,
                                True,
                                target_fwd_primer[primer_offset:],target_back_primer[:primer_length-primer_offset],offset_sequence,
                                conv_code_dict
                            )
                    #print("completed beam search")
                except ValueError as e:
                    print(f"ValueError (input validation failed): {e}")
                except RuntimeError as e:
                    print(f"RuntimeError (execution error): {e}")
                except Exception as e:
                    print(f"Unexpected error {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
            elif args.code == "conv+marker":
                try:
                    #print("post shape: ", cropped_posteriors.shape)
                    #print(cropped_posteriors)
                    #print(alphabet, args.beam, args.marker_interval, args.marker)
                    #print(target_fwd_primer,target_back_primer,offset_sequence)
                    #print(conv_code_dict)
                    #print("entering beam search")
                    seq, _starts, score, total_score_computations = marker_convolutional_beam_search_log(
                                cropped_posteriors,
                                alphabet,
                                args.beam,
                                1e-150,
                                True,
                                target_fwd_primer[primer_offset:],target_back_primer[:primer_length-primer_offset],offset_sequence,
                                conv_code_dict,
                                args.marker_interval, args.marker
                            )
                    #print("completed beam search")
                except Exception as e:
                    print(f"Exception during conv CTC decoder: {type(e).__name__}")
                    print("Exception during conv+marker CTC decoder")
            elif args.code == "base":
                try:
                    seq_base, _starts, score = vanilla_beam_search_log(
                                cropped_posteriors,
                                alphabet,
                                args.beam,
                                payload_length,
                                1e-80,
                                True,
                                target_fwd_primer[primer_offset:],target_back_primer[:primer_length-primer_offset]
                            )
                except Exception as e:
                    print(f"Exception during conv CTC decoder: {type(e).__name__}")
                    print("Exception during base CTC decoder")

            r"""
            print("primers: ", target_fwd_primer, ", ", target_back_primer)

            basecalled, path, total_scores_computed = beam_search(posteriors[start_pos-off:], alphabet, beam_size=5, beam_cut_threshold=1e-30)
            print("\n=========== align basecaller result ==============")
            payload_ref = ''.join(ref_sequence)
            est_payload_str =  basecalled

            print("payload len: ", len(payload_ref), ", basecalled len: ", len(est_payload_str))

            al0 = align_top(payload_ref, basecalled, aligner=aligner)
            score0, aligned_seq0, pattern, aligned_seq1, _alignment = al0
            print(f"{aligned_seq0[:170]}\n{pattern[:170]}\n{aligned_seq1[:170]}")

            print("guppy positions: ", sim_entry['guppy_position'])


            print("\n---------- align uncoded result -------------")
            seq_base, _starts, score = vanilla_beam_search_log(
                            cropped_posteriors,
                            alphabet,
                            args.beam,
                            payload_length,
                            1e-30,
                            True,
                            target_fwd_primer,target_back_primer
                        )

            payload_ref = ''.join(ref_sequence)
            est_payload_str =  seq_base

            print("payload len: ", len(payload_ref), ", uncoded len: ", len(est_payload_str))

            al0 = align_top(payload_ref, est_payload_str, aligner=aligner)
            score0, aligned_seq0, pattern, aligned_seq1, _alignment = al0
            print(f"{aligned_seq0}\n{pattern}\n{aligned_seq1}")

            print("\n -------- align decoder result -----------")
            payload_ref = ''.join(ref_sequence)
            est_payload_str =  seq

            print("payload len: ", len(payload_ref), ", decoded len: ", len(est_payload_str))

            al0 = align_top(payload_ref, est_payload_str, aligner=aligner)
            score0, aligned_seq0, pattern, aligned_seq1, _alignment = al0
            print(f"{aligned_seq0}\n{pattern}\n{aligned_seq1}")

            print("\n*********************************************************\n")
            # =======================================================================
            """
            data_out[entry_ind]["numIterations"] += 1

            # incomplete path: erased symbols considered erroneous. non-erased symbols are accounted for in symbol error calculations
            if len(seq) < payload_length + primer_length-primer_offset:
                # update frame error result, continue
                data_out[entry_ind]["frameErrors"] += 1
                seq = seq[primer_length-primer_offset:]
                est_payload = [base_dict[o] for o in seq]
                est_codeword = [ (est_payload[j] - offset_int[j]) % 4 for j in range(len(seq))]
                data_out[entry_ind]["symbolErrors"] += sum([1 for j in range(len(seq)) if est_codeword[j] != codeword[j]])
                data_out[entry_ind]["symbolErrors"] += (payload_length - len(seq))
                data_out[entry_ind]["FER"] = data_out[entry_ind]["frameErrors"] / data_out[entry_ind]["numIterations"]
                data_out[entry_ind]["SER"] = data_out[entry_ind]["symbolErrors"] / (data_out[entry_ind]["numIterations"] * payload_length)
                data_out[entry_ind]["misdecoding_scores"].append(float(score))
                if args.code != "base":
                    total_scores_computations_track[0] += total_score_computations / (end_col-start_pos+1)
                    total_scores_computations_track[1] += end_col-start_pos
                    data_out[entry_ind]["complexity"] = [total_scores_computations_track[0]/data_out[entry_ind]["numIterations"], 
                                                         total_scores_computations_track[1]/data_out[entry_ind]["numIterations"]]
                #lev_dist = levenshteinIterative(decoded_codewords[x], batch_codewords[idx, :].tolist())
            else:
                # decoding result is a complete path
                assert seq[:primer_length-primer_offset] == target_fwd_primer[primer_offset:], 'fwd primer mismatch in decoding result'
                if len(seq) == payload_length + 2*(primer_length-primer_offset):
                    assert seq[-primer_length+primer_offset:] == target_back_primer[:-primer_offset], 'back primer mismatch in decoding result'

                # undo offset, check if valid marker codeword/convolutional codeword
                seq = seq[primer_length-primer_offset:primer_length-primer_offset+payload_length]
                est_payload = [base_dict[o] for o in seq]
                est_codeword = [ (est_payload[j] - offset_int[j]) % 4 for j in range(payload_length)]
                assert validate_word(args.code, cc_code, mark_code, est_codeword), "invalid decoder output"

                if est_codeword != codeword:
                    data_out[entry_ind]["frameErrors"] += 1
                    if len(est_codeword) == payload_length:
                        data_out[entry_ind]["symbolErrors"] += sum([1 for j in range(payload_length) if est_codeword[j] != codeword[j]])
                    data_out[entry_ind]["misdecoding_scores"].append(float(score))
                else:
                    data_out[entry_ind]["correct_decoding_scores"].append(float(score))
                data_out[entry_ind]["FER"] = data_out[entry_ind]["frameErrors"] / data_out[entry_ind]["numIterations"]
                data_out[entry_ind]["SER"] = data_out[entry_ind]["symbolErrors"] / (data_out[entry_ind]["numIterations"] * payload_length)
                if args.code != "base":
                    total_scores_computations_track[0] += total_score_computations / (end_col-start_pos+1)
                    total_scores_computations_track[1] += end_col-start_pos
                    data_out[entry_ind]["complexity"] = [total_scores_computations_track[0]/data_out[entry_ind]["numIterations"],
                                                            total_scores_computations_track[1]/data_out[entry_ind]["numIterations"]]

            # Write to temporary file first, then rename
            os.makedirs(os.path.dirname(json_file_out), exist_ok=True)
            temp_file = json_file_out + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data_out, f, indent=4)
            os.rename(temp_file, json_file_out)
            pbar.update(1)


    duration = time.perf_counter() - t0
    if args.code != "base":
        print('Avg. complexity: ', data_out[entry_ind]["complexity"][0])
    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("fast5", help="Fast5 file containing reads")
    parser.add_argument("--ref_file", type=str, help="Path to ref file")
    parser.add_argument("--bam_file", type=str, help="Path to BAM file")
    parser.add_argument("--sim_targets", required=True, type=str,
                       help="JSON file from which simulation entries must be read. each entry must contain the raw signal ID, forward primers (and reverse primer?) and the estimated starting position of the primer in the raw signal")
    parser.add_argument("--results_file", type=str, help="Path to output json file")
    parser.add_argument('--primer_search', default='ctc', type=str, help='starting position of primer as given by ctc/basecalling+aligning')
    parser.add_argument('--primer_offset', default=0, type=int, help='which base of primer to start from')

    parser.add_argument("--code", required=True, type=str, help="Code class")
    parser.add_argument('--seg_length',type = int,default = 4096, help='length of signals used for one decoding,default 4096')
    parser.add_argument('--payload_length',type = int,default = 100, help='number of bases in payload')
    parser.add_argument('--conv_code', type=str, default="lokatt/tensorflow_op/conv_code/syndrome_JSON/cc_2_1_5.json", help='convolutional code to be used: give full path to JSON file')
    parser.add_argument("--marker_interval", type=int, default=5, help="Marker interval")
    parser.add_argument("--marker", type=str, default="AC",  help="Marker string")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beam", default=512, type=int)
    parser.add_argument("--half", action="store_true", default=False)
    return parser
