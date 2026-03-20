"""
Bonito Basecaller
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
from bonito.util import load_model
from bonito.io import DecoderWriter, PreprocessFileReader
from bonito.my_utils import *
import torch
import numpy as np
import h5py
from tqdm import tqdm

def main(args):
    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

    samples = 0
    num_reads = 0
    max_read_size = 4e6
    dtype = np.float16 if args.half else np.float32
    reader = PreprocessFileReader(args.fast5)
    f5 = h5py.File(args.fast5,'r')

    assert len(args.start_barcode) == len(args.end_barcode)
    START_BARCODE = args.start_barcode
    END_BARCODE = args.end_barcode
    START_BARCODE_RC = reverse_complement(END_BARCODE)
    END_BARCODE_RC = reverse_complement(START_BARCODE)

    json_file_out = args.output_json_file
    data_out = []

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")
    pbar = tqdm(total=len(f5.keys()))
    skipped = 0
    with reader, torch.no_grad():

        while True:

            read = reader.queue.get()
            if read is None:
                break

            i, read_id, raw_data = read
            #if read_id != "read_00258e61-fa4f-4d3c-b674-e3e12bb6b51d":
            #    continue
            #print(raw_data)
            #print('bonito: raw_data.shape: ', raw_data.shape)
            if len(raw_data) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read_id, len(raw_data)))
                continue

            num_reads += 1
            samples += len(raw_data)

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            #print(raw_data)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()

            #print('CTC dim: ', posteriors.shape[0], ', start_pos_l: ', start_pos_l, ', start_pos_r: ', start_pos_r, ', end_pos', end_pos)
            #print('CTC dim: ', posteriors.shape[0], ', start_pos_l_RC: ', start_pos_RC_l, ', start_pos_r_RC: ', start_pos_RC_r, ', end_pos', end_pos_RC)
            (start_pos, end_pos, dist_start, dist_end) = find_barcode_pos_in_posteriors_ps(posteriors, START_BARCODE, END_BARCODE)
            (start_pos_RC, end_pos_RC, dist_start_RC, dist_end_RC) = find_barcode_pos_in_ps(posteriors, START_BARCODE_RC, END_BARCODE_RC)
            #print('start_pos: ', start_pos)
            #print('start_pos_RC: ', start_pos_RC)
            rc = False
            if dist_start + dist_end > dist_start_RC + dist_end_RC:
                rc = True
                start_pos = start_pos_RC
                end_pos = end_pos_RC

            if start_pos[0] == -1 or end_pos - start_pos[-1] + 1 < args.min_len + 1:
                skipped += 1
                pbar.update(1)
                continue

            #print('start_pos: ', start_pos, 'end_pos: ', end_pos, ' min_len: ', args.min_len, ' end_pos - start_pos[-1]:', end_pos - start_pos[-1])

            #start_pos = start_pos_l * model.stride
            start_pos = [i * model.stride for i in start_pos]
            end_pos = end_pos * 3
            entry_barcode = START_BARCODE
            if rc:
                entry_barcode = START_BARCODE_RC
            new_entry = {
                "read_index": i,
                "read_id": read_id,
                "ctc_len": posteriors.shape[0] * 3,
                "fwd_primer": entry_barcode,
                "rc": rc,
                "ctc_basecall_lev_primer_position": start_pos,
                "ctc_basecall_lev_seq_end": end_pos,
            }

            data_out.append(new_entry)
            with open(json_file_out, 'w') as f:
                json.dump(data_out, f, indent=4)
            pbar.update(1)

            #print('bonito: posteriors.shape', posteriors.shape)
            #posteriors.tofile(args.post_file)
            # writer.queue.put((read_id, posteriors))

    duration = time.perf_counter() - t0

    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
    sys.stderr.write(f"> skipped {skipped} reads\n")
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("fast5", help="Fast5 file containing reads")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--output_json_file",type=str,required=True)
    parser.add_argument("--start_barcode",type=str,required=True)
    parser.add_argument("--end_barcode",type=str,required=True)
    parser.add_argument("--min_len",type=int,required=True)
    #parser.add_argument("--post_file",type=str,required=True)
    return parser
