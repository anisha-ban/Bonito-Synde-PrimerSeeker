import json, tempfile, os
from typing import Optional, Dict, Any, Tuple
import numpy as np
import datetime, pysam, random
import distance
import math
import numpy as np

def load_code(conv_code_path):
    """Load JSON files required for custom decoding"""

    conv_code = None
    if conv_code_path:
        with open(conv_code_path, 'r') as f:
            conv_code = json.load(f)

    return conv_code

def list_to_string(target_list):
    list_str = ''
    for i in target_list:
        list_str += str(i)
    return list_str

# reference extraction
def get_reference_alignment(bam_file, read_id):
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch():
            if read.query_name == read_id:
                return {
                        "reference_name": read.reference_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "cigar_string": read.cigarstring,
                        "mapping_quality": read.mapping_quality,
                        "is_reverse": read.is_reverse,
                            }
    return None

def read_fasta_to_dict(fa_file):
# Returns dictionary with sequence IDs as keys and sequences as values

    sequences = {}
    current_id = None
    current_sequence = []

    with open(fa_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('>'):
                # If we have a previous sequence, store it
                if current_id is not None:
                    sequences[current_id] = ''.join(current_sequence)

                current_id = line[1:]  # Remove '>'
                current_sequence = []

            elif line:  # Non-empty line (sequence data)
                current_sequence.append(line)

    # last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_sequence)

    return sequences

def find_primer_start(bam_file, read_id, reference, primer_sequence):
    # Get read alignment details
    alignment = get_reference_alignment(bam_file, read_id)
    if not alignment:
        return None  # Read not found

    # Extract reference sequence in the read-aligned region
    ref_seq = reference.fetch(alignment["reference_name"],
                                alignment["reference_start"],
                                alignment["reference_end"])

    # Find primer in this sequence
    primer_index = ref_seq.find(primer_sequence)
    if primer_index == -1:
        return None  # Primer not found in this region

    primer_start = alignment["reference_start"] + primer_index
    return primer_start

def get_primer_position_basecall_align(sequence, path, primer):
    r"""
    Given a basecalled sequence 'sequence', the array of base
    transition indices in the raw signal 'path' and a target
    sequence 'primer', this method seeks that part of 'sequence'
    which is at least Levenshtein distance from 'primer'.

    It returns this minimum edit distance, and that element in path
    which corresponds to where the first base transition of the
    erroneous version of 'primer', occurred in the raw signal.
    """
    basecall_length = len(sequence)
    primer_length = len(primer)
    extend_len = 25 # used by Chandak et al
    extend_penalty = 1.0 # used by Chandak et al
    one_minus_extend_penalty = 1.0-extend_penalty # used by Chandak et al

    edit_distances = []
    # levenshtein distance of each window with target primer
    for i in range(-extend_len,0):
        edit_distances.append(distance.levenshtein(primer, sequence[:primer_length + i]) + i * one_minus_extend_penalty)
    for i in range(basecall_length - primer_length):
        edit_distances.append(distance.levenshtein(primer, sequence[i:i+primer_length]))

    # pick that which minimizes LD
    min_edit_distance = min(edit_distances)
    minimizing_LD_index = edit_distances.index(min_edit_distance) - extend_len
    start_pos = path[minimizing_LD_index + 1] - 1

    return start_pos, min_edit_distance, minimizing_LD_index

#  --------------------------- marker codes --------------------------------
class MarkerCode:
    def __init__(self, marker, k_marker):
        self.marker = marker
        self.k_marker = k_marker

    def get_random_marker_codeword(self, n_block_marker, marker_int, payload_len):
        codeblock_length = n_block_marker + len(marker_int)
        cw =  [random.randint(0, 3) if ((i % codeblock_length) < n_block_marker) else marker_int[(i % codeblock_length) - n_block_marker] for i in range(payload_len)]
        return cw

    def insert_markers(self, cw, num_marker_insertions, payload_length):
        est_codeword = cw[:self.k_marker] + self.marker
        for jj in range(1, num_marker_insertions):
            start = jj * self.k_marker
            est_codeword = est_codeword + cw[start : start + self.k_marker] + self.marker
        if len(est_codeword) < payload_length:
            est_codeword = est_codeword + cw[num_marker_insertions * self.k_marker:]
        assert len(est_codeword) == payload_length
        return est_codeword
    def validate_markers(self, cw):
        marker_len = len(self.marker)
        cw_len = len(cw)
        num_marker_appearances = math.ceil(cw_len / (self.k_marker + marker_len))
        cw = list(cw)
        for i in range(num_marker_appearances):
            substring = cw[i * (self.k_marker + marker_len) + self.k_marker: min(cw_len, (i + 1) * (self.k_marker + marker_len))]
            if substring != self.marker[:len(substring)]:
                return False
        return True
    def remove_markers(self, cw):
        marker_len = len(self.marker)
        cw_len = len(cw)
        cw = list(cw)
        temp_codeword = []
        for i in range(math.ceil(cw_len / (self.k_marker + marker_len))):
            temp_codeword = temp_codeword + cw[i * (self.k_marker + marker_len) : min(cw_len, i * (self.k_marker + marker_len) + self.k_marker)]
        return temp_codeword

def _dec2bi(self, decimal, num_bits):
    """Convert decimal to binary list (LSB first)."""
    binary = []
    for _ in range(num_bits):
        binary.append(bool(decimal & 1))
        decimal >>= 1
    return binary

# ---------------------------- convolutional codes -------------------------
class ConvCode:
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        self.cc_file = json_file
        self.numEdgesBin = data["numInputs_binary"]
        self.allowedEdges = data["allowedEdges"]
        self.allowedEdges_term = data["allowedEdges_term"]
        self.c, self.b, self.K, self.num_term_symbols = data["c"], data["b"], data["K"], data["num_term_code_symbols"]
        self.c_bin = len(self.numEdgesBin)

    def is_codeword(self, codeword):
        if isinstance(codeword, str):
            codeword = [int(el) for el in codeword]
        cur_state = 0
        is_codeword = True
        num_non_term_symbols = len(codeword) - self.num_term_symbols
        for t in range(len(codeword)):
            sym = codeword[t]
            sym_found = False

            if t < num_non_term_symbols:
                allowed_edges = self.allowedEdges[t % self.c][str(cur_state)]
            else:
                allowed_edges = self.allowedEdges_term[t - num_non_term_symbols][str(cur_state)]
            for el in allowed_edges:
                if sym == el[0]:
                    sym_found = True
                    cur_state = el[1]
                    break

            if not sym_found:
                return False
        return True

    def encode(self, msg):

        msg_len = len(msg)
        num_codeword_symbols = (msg_len * self.c // self.b) + self.num_term_symbols

        cw = [0] * num_codeword_symbols
        ext = 2

        # Convert message symbols to binary
        dec_map = {0: '00', 1: '01', 2:'10', 3:'11'}
        msg_bin = ''.join([dec_map[el] for el in msg])
        msg_bin = [1 if ch == '1' else 0 for ch in msg_bin]
        msg_bin_track = 0
        syn_cs = 0  # syndrome state

        # Encode information blocks
        for t in range(num_codeword_symbols - self.num_term_symbols):
            # Determine which bit positions are information levels
            indices = []
            for j in range(ext):
                ## Check if this position has 2 allowed inputs (i.e., is an information level)
                if self.numEdgesBin[(ext * t + j) % self.c_bin] == 2:
                    indices.append(j)

            if not indices:
                # No information bits at this level, use first allowed edge
                cw[t] = self.allowedEdges[t % self.c][str(syn_cs)][0]
            else:
                # Find which input matches the information bits
                for in_val in self.allowedEdges[t % self.c][str(syn_cs)]:
                    ns = in_val[1]
                    in_val = in_val[0]

                    dummy = [int(ch) for ch in dec_map[in_val]]
                    flag = True

                    for count, ip in enumerate(indices):
                        if dummy[ip] != msg_bin[msg_bin_track + count]:
                            flag = False
                            break
                    if flag:
                        input_val = in_val
                        next_state = ns
                        break

                cw[t] = input_val
                syn_cs = next_state # Update syndrome state
                msg_bin_track += len(indices)

        # Termination
        for j in range(num_codeword_symbols - self.num_term_symbols, num_codeword_symbols):
            i = j - (num_codeword_symbols - self.num_term_symbols)
            entry = self.allowedEdges_term[i][str(syn_cs)][0]
            cw[j] = entry[0]
            syn_cs = entry[1]

        return cw

    def get_random_cc_codeword(self, cw_len):

        assert (cw_len - self.num_term_symbols) % self.c == 0, f'incompatible codeword length. {self.num_term_symbols} terminating symbols'
        msg_len = ((cw_len - self.num_term_symbols) // self.c) * self.b

        symbols = ['0', '1', '2', '3']
        msg = ''.join(random.choices(symbols, k=msg_len))

        #cmd = f"conv_code/encode -file {self.cc_file} -msg {msg}"
        #output = os.popen(cmd).read().strip()
        #cw =  list(map(int, output.split()))
        cw = self.encode([int(ch) for ch in msg])

        return msg, cw

    def get_info_vec(self, codeword):
        cw_str = ''.join(map(str,codeword))
        cmd = f"conv_code/get_info -file {self.cc_file} -cw {cw_str}"
        output = os.popen(cmd).read().strip()
        info_vec = [int(digit) for digit in output]
        return info_vec


def get_code_params(args, base_dict):
    code_params = ""
    payload_length = args.payload_length
    conv_codeword_length = 0
    if args.code == 'marker':
        mark_code = MarkerCode([base_dict[base] for base in args.marker], args.marker_interval)
        code_params = f"{args.marker_interval}_{args.marker}"
        num_marker_symbols = math.floor(args.payload_length / args.marker_interval) * len(args.marker)
        rate = args.marker_interval / (args.marker_interval + len(args.marker))
    elif args.code == 'conv':
        num_marker_symbols = 0
        cc_code = ConvCode(args.conv_code)
        #total_states, numInputs, allowed_edges, term_cs, term_inp, term_ns, term_offsets = cc_code.process_conv_json()
        code_params = f"{cc_code.c}_{cc_code.b}_{cc_code.K}"
        if (payload_length - cc_code.num_term_symbols) % cc_code.c != 0:
            payload_length += cc_code.c - ((payload_length - cc_code.num_term_symbols) %  cc_code.c)
        assert (payload_length - cc_code.num_term_symbols) % cc_code.c == 0
        conv_codeword_length = payload_length
        rate = ((payload_length - cc_code.num_term_symbols) * cc_code.b // cc_code.c) / payload_length
        print(f"Payload_length = cw length = {payload_length}, rate={rate}")
    elif args.code == 'conv+marker':
        # conv code
        cc_code = ConvCode(args.conv_code)
        #total_states, numInputs, allowed_edges, term_cs, term_inp, term_ns, term_offsets = cc_code.process_conv_json()
        # marker code
        mark_code = MarkerCode([base_dict[base] for base in args.marker],args.marker_interval)
        code_params = f"cc:{cc_code.c}_{cc_code.b}_{cc_code.K}_marker:{args.marker_interval}_{args.marker}"
        # determine payload length
        marker_block_length = args.marker_interval + len(args.marker)
        conv_codeword_length = math.floor(payload_length * args.marker_interval / marker_block_length)
        if (conv_codeword_length - cc_code.num_term_symbols) % cc_code.c != 0:
            conv_codeword_length += cc_code.c - ((conv_codeword_length - cc_code.num_term_symbols) %  cc_code.c)
        assert (conv_codeword_length - cc_code.num_term_symbols) % cc_code.c == 0
        num_marker_symbols = math.floor(conv_codeword_length / args.marker_interval) * len(args.marker)
        payload_length =  conv_codeword_length + num_marker_symbols
        rate = ((conv_codeword_length - cc_code.num_term_symbols) * cc_code.b // cc_code.c) / payload_length
        print(f"payload_length={payload_length}. rate={rate}, cw length={conv_codeword_length}, markers: {num_marker_symbols}")
    elif PARAMS.code == 'base':
        rate = 1
    return payload_length, conv_codeword_length, num_marker_symbols, rate, code_params


def get_init_entry(args, code_params, rate, payload_length, conv_codeword_length, num_marker_symbols):
    if args.code == "conv":
        sim_results = {
                    "numIterations" : 0,
                    "frameErrors" : 0,
                    "symbolErrors" : 0,
                    "FER" : 0.0,
                    "SER":0.0,
                    "datetime" : str(datetime.datetime.now()),
                    "f5_file": str(args.fast5),
                    "primer_search": str(args.primer_search),
                    "num_beams": int(args.beam),
                    "code" : args.code,
                    "code_params" : code_params,
                    "rate" : rate,
                    "payload_length" : payload_length,
                    "correct_decoding_scores" : [],
                    "misdecoding_scores" : [],
                    "hamming_distance" :  [],
                    "lev_distance" :  []
                }
    elif args.code == 'marker':
        sim_results = {
                        "numIterations" : 0,
                        "frameErrors" : 0,
                        "symbolErrors" : 0,
                        "FER" : 0.0,
                        "SER":0.0,
                        "datetime" : str(datetime.datetime.now()),
                        "f5_file": str(args.fast5),
                        "primer_search": str(args.primer_search),
                        "num_beams": int(args.beam),
                        "code" : args.code,
                        "code_params" : code_params,
                        "rate" : rate,
                        "payload_length" : payload_length,
                        "num_markers" : num_marker_symbols,
                        "correct_decoding_scores" : [],
                        "misdecoding_scores" : [],
                        "hamming_distance" :  [],
                        "lev_distance" :  []
                    }
    elif args.code == 'conv+marker':
        sim_results = {
                        "numIterations" : 0,
                        "frameErrors" : 0,
                        "symbolErrors" : 0,
                        "FER" : 0.0,
                        "SER":0.0,
                        "datetime" : str(datetime.datetime.now()),
                        "f5_file": str(args.fast5),
                        "primer_search": str(args.primer_search),
                        "num_beams": int(args.beam),
                        "code" : args.code,
                        "code_params" : code_params,
                        "rate" : rate,
                        "payload_length" : payload_length,
                        "num_markers" : num_marker_symbols,
                        "conv_codeword_length" : conv_codeword_length,
                        "correct_decoding_scores" : [],
                        "misdecoding_scores" : [],
                        "hamming_distance" :  [],
                        "lev_distance" :  []
                    }
    return sim_results

def get_random_codeword(args, payload, cc_code, mark_code, conv_codeword_length, base_dict):
    payload_length = len(payload)
    if args.code == 'marker':
        marker_int = [base_dict[base] for base in args.marker]
        cw = mark_code.get_random_marker_codeword(args.marker_interval, marker_int, payload_length)
        assert mark_code.validate_markers(cw), f'{cw} is not a marker codeword'
    elif args.code == 'conv':
        msg, cw = cc_code.get_random_cc_codeword(conv_codeword_length)
        assert cc_code.is_codeword(cw), f'{cw} not conv codeword'
    elif args.code == 'conv+marker':
        msg, cw_no_marker = cc_code.get_random_cc_codeword(conv_codeword_length)
        assert cc_code.is_codeword(cw_no_marker), 'Not conv codeword'
        num_marker_insertions = math.floor(conv_codeword_length / args.marker_interval)
        cw = mark_code.insert_markers(cw_no_marker, num_marker_insertions, payload_length)
    elif args.code == 'base':
        cw = payload_int

    return cw

# --------------------- chandak helper --------------------------------
def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join([complement[base] for base in dna[::-1]])

def lev_align(A, B):
    """
    Align short string B to long string A, using Lev dist computations
    Returns (start_idx, end_idx, edit_distance)
    """
    m, n = len(B), len(A)
    D = np.zeros((m+1, n+1), dtype=int) # short matrix

    # init
    for i in range(1, m+1):
        D[i, 0] = i
    for j in range(1, n+1):
        D[0, j] = 0   # free start anywhere in A

    # DP
    for i in range(1, m+1):
        for j in range(1, n+1):
            D[i, j] = min(
                D[i-1, j] + 1,
                D[i, j-1] + 1,
                
                D[i-1, j-1] + (B[i-1] != A[j-1])
            )

    # best end
    end_j = np.argmin(D[m, :])
    dist = D[m, end_j]

    # traceback to find start
    i, j = m, end_j
    while i > 0:
        if j > 0 and D[i, j] == D[i-1, j-1] + (B[i-1] != A[j-1]):
            i -= 1; j -= 1
        elif j > 0 and D[i, j] == D[i, j-1] + 1:
            j -= 1
        else:
            i -= 1

    start_j = j
    return start_j, end_j - 1, dist


def find_barcode_pos_in_posteriors_stanford(posteriors, start_barcode, end_barcode, extend_len=25, extend_penalty=0.6):
    '''
    find position of best edit distance match for barcodes in the post matrix
    looks at fastq to find the best match for barcode_start and barcode_end and then finds
    corresponding entries in trans_filename. Returns a tuple (start_pos,end_pos) which represents
    start and end position of actual payload in the post matrix (both inclusive, zero-indexed).
    One could then slightly extend these or not, depending on what works best.
    If things fail, return (-1,-1)
    extend_len: extra length at start and end to search for barcode match (useful if basecaller cuts
    a bit of the barcode, e.g., with guppy)
    extend_penalty: float between 0.0 and 1.0 telling the penalty we impose per base on extend operation (i.e., barcode hanging off the sides). Setting to 1 just means we compute edit distance as it is (this can penalize a bit much and miss perfect match of say 10 bases out of 25 base barcode). Setting to 0 is bad because then there is chance that the empty match (completely hanging off) is selected.
    '''
    assert extend_len >= 0
    assert 0.0 <= extend_penalty <= 1.0
    one_minus_extend_penalty = 1.0-extend_penalty

    # Perform CTC decoding (greedy)
    labels = ["N", "A", "C", "G", "T"]
    basecall = ""
    trans_arr = []  # timesteps when new base arrives
    prev_base = 'N'

    for i in range(posteriors.shape[0]):
        next_base = labels[np.argmax(posteriors[i, :])]
        if next_base != prev_base:
            prev_base = next_base
            if next_base != 'N':
                trans_arr.append(i)
                basecall = basecall + next_base

    basecall_len = len(basecall)
    start_barcode_len = len(start_barcode)
    end_barcode_len = len(end_barcode)

    if start_barcode_len + end_barcode_len > basecall_len:
        return ([-1], -1, np.inf, np.inf)

    start_bc_edit_distance = []
    for i in range(-extend_len,0):
        start_bc_edit_distance.append(distance.levenshtein(start_barcode,basecall[:start_barcode_len+i])+i*one_minus_extend_penalty)
    for i in range(basecall_len-start_barcode_len):
        start_bc_edit_distance.append(distance.levenshtein(start_barcode,basecall[i:i+start_barcode_len]))

    # find best match positions
    start_dist = min(start_bc_edit_distance)
    start_bc_first_base = max(start_bc_edit_distance.index(start_dist) - extend_len, 0) # starting position, may be negative!!
    start_bc_last_base = start_bc_first_base + start_barcode_len - 1

    end_bc_edit_distance = []
    for i in range(start_bc_last_base,basecall_len - end_barcode_len + 1): # end_bc must be after start_bc
        end_bc_edit_distance.append(distance.levenshtein(end_barcode,basecall[i:i+end_barcode_len]))
    for i in range(1, extend_len + 1):
        end_bc_edit_distance.append(distance.levenshtein(end_barcode,basecall[basecall_len-end_barcode_len+i:basecall_len])-i*one_minus_extend_penalty)

    # find best match positions
    # need sanity check in case range(start_bc_last_base,basecall_len-end_barcode_len) is empty
    if start_bc_last_base >= basecall_len - end_barcode_len:
      end_bc_first_base = basecall_len - end_barcode_len + 1 + end_bc_edit_distance.index(min(end_bc_edit_distance))
    else:
      end_bc_first_base = start_bc_last_base + end_bc_edit_distance.index(min(end_bc_edit_distance))

    start_pos = trans_arr[start_bc_last_base + 1] - 1
    end_pos = trans_arr[end_bc_first_base - 1] - 1

    if end_pos < start_pos:
        return ([-1], -1, np.inf, np.inf)
    return (trans_arr[start_bc_first_base:start_bc_last_base + 1], end_pos, start_dist, min(end_bc_edit_distance))

def find_barcode_pos_in_posteriors_ps(posteriors, start_barcode, end_barcode,
                                   extend_len=25, extend_penalty=0.6):
    '''
    Find position of best edit distance match for barcodes in the posterior matrix.
    Works directly with the posteriors array without file I/O.

    Args:
        posteriors: numpy array of shape (timesteps, 5) with probabilities
        start_barcode: string with start barcode sequence
        end_barcode: string with end barcode sequence
        extend_len: extra length at start and end to search for barcode match
        extend_penalty: penalty per base on extend operation (0.0 to 1.0)

    Returns:
        tuple: (start_pos, end_pos, dist_start, dist_end)
               start_pos, end_pos are indices in the posterior matrix (inclusive, zero-indexed)
               dist_start, dist_end are edit distances for the barcode matches
    '''


    assert extend_len >= 0
    assert 0.0 <= extend_penalty <= 1.0
    one_minus_extend_penalty = 1.0 - extend_penalty

    # Perform CTC decoding (greedy)
    labels = ["N", "A", "C", "G", "T"]
    basecall = ""
    trans_arr = []  # timesteps when new base arrives
    prev_base = 'N'

    for i in range(posteriors.shape[0]):
        next_base = labels[np.argmax(posteriors[i, :])]
        if next_base != prev_base:
            prev_base = next_base
            if next_base != 'N':
                trans_arr.append(i)
                basecall = basecall + next_base

    basecall_len = len(basecall)
    start_barcode_len = len(start_barcode)
    end_barcode_len = len(end_barcode)

    if start_barcode_len + end_barcode_len > basecall_len:
        return ([-1], -1, np.inf, np.inf)

    # Find start barcode
    r"""
    start_bc_edit_distance = []
    for i in range(-extend_len, 0):
        start_bc_edit_distance.append(
            distance.levenshtein(start_barcode, basecall[:start_barcode_len+i]) +
            i * one_minus_extend_penalty
        )
    for i in range(basecall_len - start_barcode_len):
        start_bc_edit_distance.append(
            distance.levenshtein(start_barcode, basecall[i:i+start_barcode_len])
        )

    start_bc_first_base = start_bc_edit_distance.index(min(start_bc_edit_distance))
    if start_bc_first_base < extend_len:
        start_bc_last_base = start_bc_first_base
        start_bc_first_base = 0
    else:
        start_bc_first_base -= extend_len
        start_bc_last_base = start_bc_first_base + start_barcode_len - 1
    """
    start_bc_first_base, start_bc_last_base, start_dist = lev_align(basecall, start_barcode)
    assert start_bc_first_base < start_bc_last_base, "invalid lev-dist alignment"
    assert start_dist >= 0, "invalid lev-dist"
    # Find end barcode
    end_bc_edit_distance = []
    for i in range(start_bc_last_base, basecall_len - end_barcode_len + 1):
        end_bc_edit_distance.append(
            distance.levenshtein(end_barcode, basecall[i:i+end_barcode_len])
        )
    for i in range(1, extend_len + 1):
        end_bc_edit_distance.append(
            distance.levenshtein(end_barcode, basecall[basecall_len-end_barcode_len+i:basecall_len]) -
            i * one_minus_extend_penalty
        )

    # Find best match positions
    if start_bc_last_base >= basecall_len - end_barcode_len:
        end_bc_first_base = basecall_len - end_barcode_len + 1 + end_bc_edit_distance.index(min(end_bc_edit_distance))
    else:
        end_bc_first_base = start_bc_last_base + end_bc_edit_distance.index(min(end_bc_edit_distance))

    start_pos_l = trans_arr[start_bc_first_base]
    start_pos_r = trans_arr[start_bc_last_base]
    end_pos = trans_arr[end_bc_first_base - 1] - 1

    if end_pos < start_pos_r or start_pos_r < start_pos_l:
        return ([-1], -1, np.inf, np.inf)

    #print('\nstart_bc_first_base: ', start_bc_first_base, ', start_bc_last_base: ', start_bc_last_base, ', end_bc_first_base: ', end_bc_first_base)
    #return (start_pos_l, start_pos_r, end_pos, min(start_bc_edit_distance), min(end_bc_edit_distance))
    return (trans_arr[start_bc_first_base:start_bc_last_base+1], end_pos, start_dist, min(end_bc_edit_distance))
