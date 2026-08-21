# Bonito

- Forked from [Bonito](https://github.com/nanoporetech/bonito) (v0.1.2) by ONT
- Based on the work `SynDe: Syndrome--guided Decoding of Raw Nanopore Reads` [Arxiv version](https://arxiv.org/abs/2604.01054)
- The `my-extension` branch of this repository includes the Bonito-based implementations (v0.1.2) of our novel algorithms PrimerSeeker and Synde.
    - PrimerSeeker: a dedicated algorithm that locates the start of a primer in the raw read
    - Synde: a solution for basecaller-decoder integration that performs convolutional decoding by performing a constrained beam search -  one that exploits the syndrome trellis representation of the concerned convolutional code. Its main advantage is that its complexity is independent of the memory of the convolutional code.
- Thanks to [Roman Sokolovskii](https://github.com/rsokolovskii) for contributing to this project!!

## Download and installation

Needs our custom `fast-ctc-decode`:
```bash
git clone --recursive -b my-extension https://github.com/anisha-ban/fast-ctc-decode-synde-primerseeker
cd fast-ctc-decode-synde-primerseeker/
pip install "maturin>=0.14,<0.15"
python -m maturin build --release --features python # this should create a folder `target` named target with the required wheel file
pip install target/wheels/*.whl --force-reinstall
cd ..
```

Now install Bonito:
```bash
git clone --recursive -b my-extension https://github.com/anisha-ban/Bonito-Synde-PrimerSeeker
cd Bonito-Synde-PrimerSeeker/
pip install --no-build-isolation -r requirements.txt
pip install -e .
cd ..
```

## Usage

- The original functionality (basecaller, view, ...) are preserved and the information on the use of these functions can be found in `Bonito-old-readme.md`.


### PrimerSeeker

Takes as inputs:
- `model`: neural network model to be used - depends on the MinION kit used for sequencing
- `f5_file`: a fast5 file or an HDF5 file (created in [nanopore_dna_storage](https://github.com/shubhamchandak94/nanopore_dna_storage) repository by Chandak et al.)
- `sim_targets_file`: a JSON file containing entries, each of which lists the read ID of the raw signal to be scanned, along with the primer sequence to be located.
- `beam`: max number of beams to be maintained during beam search
- `shift`: minimum separation between any two candidate starting positions that are examined during a single round of beam search.
- `subsample`, `conc_thresh` refer to the _subsampling factor_ and the _concentration threshold_, two user-defined parameters used to speed up PrimerSeeker. The former is essentially responsible for filtering out the most promising starting position for a primer in each window of `subsample` contiguous samples in the raw read. `conc_thresh` on the other hand is an additional pruning step that gets rid of the Z least-likely beams (Z<beam), if the top (beam-Z) beams contain at least 100*`conc_thresh` percent of the total sum of probability scores over all the beams.

```bash
model=dna_r9.4.1
FAST5_FILE=test_files/raw_data/raw_signal_1.hdf5
SIM_TARGETS_FILE=test_files/test_input_sim_targets/test_2020_raw_signal_1.json
BEAM=6
SUBSAMPLE=15
RESULTS_FILE=test_files/test_output/sim_raw_signal_1_B${BEAM}_SS${SUBSAMPLE}.json
CONC_THRESH=0.98

bonito primer_search "$model" $FAST5_FILE --sim_targets $SIM_TARGETS_FILE \
                    --results_file $RESULTS_FILE \
                    --version opt --shift 100 --beam $BEAM \
                    --subsample $SUBSAMPLE --conc_thresh $CONC_THRESH --device cpu;

basename=batch_1667_strand14_batch1
f5_file=test_files/raw_data/$basename.fast5
sim_targets_file=test_files/test_input_sim_targets/${basename}_25.json
bonito primer_search "$model" $f5_file --sim_targets $sim_targets_file \
                --results_file $sim_targets_file --primer_search_method ctc \
                --version opt  --shift 100 --beam 8 --subsample 6 --conc_thresh 0.98 \
                --start_offet 0 \
                --device cuda;
```

### Synde

Much like before, takes as inputs:
- `model`: neural network model to be used - depends on the MinION kit used for sequencing
- `f5_file`: a fast5 file or an HDF5 file (created in [nanopore_dna_storage](https://github.com/shubhamchandak94/nanopore_dna_storage) repository by Chandak et al.)
- `sim_targets_file`: a JSON file containing entries, each of which lists the read ID of the raw signal to be scanned, along with the leading and trailing primer sequences that surround the payload.
- `payload_length`: length of the codeword (or the payload)
- `beam`: max number of beams to be maintained during beam search
- `bam_file`: If the f5_file is not an HDF5 file, we need the BAM file (obtained by basecalling and then performing alignment with the reference strands).
- `ref_file`: contains the original set of synthesized DNA strands.
- `code`: which encoding scheme is used: either `conv` if only convolutional code. Else `conv+marker` if a marker code is also used.
- `cc_code_file`: a JSON file that stores the syndrome trellis representation of the convolutional code to be used.
- `marker_interval` and `marker` specify the parameters of the marker code: the former indicates the interval at which the marker symbol/sequence is to be inserted and the latter represents the fixed DNA symbol/sequence that is inserted periodically.

```bash
FAST5_FILE=test_files/raw_data/raw_signal_1.hdf5
REF_FILE=test_files/raw_data/oligos_8_4_20/oligos_1.fa
SIM_TARGETS_FILE=test_files/test_input_sim_targets/test_2020_raw_signal_1.json
CONV_CODE_FILE=test_files/conv_code/cc_4_3_9.json
RESULTS_FILE=test_files/test_output/decode_raw_signal_1_cc439.json

bonito my_decoder "dna_r9.4.1" "$FAST5_FILE" \
                --ref_file $REF_FILE --sim_targets $SIM_TARGETS_FILE \
                --code "conv" \
                --conv_code $CONV_CODE_FILE \
                --results_file $RESULTS_FILE --primer_search ctc \
                --payload_length 114 --beam 512;



basename=batch_1667_strand14_batch1
f5_file=test_files/raw_data/$basename.fast5
sim_targets_file=test_files/test_input_sim_targets/${basename}_25.json

ref_file=test_files/ref_file/references.fasta
bam_file=test_files/bam_file/$basename.sorted.bam

cc_code=cc_4_3_9.json
cc_code_file=test_files/conv_code/$cc_code
primer_search=ctc

results_file="~/output/decode/test.json"

# only convolutional code
bonito my_decoder "dna_r9.4.1" "$f5_file" --bam_file $bam_file \
                --ref_file $ref_file --sim_targets $sim_targets_file \
                --code "conv" --conv_code $cc_code_file \
                --results_file $results_file --primer_search $primer_search \
                --payload_length 110;

# convolutional code + 1 marker symbol every 5 symbols
marker_int=5
bonito my_decoder "dna_r9.4.1" "$f5_file" --ref_file $ref_file \
                --bam_file $bam_file --sim_targets $sim_targets_file \
                --code "conv+marker" --marker_interval $marker_int --marker C \
                --conv_code $cc_code_file \
                --results_file $results_file --primer_search $primer_search \
                --payload_length 110;
```


### Basecall complexity

Takes as inputs:
- a fast5 or an hdf5 file (created in [nanopore_dna_storage](https://github.com/shubhamchandak94/nanopore_dna_storage) repository by Chandak et al.)
- a sim_targets_file of JSON type that lists the read IDs correpsonding to the raw signals that should be basecalled.

This function measures the 'mean beam complexity', i.e., the average number of beam extensions performed for per column of the CTC matrix produced for the raw signals, and stores this quantity in a file named `complexity_bs.txt` in the same folder as sim_targets_file.

```bash
f5_file=test_files/raw_data/raw_signal_1.hdf5
sim_targets_file=test_files/test_input_sim_targets/test_2020_raw_signal_1.json
bonito basecall_complexity dna_r9.4.1 $f5_file --sim_targets $sim_targets_file --device cuda --beam 5
```



