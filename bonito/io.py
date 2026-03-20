"""
Bonito Input/Output
"""

import os
import sys
import h5py
from glob import glob
from textwrap import wrap
from multiprocessing import Process, Queue
import numpy as np
from tqdm import tqdm

from bonito.decode import decode
from bonito.util import get_raw_data, preprocess


class PreprocessReader(Process):
    """
    Reader Processor that reads and processes fast5 files
    """
    def __init__(self, directory, maxsize=1000):
        super().__init__()
        self.directory = directory
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for fast5 in tqdm(glob("%s/*fast5" % self.directory), ascii=True, ncols=100):
            for read_id, raw_data in get_raw_data(fast5):
                self.queue.put((read_id, raw_data))
        self.queue.put(None)

    def stop(self):
        self.join()

def get_fast5_raw_signal_from_hdf5_data(raw_data):
    raw_data = np.array(raw_data)
    # create fast5 (from https://nanoporetech.github.io/fast5_research/examples.html)
    # example of how to digitize data
    start, stop = int(min(raw_data - 1)), int(max(raw_data + 1))
    rng = stop - start
    digitisation = 8192.0
    bins = np.arange(start, stop, rng / digitisation)
    # np.int16 is required, the library will refuse to write anything other
    if raw_data.dtype == np.int16:
        raw_data_binned = raw_data
    else:
        raw_data_binned = np.digitize(raw_data, bins).astype(np.int16)
    
    scaling = rng / digitisation
    offset = int(0)
    scaled = np.array(scaling * (raw_data_binned + offset), dtype=np.float32)    
    return preprocess(scaled)


class PreprocessFileReader(Process):
    """
    Reader Processor that reads and process a given fast5 file
    """
    def __init__(self, fast5_file, maxsize=1000):
        super().__init__()
        self.fast5_file = fast5_file
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        if self.fast5_file.endswith('fast5'):
            for read_id, raw_data in get_raw_data(self.fast5_file):
                self.queue.put((read_id, raw_data))
        elif self.fast5_file.endswith('hdf5'):
            f5 = h5py.File(self.fast5_file,'r')
            for i, read_id in enumerate(f5.keys()):
                raw_data = f5[read_id]['raw_signal']
                raw_data_binned = get_fast5_raw_signal_from_hdf5_data(raw_data)
                self.queue.put((i, read_id, raw_data_binned))
        self.queue.put(None)

    def stop(self):
        self.join()

class DecoderWriter(Process):
    """
    Decoder Process that writes fasta records to stdout
    """
    def __init__(self, alphabet, beamsize=5, wrap=100):
        super().__init__()
        self.queue = Queue()
        self.wrap = wrap
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions = job
            sequence = decode(predictions, self.alphabet, self.beamsize)
            if sequence:
                sys.stdout.write(">%s\n" % read_id)
                sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, self.wrap)))
                sys.stdout.flush()
            else:
                sys.stderr.write("> skippingempty sequnece %s\n" % read_id)

    def stop(self):
        self.queue.put(None)
        self.join()
