import pdb

import numpy as np
from streaming import StreamingDataset

# Define the path or remote storage location of your MDS dataset
dataset_path = "data"

# Define your MDS dataset using Mosaic Streaming
mds_dataset = StreamingDataset(
    local=dataset_path,       # Path to the local MDS files
    remote=None,              # Optional remote storage (S3, GCS, etc.)
    shuffle=False,             # Whether to shuffle data
    batch_size=1,            # Batch size for loading data
)

# Iterate over the dataset to get individual samples
for i, sample in enumerate(mds_dataset):
    print(i)
    num_eos = len(np.where(sample['input_ids'] == 128001)[0])
    if num_eos > 0:
        pdb.set_trace()

    num_bos = len(np.where(sample['input_ids'] == 128000)[0])
    if num_bos > 1:
        pdb.set_trace()