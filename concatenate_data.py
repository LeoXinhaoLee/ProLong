import pdb

import os
import os.path as osp
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from streaming import StreamingDataset


# Define the path or remote storage location of your MDS dataset
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/arxiv"  # len: 2048
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/stackexchange"  # len: 2048
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/tuluv2"  # len: 512
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/fineweb-edu"  # len: 12288
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/fineweb-2023-50"  # len: 12288, domain = '.'?
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/thestackv1_concat_by_repo-65536"  # len: 6144
# dataset_path = "/juice5/scr5/nlp/mttt/datasets/prolong-data-512K/thestackv1_concat_by_repo-524288"  # len: 6144

seed = 42
random.seed(seed)
np.random.seed(seed)

data_dir = Path("/juice5/scr5/nlp/mttt/datasets/prolong-data-512K")
# domains = [
    # "thestackv1_concat_by_repo-65536",  # 13G
    # "tuluv2",  # 1G
    # "fineweb-edu",  # 25G
    # "textbooks",  # 850M
    # "dolmawiki",  # 4G
    # "arxiv",  # 4G
    # "book-524288", # 8G
    # "stackexchange",  # 4G
# ]  # 60G
#
# domains = [
#     "book-65536",  # 17G
#     "thestackv1_concat_by_repo-524288",  # 13G
#     "fineweb-2023-50",  # 25G
#     "openwebmath",  # 4G
# ]  # 59G

domains = [
    "thestackv1_concat_by_repo-65536",  # 13G
    "tuluv2",  # 1G
    "fineweb-edu",  # 25G
    "textbooks",  # 850M
    "dolmawiki",  # 4G
    "arxiv",  # 4G
    "book-524288", # 8G
    "stackexchange",  # 4G
    "book-65536",  # 17G
    "thestackv1_concat_by_repo-524288",  # 13G
    "fineweb-2023-50",  # 25G
    "openwebmath",  # 4G
]  # 115G


# output_file = '/juice5/scr5/nlp/mttt/datasets/prolong-data-512K-concat'
# os.makedirs(output_file, exist_ok=True)
# output_file = osp.join(output_file, 'train.npy')
# batch = []
# batch_size = 512
# with open(output_file, 'wb') as f_out:
#     for domain in tqdm(domains, desc='Processing domains'):
#         dataset_path = data_dir / domain
#         mds_dataset = StreamingDataset(
#             local=dataset_path,
#             remote=None,
#             shuffle=False,
#             batch_size=1,
#         )
#
#         batch = []
#         for i, sample in tqdm(enumerate(mds_dataset), total=len(mds_dataset), desc='Processing samples'):
#             input_ids = sample['input_ids']
#             batch.append(input_ids)
#             if len(batch) >= batch_size:
#                 concatenated_batch = np.concatenate(batch, axis=0)
#                 np.save(f_out, concatenated_batch)
#                 batch.clear()
#
#         if batch:
#             concatenated_batch = np.concatenate(batch, axis=0)
#             np.save(f_out, concatenated_batch)

output_file = '/juice5/scr5/nlp/mttt/datasets/prolong-data-512K-concat'
os.makedirs(output_file, exist_ok=True)
output_file = osp.join(output_file, 'train.h5')

batch_size = 1024

# Create or open the HDF5 file
with h5py.File(output_file, 'a') as f_out:

    # Check if the dataset exists, if not, create it
    if 'train_dataset' not in f_out:
        dset = f_out.create_dataset('train_dataset', shape=(0,), maxshape=(None,), dtype='uint32')
    else:
        dset = f_out['train_dataset']

    # Iterate through the domains and process the data
    with tqdm(domains, desc='Processing domains') as domain_progress:
        for domain in domain_progress:
            dataset_path = data_dir / domain
            mds_dataset = StreamingDataset(
                local=dataset_path,
                remote=None,
                shuffle=False,
                batch_size=1,
            )

            batch = []
            for i, sample in tqdm(enumerate(mds_dataset), total=len(mds_dataset), desc='Processing samples'):
                input_ids = sample['input_ids']
                batch.append(input_ids)

                # When the batch reaches the specified size, append it to the HDF5 file
                if len(batch) >= batch_size:
                    concatenated_batch = np.concatenate(batch, axis=0)

                    # Resize the dataset to accommodate the new data
                    dset.resize((dset.shape[0] + concatenated_batch.shape[0],))
                    dset[-concatenated_batch.shape[0]:] = concatenated_batch

                    batch.clear()

            # Append any remaining data in the last batch
            if batch:
                concatenated_batch = np.concatenate(batch, axis=0)
                dset.resize((dset.shape[0] + concatenated_batch.shape[0],))
                dset[-concatenated_batch.shape[0]:] = concatenated_batch
