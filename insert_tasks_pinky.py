from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_nuc_merge_tasks, MinnieNucMergeTask
import numpy as np

nuc_file = "pinky100_v185_alyssa_sheet_cleaned.pkl"
df = pd.read_pickle(nuc_file)
nuc_seg_source = 'precomputed://gs://neuroglancer/pinky100_v0/nucleus/seg'
mip = 4
seg_source = 'precomputed://gs://microns_public_datasets/pinky100_v185/seg'

#df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest-nucfind", n_threads=0)
tq.insert_all(create_nuc_merge_tasks(df, nuc_seg_source, 
                                          seg_source,
                                          'gs://allen-minnie-phase3/pinky100-nuc-merge/', find_merge=False))
