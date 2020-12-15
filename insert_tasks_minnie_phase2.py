from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_nuc_merge_tasks, MinnieNucMergeTask
import numpy as np

nuc_file = "/Users/forrestc/ConnectomeStack/analysis_temp/p2minnie_nucleus_seg_info.df.csv.df"
df = pd.read_csv(nuc_file)

nuc_seg_source = 'precomputed://https://s3-hpcrc.rc.princeton.edu/minnie65-phase2-seg/nuclei/v0/seg'
mip = 4
seg_source = 'precomputed://https://storage.googleapis.com/microns-seunglab/minnie65/seg_minnie65_0'

df_filt = df[df['size']>200000].reset_index()

#df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest-nucfind", n_threads=0)
tq.insert_all(create_nuc_merge_tasks(df_filt, nuc_seg_source, 
                                          seg_source,
                                          'gs://allen-minnie-phase3/minniephase2-nuc-merge/',
                                          find_merge=False))
