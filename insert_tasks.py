from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_nuc_merge_tasks, MinnieNucMergeTask
import numpy as np

nuc_file = "/Users/forrestc/ConnectomeStack/analysis_temp/p3minnie_nucleus_seg_info.df.csv.df"

df = pd.read_csv(nuc_file, dtype=np.int32)
nuc_seg_source = 'precomputed://https://s3-hpcrc.rc.princeton.edu/minnie65-phase3-ws/nuclei/v0/seg'
mip = 4
seg_source = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1'

df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest", n_threads=0)
tq.insert_all(create_nuc_merge_tasks(df_filt, nuc_seg_source, 
                                          seg_source,
                                          'gs://allen-minnie-phase3/minniev1-nuc-merge/'))
