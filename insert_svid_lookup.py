from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_sv_lookup_tasks, MinnieSupervoxelLookup
import numpy as np

nuc_file = "/Users/forrestc/ConnectomeStack/analysis_temp/p3minnie_nucleus_seg_info.df.csv.df"

df = pd.read_csv(nuc_file, dtype=np.int32)

seg_source = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1'

df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest-nucfind", n_threads=0)
tq.insert_all(create_sv_lookup_tasks(df_filt, seg_source,
                                     'gs://allen-minnie-phase3/minniev1-nuc-svids/'))
