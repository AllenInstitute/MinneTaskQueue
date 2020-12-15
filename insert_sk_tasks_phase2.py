from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_skeletonize_tasks, MinnieSkeletonizeNeuron
import numpy as np

df = pd.read_pickle('phase2_100cells.pkl')

pcg_source = 'precomputed://https://storage.googleapis.com/microns-seunglab/minnie65/seg_minnie65_0'

#df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest-nucfind", n_threads=0)
iterator=create_skeletonize_tasks(df, pcg_source,
                                          'gs://allen-minnie-phase3/minniephase2-skeletons/',
                                          remove_duplicate_vertices=True)
tq.insert_all(iterator)
