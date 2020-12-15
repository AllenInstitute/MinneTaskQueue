from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_skeletonize_tasks, MinnieSkeletonizeNeuron
import numpy as np

df = pd.read_pickle('df_100cells.pkl')
pcg_source = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1'

#df_filt = df[df['size']>200000].reset_index()

tq= TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest-nucfind", n_threads=0)
tq.insert_all(create_skeletonize_tasks(df, pcg_source,
                                          'gs://allen-minnie-phase3/minniephase3-skeletons/'))
