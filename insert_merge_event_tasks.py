from taskqueue import TaskQueue
import json
import pandas as pd
from MinnieTask import create_perform_nuc_merge_tasks, MinniePerformNucMergeTask
import numpy as np

merge_file = "minnie65_nuc_merges_round7.pkl"

df_merges = pd.read_pickle(merge_file)
seg_source = 'graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1'

bad_ids=np.array([864691135385177685,
 864691136286543299,
 864691135739570580,
 864691135360106055,
 864691135945381412,
 864691135850406855,
 864691135454044266,
 864691135428500016,
 864691135761504822,
 864691136236668175,
 864691135544367912,
 864691135515922131,
 864691135654035394,
 864691135815455311,
 864691135382463194,
 864691135968892261,
 864691135113149337,
 864691135683227378,
 864691135272160529,
 864691135847922782,
 864691136390386047,
 864691135771619067,
 864691135454072170,
 864691135571243045,
 864691135446623828,
 864691135564554007,
 864691135340986053,
 864691136273649421,
 864691135776567392], dtype=np.uint64)

df_merges['num_merges_per_cell_id'] = df_merges.groupby('cell_id').nuc_id.transform(len)
df_single_merge_df = df_merges.groupby('cell_id').first().reset_index()
df_single_merge_df=df_single_merge_df[~np.isin(df_single_merge_df.cell_id, bad_ids)]

print(df_single_merge_df.cell_id.dtype, bad_ids.dtype)
print(df_single_merge_df.shape)
tq=TaskQueue(qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest", n_threads=0)
tq.insert_all(create_perform_nuc_merge_tasks(df_single_merge_df,
                                seg_source,
                               'gs://allen-minnie-phase3/minniev1-nuc-merge-log/',
                                resolution=(4,4,40)))
