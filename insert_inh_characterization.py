import caveclient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from taskqueue import queueable, TaskQueue
from functools import partial
from inh_characterize import characterize_neuron

tq = TaskQueue(
    qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest", n_threads=8
)


client = caveclient.CAVEclient("minnie65_phase3_v1")

cloud_path = (
    f"gs://allen-minnie-phase3/inh_ct_characterization_v{client.materialize.version}/"
)

ct_df = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
single_neuron_df = client.materialize.query_view("single_neurons")


inh_df = ct_df[ct_df["classification_system"].str.contains("inhibitory_neuron")]
# replace the pt_root_id with the pt_root_id from single_neuron_df
inh_dfm = pd.merge(
    inh_df, single_neuron_df, left_on="target_id", right_on="id", how="inner"
)
print(inh_dfm.columns)

tasks = (
    partial(characterize_neuron, row.pt_root_id_y, cloud_path)
    for row in inh_dfm.itertuples()
)
tq.insert(tasks)
