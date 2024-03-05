from taskqueue import TaskQueue
from caveclient import CAVEclient
from MinnieTask import make_nucleus_adjustment_tasks, readjust_nuclei

client = CAVEclient("minnie65_phase3_v1")
seg_source = client.info.segmentation_source()
nuc_cv_path = client.materialize.get_table_metadata("nucleus_detection_v0")[
    "flat_segmentation_source"
]
nuc_df = client.materialize.query_table("nucleus_detection_v0")


tq = TaskQueue(
    qurl="https://sqs.us-west-2.amazonaws.com/629034007606/forrest", n_threads=8
)
tq.insert(
    make_nucleus_adjustment_tasks(
        nuc_df,
        nuc_cv_path,
        seg_source,
        "file://./minniev1-nuc-adjustments-jan2023/",
        position_column="pt_position",
        nuc_id_column="id",
        nuc_sv_column="pt_supervoxel_id",
        nuc_pos_resolution=(4, 4, 40),
        radius=3000,
    ),
    total=len(nuc_df),
)
