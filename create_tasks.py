import pandas as pd
import numpy as np
import argschema
from cloudvolume.lib import Bbox, Vec
from taskqueue import TaskQueue
import cloudvolume
from annotationframeworkclient import FrameworkClient

example_d = {
    'nuc_file': "/Users/forrestc/ConnectomeStack/analysis_temp/p3minnie_nucleus_seg_info.df.csv.df",
    'nucleus_segmentation_source': 'precomputed://https://s3-hpcrc.rc.princeton.edu/minnie65-phase3-ws/nuclei/v0/seg',
    'aligned_volume': "minnie65_phase3",
    'size_threshold': 10000,
    'mip': 3,

}
class CellDetectionSchema(argschema.ArgSchema):
    nuc_file = argschema.fields.InputFile(required=True, description="csv file of nucleus detection synaptor output")
    mip = argschema.fields.Int(required=True, default=3, description = "mip of nucleus detection")
    size_threshold = argschema.fields.Int(required=True, default=10000,
                                           description="minimum size of detection in cleft_size units foudn in csv file")
    aligned_volume = argschema.fields.Str(required=True, default="minnie65_phase3", 
                                          description="name of datastack this was run on")
    nucleus_segmentation_source = argschema.fields.Str(required=True,
                                                       description="cvpath to nucleus segmentation")



def main(args):

    

    # startcoord = Vec(*config["startcoord"])
    # volshape = Vec(*config["volshape"])

    # bounds = Bbox(startcoord, startcoord + volshape)
    # print(config["tempoutput"])
    # print(bounds)
    iterator = tc.create_connected_component_tasks(
                   config["descriptor"], config["tempoutput"],
                   storagestr=config["storagestrs"][0],
                   storagedir=config["storagestrs"][1],
                   cc_thresh=config["ccthresh"], sz_thresh=config["dustthresh"],
                   bounds=bounds, shape=config["chunkshape"],
                   mip=config["voxelres"], hashmax=config["nummergetasks"])

    tq = TaskQueue(config["queueurl"])
    tq.insert_all(iterator)


if __name__ == "__main__":

    mod = argschema.ArgSchemaParser(schema_type=CellDetectionSchema, input_data=example_d)
    
    print(mod.args)