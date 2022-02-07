import pandas as pd
from scipy.ndimage import morphology
import numpy as np
import cloudvolume
import fastremap
import datetime
from taskqueue import RegisteredTask, queueable
from cloudfiles import CloudFiles
from caveclient import CAVEclient
from requests import HTTPError
import time
from meshparty import trimesh_io, skeletonize, skeleton_io
import os
import copy
from functools import partial

model1 = None
model2 = None
model3 = None


class NoIDFoundException(Exception):
    """there was no ID in the mask that wasn't excluded"""


def get_ids_in_mask(mask, id_space, exclude_list=(0)):
    id_list = id_space[mask]
    id_list = id_list[~np.isin(id_list, exclude_list)]
    vals, counts = np.unique(id_list, return_counts=True)
    return vals, counts


def find_max_id_in_mask(mask, id_space, exclude_list=(0)):
    id_list, counts = get_ids_in_mask(mask, id_space, exclude_list=exclude_list)
    if len(id_list) > 0:
        max_ind = np.argmax(counts)
        max_id = id_list[max_ind]
        max_frac = counts[max_ind] / np.sum(counts)
        return max_id, max_frac
    else:
        raise NoIDFoundException("could not find ")


def locate_id_in_mask(mask, sup_img, mins, scale=(16, 16, 1)):
    sv_id, sv_frac = find_max_id_in_mask(mask, sup_img)
    sv_mask = np.array(sup_img == sv_id)
    sv_id_loc = np.where(sv_mask)
    sv_id_loc = (
        np.array([a[0] for a in sv_id_loc[:-1]]) * scale + np.array(mins) * scale
    )

    return sv_id, sv_frac, sv_id_loc, sv_mask


def get_id_spaces(nuc_seg_source, pcg_source, bbox, timestamp, resolution=(64, 64, 40)):
    seg_cv = cloudvolume.CloudVolume(
        pcg_source, mip=resolution, use_https=True, fill_missing=True, bounded=False
    )

    if pcg_source.startswith("graphene"):
        sup_img = seg_cv.download(bbox, agglomerate=False)
        flat_img = seg_cv.agglomerate_cutout(np.array(sup_img), timestamp=timestamp)
    else:
        flat_img = seg_cv.download(bbox)
        sup_img = flat_img
    if nuc_seg_source is not None:
        nuc_cv = cloudvolume.CloudVolume(
            nuc_seg_source,
            mip=resolution,
            fill_missing=True,
            bounded=False,
            use_https=True,
        )
        nuc_seg_cutout = nuc_cv.download(bbox)
    else:
        nuc_seg_cutout = flat_img

    return sup_img, flat_img, nuc_seg_cutout


class MinnieSkeletonizeNeuron(RegisteredTask):
    def __init__(
        self,
        seg_id,
        pcg_source="",
        bucket_save_location="",
        ctr_pt_nm=(0, 0, 0),
        remove_duplicate_vertices=False,
    ):
        super().__init__(
            seg_id,
            pcg_source,
            bucket_save_location,
            ctr_pt_nm,
            remove_duplicate_vertices,
        )
        self.pcg_source = pcg_source
        self.seg_id = seg_id
        self.bucket_save_location = bucket_save_location
        self.ctr_pt_nm = ctr_pt_nm
        self.remove_duplicate_vertices = remove_duplicate_vertices

    def execute(self):
        # print(self.remove_duplicate_vertices)
        if self.remove_duplicate_vertices:
            cv = cloudvolume.CloudVolume(self.pcg_source)
            cvmesh = cv.mesh.get(
                self.seg_id, remove_duplicate_vertices=True, fuse=True, chunk_size=None
            )
            cvmesh = cvmesh.consolidate()
            mesh = trimesh_io.Mesh(vertices=cvmesh.vertices, faces=cvmesh.faces)
        else:
            mm = trimesh_io.MeshMeta(cv_path=self.pcg_source, cache_size=1)
            mesh = mm.mesh(
                seg_id=self.seg_id,
                remove_duplicate_vertices=self.remove_duplicate_vertices,
            )
        sk = skeletonize.skeletonize_mesh(
            mesh, soma_pt=np.array(self.ctr_pt_nm), compute_radius=False
        )
        skfname = f"{self.seg_id}.h5"
        skeleton_io.write_skeleton_h5(sk, skfname, overwrite=True)
        with open(skfname, "rb") as fp:
            data = fp.read()

        cf = CloudFiles(self.bucket_save_location)
        cf.put(skfname, data, content_type="application/x-hdf")
        os.remove(skfname)

        return


def create_skeletonize_tasks(
    df,
    pcg_source,
    bucket_save_location,
    resolution=(4, 4, 40),
    remove_duplicate_vertices=False,
):
    class SkeletonTaskIterator(object):
        def __init__(
            self,
            df,
            pcg_source,
            bucket_save_location,
            resolution,
            remove_duplicate_vertices,
        ):
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.df = df
            self.resolution = resolution
            self.remove_duplicate_vertices = remove_duplicate_vertices

        def __len__(self):
            return len(self.df)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.df = df.iloc[slc]
            return itr

        def __iter__(self):

            for num, row in self.df.iterrows():
                ctr_pt = row.nuc_loc * self.resolution
                yield MinnieSkeletonizeNeuron(
                    row.cell_id,
                    self.pcg_source,
                    bucket_save_location=self.bucket_save_location,
                    ctr_pt_nm=ctr_pt,
                    remove_duplicate_vertices=self.remove_duplicate_vertices,
                )

    return SkeletonTaskIterator(
        df, pcg_source, bucket_save_location, resolution, remove_duplicate_vertices
    )


class MinniePerformNucMergeTask(RegisteredTask):
    """[summary]

    Args:
        RegisteredTask ([type]): [description]
    """

    def __init__(
        self,
        pcg_source="",
        bucket_save_location="",
        nuc_sv_id=0,
        cell_sv_id=0,
        nuc_sv_id_loc=(0, 0, 0),
        cell_sv_id_loc=(0, 0, 0),
        resolution=(4, 4, 40),
        datastack_name="minnie65_phase3_v1",
    ):
        """[summary]

        Args:
            pcg_source (str, optional): [description]. Defaults to "".
            bucket_save_location (str, optional): [description]. Defaults to "".
            nuc_sv_id (int, optional): [description]. Defaults to 0.
            cell_sv_id (int, optional): [description]. Defaults to 0.
            nuc_sv_id_loc (tuple, optional): [description]. Defaults to (0,0,0).
            cell_sv_id_loc (tuple, optional): [description]. Defaults to (0,0,0).
            resolution (tuple, optional): [description]. Defaults to (4,4,40).
        """
        super().__init__(
            pcg_source,
            bucket_save_location,
            nuc_sv_id,
            cell_sv_id,
            nuc_sv_id_loc,
            cell_sv_id_loc,
            resolution,
            datastack_name,
        )

        self.pcg_source = pcg_source
        self.bucket_save_location = bucket_save_location
        self.nuc_sv_id = nuc_sv_id
        self.cell_sv_id = cell_sv_id
        self.nuc_sv_id_loc = nuc_sv_id_loc
        self.cell_sv_id_loc = cell_sv_id_loc
        self.resolution = resolution
        self.datastack_name = datastack_name

    def execute(self):
        client = CAVEclient(self.datastack_name)
        roots = client.chunkedgraph.get_roots([self.nuc_sv_id, self.cell_sv_id])
        r = {}
        if roots[0] != roots[1]:
            try:
                client.chunkedgraph.do_merge(
                    [self.nuc_sv_id, self.cell_sv_id],
                    [self.nuc_sv_id_loc, self.cell_sv_id_loc],
                    resolution=self.resolution,
                )
                r["did_merge"] = True
                # r["new_root_id"] = r.pop("new_root_ids")[0]
            except HTTPError:
                r = {}
                print("timeout error.. ")
                for i in range(7):
                    print("sleeping..")
                    time.sleep(30)
                    root_test = client.chunkedgraph.get_roots(
                        [self.nuc_sv_id, self.cell_sv_id]
                    )
                    if root_test[0] == root_test[1]:

                        print("merge test found completion")
                        r["did_merge"] = True
                        r["new_root_id"] = root_test[1]
                        break
                if "did_merge" not in r.keys():
                    raise Exception(
                        "Merge appeared to have failed after 3.5 minutes of waiting"
                    )
        else:

            r["did_merge"] = False
        r["nuc_sv_id"] = self.nuc_sv_id
        r["cell_sv_id"] = self.cell_sv_id
        r["nuc_sv_id_loc"] = self.nuc_sv_id_loc
        r["cell_sv_id_loc"] = self.cell_sv_id_loc
        r["nuc_root"] = roots[0]
        r["cell_root"] = roots[1]
        cf = CloudFiles(self.bucket_save_location)
        cf.put_json(f"{self.nuc_sv_id}.json", r)
        print(r)
        return


class MinnieSupervoxelLookup(RegisteredTask):
    def __init__(
        self,
        pcg_source="",
        flat_nuc_id=0,
        centroid=(5, 5, 5),
        resolution=[64, 64, 40],
        bucket_save_location="",
    ):
        super().__init__(
            pcg_source, flat_nuc_id, centroid, resolution, bucket_save_location
        )

        self.pcg_source = pcg_source
        self.flat_nuc_id = int(flat_nuc_id)
        self.centroid = centroid
        self.resolution = resolution
        self.bucket_save_location = bucket_save_location

    def execute(self):
        seg_cv = cloudvolume.CloudVolume(
            self.pcg_source,
            mip=self.resolution,
            use_https=True,
            fill_missing=True,
            bounded=False,
        )
        svid = np.squeeze(
            seg_cv.download_point(
                self.centroid, size=1, coord_resolution=self.resolution
            )
        )

        cf = CloudFiles(self.bucket_save_location)
        lookup_d = {"flat_nuc_id": self.flat_nuc_id, "svid": np.uint64(svid)}
        cf.put_json(f"{self.flat_nuc_id}.json", lookup_d)
        print(lookup_d)


class MinnieNucMergeTask(RegisteredTask):
    def __init__(
        self,
        nuc_source=None,
        pcg_source="",
        flat_nuc_id=0,
        mins=(0, 0, 0),
        maxs=(10, 10, 10),
        centroid=(5, 5, 5),
        resolution=[64, 64, 40],
        bucket_save_location="",
        point_resolution=[4, 4, 40],
        timestamp=datetime.datetime.now(),
        find_merge=True,
    ):
        super().__init__(
            nuc_source,
            pcg_source,
            flat_nuc_id,
            mins,
            maxs,
            centroid,
            resolution,
            bucket_save_location,
            point_resolution,
            timestamp,
            find_merge,
        )

        self.nuc_source = nuc_source
        self.pcg_source = pcg_source
        self.flat_nuc_id = int(flat_nuc_id)
        self.mins = mins
        self.maxs = maxs
        self.centroid = centroid
        self.resolution = resolution
        self.bucket_save_location = bucket_save_location
        self.timestamp = timestamp
        self.find_merge = find_merge
        self.scale = np.array(resolution, dtype=np.float) / point_resolution

    def execute(self):

        bbox = cloudvolume.Bbox(self.mins, self.maxs)
        print(f"findin merging {self.flat_nuc_id} at {self.centroid * self.scale}")
        print(bbox)
        # dt_timestamp = datetime.datetime.fromisoformat(self.timestamp)
        sup_img, flat_img, nuc_seg_cutout = get_id_spaces(
            self.nuc_source, self.pcg_source, bbox, self.timestamp, self.resolution
        )
        n_zeros = np.sum(sup_img.ravel() == 0)
        n_voxels = np.prod(sup_img.shape)
        frac_zeros = float(n_zeros / n_voxels)

        ctr_ind = np.array(self.centroid, dtype=np.int32) - np.array(
            self.mins, dtype=np.int32
        )
        cf = CloudFiles(self.bucket_save_location)

        # get border voxels using masks
        # if no flat nucleus given, use voxel at center
        if self.flat_nuc_id == 0:
            ctr = bbox.size() // 2
            self.flat_nuc_id = nuc_seg_cutout[ctr[0], ctr[1], ctr[2]][0]

        nuc_mask = np.array(nuc_seg_cutout == self.flat_nuc_id)
        print(np.sum(nuc_mask), nuc_mask.shape)
        if (np.sum(nuc_mask) == 0) | np.any(np.array(nuc_mask.shape[:3]) == 1):
            merge_events = [
                {
                    "frac_zeros": frac_zeros,
                    "flat_nuc_id": self.flat_nuc_id,
                    "ctr_pt_id": 0,
                    "nuc_id": 0,
                    "cell_id": 0,
                    "centroid": self.centroid * self.scale,
                    "message": "nucleus is only in single section or contains zero voxels",
                }
            ]
            cf.put_json(f"{self.flat_nuc_id}.json", merge_events)
            print(merge_events)
            return

        nuc_mask_enlarg1 = morphology.binary_dilation(nuc_mask, iterations=1)
        nuc_mask_enlarg3 = morphology.binary_dilation(nuc_mask_enlarg1, iterations=2)
        nuc_mask_shrink = morphology.binary_erosion(np.squeeze(nuc_mask), iterations=3)[
            :, :, :, np.newaxis
        ]
        cell_border_nuc_mask = np.logical_xor(nuc_mask_enlarg1, nuc_mask_enlarg3)

        ctr_seg_id = flat_img[ctr_ind[0], ctr_ind[1], ctr_ind[2], 0]

        try:
            cell_id, cell_frac = find_max_id_in_mask(
                cell_border_nuc_mask, flat_img, exclude_list=(0)
            )
        except NoIDFoundException:
            merge_events = [
                {
                    "frac_zeros": frac_zeros,
                    "flat_nuc_id": self.flat_nuc_id,
                    "ctr_pt_id": ctr_seg_id,
                    "nuc_id": ctr_seg_id,
                    "cell_id": 0,
                    "centroid": self.centroid * self.scale,
                    "message": "no cell ids found in border, likely out of volume",
                }
            ]
            cf.put_json(f"{self.flat_nuc_id}.json", merge_events)
            print(merge_events)
            return

        try:
            nuc_ids, nuc_counts = get_ids_in_mask(
                nuc_mask_shrink, flat_img, exclude_list=(0, cell_id)
            )
        except NoIDFoundException:
            merge_events = [
                {
                    "frac_zeros": frac_zeros,
                    "flat_nuc_id": self.flat_nuc_id,
                    "ctr_pt_id": ctr_seg_id,
                    "nuc_id": ctr_seg_id,
                    "cell_id": cell_id,
                    "cell_frac": cell_frac,
                    "centroid": self.centroid * self.scale,
                    "message": "no IDs in nucleus mask other than cell_id",
                }
            ]
            cf.put_json(f"{self.flat_nuc_id}.json", merge_events)
            print(merge_events)
            return

        merge_events = []

        if self.find_merge:
            cell_mask = np.array(flat_img == cell_id)
            cell_mask_grow = morphology.binary_dilation(cell_mask, iterations=1)
            nuc_border_cell_mask = np.logical_and(cell_mask_grow, nuc_mask)

            for nuc_id, nuc_voxels in zip(nuc_ids, nuc_counts):

                nuc_border_cell_mask2 = np.logical_and(
                    nuc_border_cell_mask, np.array(flat_img == nuc_id)
                )

                try:
                    vals = locate_id_in_mask(
                        nuc_border_cell_mask2, sup_img, self.mins, scale=self.scale
                    )
                    nuc_sv_id, nuc_sv_id_frac, nuc_sv_id_loc, nuc_sv_id_mask = vals

                    nuc_sv_id_mask_expand = morphology.binary_dilation(
                        nuc_sv_id_mask, iterations=2
                    )
                    border_nuc_sv_id_mask = np.logical_and(
                        nuc_sv_id_mask_expand, cell_mask
                    )

                    vals = locate_id_in_mask(
                        border_nuc_sv_id_mask, sup_img, self.mins, scale=self.scale
                    )
                    cell_sv_id, cell_sv_id_frac, cell_sv_id_loc, cell_sv_id_mask = vals

                    print(
                        nuc_id,
                        nuc_sv_id,
                        nuc_sv_id_loc,
                        cell_sv_id,
                        cell_sv_id_frac,
                        cell_id,
                        cell_frac,
                    )
                    merge_event = {
                        "frac_zeros": frac_zeros,
                        "flat_nuc_id": self.flat_nuc_id,
                        "ctr_pt_id": ctr_seg_id,
                        "nuc_id": nuc_id,
                        "nuc_sv_id": nuc_sv_id,
                        "nuc_sv_id_loc": nuc_sv_id_loc.tolist(),
                        "cell_id": cell_id,
                        "cell_sv_id": cell_sv_id,
                        "cell_sv_id_loc": cell_sv_id_loc.tolist(),
                        "cell_frac": cell_frac,
                        "nuc_id_voxels": nuc_voxels,
                        "centroid": self.centroid * self.scale,
                        "message": "success, merge found",
                    }
                    merge_events.append(merge_event)
                except NoIDFoundException:
                    merge_event = {
                        "frac_zeros": frac_zeros,
                        "nuc_id": nuc_id,
                        "cell_id": cell_id,
                        "flat_nuc_id": self.flat_nuc_id,
                        "ctr_pt_id": ctr_seg_id,
                        "cell_frac": cell_frac,
                        "nuc_id_voxels": nuc_voxels,
                        "centroid": self.centroid * self.scale,
                        "message": "could not find merge for nuc_id to cell_id",
                    }
                    merge_events.append(merge_event)
            if len(merge_events) == 0:
                merge_events = [
                    {
                        "frac_zeros": frac_zeros,
                        "flat_nuc_id": self.flat_nuc_id,
                        "ctr_pt_id": ctr_seg_id,
                        "nuc_id": ctr_seg_id,
                        "cell_id": cell_id,
                        "cell_frac": cell_frac,
                        "centroid": self.centroid * self.scale,
                        "message": "no merge events found",
                    }
                ]
        else:
            merge_events = [
                {
                    "frac_zeros": frac_zeros,
                    "flat_nuc_id": self.flat_nuc_id,
                    "ctr_pt_id": ctr_seg_id,
                    "nuc_id": ctr_seg_id,
                    "cell_id": cell_id,
                    "cell_frac": cell_frac,
                    "centroid": self.centroid * self.scale,
                    "message": "success, no merge search",
                }
            ]
        cf.put_json(f"{self.flat_nuc_id}.json", merge_events)
        print(merge_events)


def create_perform_nuc_merge_tasks(
    df, pcg_source, bucket_save_location, resolution=(4, 4, 40)
):
    class PerformNucMergeTaskIterator(object):
        def __init__(self, df, pcg_source, bucket_save_location, resolution=(4, 4, 40)):
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.df = df
            self.resolution = resolution

        def __len__(self):
            return len(self.df)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.df = df.iloc[slc]
            return itr

        def __iter__(self):

            for num, row in self.df.iterrows():
                yield MinniePerformNucMergeTask(
                    self.pcg_source,
                    self.bucket_save_location,
                    row.nuc_sv_id,
                    row.cell_sv_id,
                    row.nuc_sv_id_loc,
                    row.cell_sv_id_loc,
                    self.resolution,
                )

    return PerformNucMergeTaskIterator(df, pcg_source, bucket_save_location, resolution)


def create_nuc_merge_tasks(
    df,
    nuc_source,
    pcg_source,
    bucket_save_location,
    point_resolution=(4, 4, 40),
    resolution=(64, 64, 40),
    timestamp=datetime.datetime.now(),
    find_merge=True,
    column_name="cleft_segid",
):
    class NucMergeTaskIterator(object):
        def __init__(
            self,
            df,
            nuc_source,
            pcg_source,
            bucket_save_location,
            point_resolution,
            resolution,
            timestamp,
            find_merge,
            column_name,
        ):
            self.nuc_source = nuc_source
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.timestamp = timestamp
            self.df = df
            self.resolution = resolution
            self.find_merge = find_merge
            self.point_resolution = point_resolution
            self.column_name = column_name

        def __len__(self):
            return len(self.df)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.df = df.iloc[slc]
            return itr

        def __iter__(self):

            for num, row in self.df.iterrows():
                mins = [int(row.bbox_bx), int(row.bbox_by), int(row.bbox_bz)]
                maxs = [int(row.bbox_ex), int(row.bbox_ey), int(row.bbox_ez)]
                centroid = [
                    int(row.centroid_x),
                    int(row.centroid_y),
                    int(row.centroid_z),
                ]

                yield MinnieNucMergeTask(
                    self.nuc_source,
                    self.pcg_source,
                    flat_nuc_id=row[self.column_name],
                    mins=mins,
                    maxs=maxs,
                    centroid=centroid,
                    resolution=self.resolution,
                    point_resolution=self.point_resolution,
                    bucket_save_location=self.bucket_save_location,
                    timestamp=self.timestamp.isoformat(),
                    find_merge=self.find_merge,
                )

    return NucMergeTaskIterator(
        df,
        nuc_source,
        pcg_source,
        bucket_save_location,
        point_resolution,
        resolution,
        timestamp,
        find_merge,
        column_name,
    )


def create_nuc_merge_tasks_from_points(
    pts,
    nuc_source,
    pcg_source,
    bucket_save_location,
    resolution=[77.6, 77.6, 45.0],
    timestamp=datetime.datetime.now(),
    cutout_radius_nm=(4000, 4000, 4000),
    point_resolution=(9, 9, 45),
    find_merge=True,
):
    class NucMergeTaskIterator(object):
        def __init__(
            self,
            pts,
            nuc_source,
            pcg_source,
            bucket_save_location,
            cutout_radius_nm,
            point_resolution,
            resolution,
            timestamp,
            find_merge,
        ):
            self.nuc_source = nuc_source
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.cutout_radius_nm = cutout_radius_nm
            self.point_resolution = point_resolution
            self.timestamp = timestamp
            self.pts = pts
            self.resolution = resolution

            self.find_merge = find_merge
            self.cv = cloudvolume.CloudVolume(
                pcg_source, mip=self.resolution, use_https=True
            )

        def __len__(self):
            return len(pts)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.pts = pts[slc]
            return itr

        def __iter__(self):

            for pt in pts:
                pt = (pt * self.point_resolution) / self.cv.resolution
                pt = pt.astype(np.int32)
                cutout_radius_vx = np.array(self.cutout_radius_nm) / self.cv.resolution
                cutout_radius_vx = cutout_radius_vx.astype(np.int32)
                mins = pt - cutout_radius_vx
                maxs = pt + cutout_radius_vx
                centroid = pt

                yield MinnieNucMergeTask(
                    self.nuc_source,
                    self.pcg_source,
                    flat_nuc_id=0,
                    mins=mins,
                    maxs=maxs,
                    centroid=centroid,
                    resolution=self.resolution,
                    bucket_save_location=self.bucket_save_location,
                    point_resolution=self.point_resolution,
                    timestamp=self.timestamp.isoformat(),
                    find_merge=self.find_merge,
                )

    return NucMergeTaskIterator(
        pts,
        nuc_source,
        pcg_source,
        bucket_save_location,
        cutout_radius_nm,
        point_resolution,
        resolution,
        timestamp,
        find_merge,
    )


def create_sv_lookup_tasks(
    df, pcg_source, bucket_save_location, resolution=(64, 64, 40)
):
    class SupervoxelLookupTaskIterator(object):
        def __init__(self, df, pcg_source, bucket_save_location, resolution):
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.df = df
            self.resolution = resolution

        def __len__(self):
            return len(self.df)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.df = df.iloc[slc]
            return itr

        def __iter__(self):

            for num, row in self.df.iterrows():
                centroid = [
                    int(row.centroid_x),
                    int(row.centroid_y),
                    int(row.centroid_z),
                ]
                yield MinnieSupervoxelLookup(
                    self.pcg_source,
                    flat_nuc_id=row.cleft_segid,
                    centroid=centroid,
                    resolution=self.resolution,
                    bucket_save_location=self.bucket_save_location,
                )

    return SupervoxelLookupTaskIterator(
        df, pcg_source, bucket_save_location, resolution
    )


@queueable
def quantify_soma_region(
    pcg_source, nuc_id, mins, maxs, resolution, bucket_save_location
):

    seg_cv = cloudvolume.CloudVolume(
        pcg_source, mip=resolution, use_https=True, fill_missing=True, bounded=False
    )
    bbox = cloudvolume.Bbox(mins, maxs)
    if pcg_source.startswith("graphene"):
        sv_img = seg_cv.download(bbox, agglomerate=False)
    else:
        sv_img = seg_cv.download(bbox)

    n_zeros = np.sum(sv_img.ravel() == 0)
    n_voxels = np.prod(sv_img.shape)
    frac = n_zeros / n_voxels
    print(nuc_id, mins, maxs, frac)
    d = {"frac_zeros": float(frac)}

    cf = CloudFiles(bucket_save_location)
    cf.put_json(f"{nuc_id}.json", d)
    print(d)


def make_quantify_soma_region_tasks(
    df,
    pcg_source,
    bucket_save_location,
    resolution=(64, 64, 40),
    pt_resolution=(4, 4, 40),
    radius=15000,
):

    cv = cloudvolume.CloudVolume(pcg_source, use_https=True)
    res_mip = next(
        k for k, s in enumerate(cv.scales) if s["resolution"] == list(resolution)
    )

    def make_bnd_func(row):
        ctr_pt = row.pt_position
        rad_box = (np.array([radius, radius, radius]) / pt_resolution).astype(np.int32)
        mins = ((ctr_pt - rad_box) / [2, 2, 1]).astype(np.int32)
        maxs = ((ctr_pt + rad_box) / [2, 2, 1]).astype(np.int32)
        bbox_mip0 = cloudvolume.Bbox(mins, maxs)
        bbox_mipr = cv.bbox_to_mip(bbox_mip0, 0, res_mip)
        bound_fn = partial(
            quantify_soma_region,
            pcg_source,
            row["id"],
            bbox_mipr.minpt.tolist(),
            bbox_mipr.maxpt.tolist(),
            resolution,
            bucket_save_location,
        )
        return bound_fn

    tasks = (make_bnd_func(row) for num, row in df.iterrows())
    return tasks
