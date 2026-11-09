"""Pinned official exports. Network import not validated in authoring environment.

Raw data are not redistributed. Unit tests exercise synthetic fixtures only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

COMMIT="513a6fe738f91ce179650f1cadf26d4a4728b21a"
BASE=f"https://raw.githubusercontent.com/AllenInstitute/connectomics_at_cosyne/{COMMIT}/docs/resources/data/"
FILES={"v1718_cell_info.csv":"65bc15f566288ba763a25c7a6f91a94ab8a970ce",
       "v1718_v1_column_synapses.feather":"48ba7fa1b6aee968ab53304a655461363deddbd1"}


def git_blob_sha(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def download_verified(path,expected_sha):
    if not path.exists():
        with urlopen(BASE+path.name,timeout=60) as response:data=response.read()
        if git_blob_sha(data)!=expected_sha:
            raise RuntimeError(f"Unexpected upstream blob: {path.name}")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    data=path.read_bytes()
    if git_blob_sha(data)!=expected_sha:
        raise RuntimeError(f"Local data hash mismatch: {path}; no silent replacement")
    return {"url":BASE+path.name,"git_blob_sha1":expected_sha,
            "sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}


def prepare(nodes,synapses):
    """Keep column nodes including isolates; aggregate synapses by ordered pair."""
    required={"pt_root_id","is_column","broad_type"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"Missing node columns: {required-set(nodes.columns)}")
    for col in ("pre_pt_root_id","post_pt_root_id"):
        if col not in synapses:raise ValueError(f"Missing {col}")
        if synapses[col].dtype.kind not in "iu":
            raise ValueError("Synapse root IDs must be integer, never floating point")
    flag=nodes.is_column.astype(str).str.lower()
    if not flag.isin(["true","false","1","0"]).all():
        raise ValueError("Unknown is_column values")
    column=nodes.loc[flag.isin(["true","1"])].copy()
    column.pt_root_id=column.pt_root_id.astype(str)
    if column.empty or column.pt_root_id.duplicated().any():
        raise ValueError("Empty column or duplicate root IDs")
    column=column.sort_values("pt_root_id").reset_index(drop=True)
    mapping={value:i for i,value in enumerate(column.pt_root_id)}
    if "id" in synapses and synapses.id.duplicated().any():
        raise ValueError("Duplicate synapse IDs would inflate counts")
    src_ids=synapses.pre_pt_root_id.astype(str);dst_ids=synapses.post_pt_root_id.astype(str)
    included=src_ids.isin(mapping)&dst_ids.isin(mapping)
    pairs=np.column_stack([src_ids[included].map(mapping),dst_ids[included].map(mapping)]).astype(np.int64)
    pairs,counts=np.unique(pairs,axis=0,return_counts=True)
    signs=column.broad_type.map({"excitatory":1,"inhibitory":-1}).fillna(0).to_numpy(np.float32)
    groups=column.broad_type.fillna("unknown").astype(str).to_numpy()
    if "layer" in column:
        groups=np.array([f"{g}|{layer}" for g,layer in zip(groups,column.layer.fillna("unknown"))])
    payload={"src":pairs[:,0],"dst":pairs[:,1],"synapse_count":counts,
             "signs":signs,"root_ids":column.pt_root_id.to_numpy(dtype=str),"groups":groups.astype(str)}
    xyz=[f"pt_position_{axis}_tform" for axis in "xyz"]
    if all(key in column for key in xyz):
        positions=column[xyz].to_numpy(float)
        if np.isfinite(positions).all():payload["positions_um"]=positions
    report={"nodes":len(column),"directed_pairs":len(pairs),
            "included_synapses":int(included.sum()),"excluded_synapses":int((~included).sum()),
            "autapse_pairs":int((pairs[:,0]==pairs[:,1]).sum()),
            "excitatory":int((signs==1).sum()),"inhibitory":int((signs==-1).sum()),
            "unknown_sign":int((signs==0).sum()),"layer_metadata_present":"layer" in column,
            "boundary_retention":"not estimable from internal-column synapses alone"}
    return payload,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=Path("data/raw"))
    parser.add_argument("--output",type=Path,default=Path("data/processed/column_v1718.npz"))
    args=parser.parse_args()
    import pandas as pd
    provenance={name:download_verified(args.cache/name,sha) for name,sha in FILES.items()}
    nodes=pd.read_csv(args.cache/"v1718_cell_info.csv",dtype={"pt_root_id":"string"})
    synapses=pd.read_feather(args.cache/"v1718_v1_column_synapses.feather")
    payload,report=prepare(nodes,synapses)
    report.update({"dataset":"MICrONS static teaching export","materialization":1718,
                   "upstream_commit":COMMIT,"sources":provenance,"actual_import":True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**payload)
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
