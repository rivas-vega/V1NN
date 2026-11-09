"""Synthetic import fixture, not a validated real MICrONS download."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
pd=pytest.importorskip("pandas")
spec=importlib.util.spec_from_file_location("prepare",Path(__file__).parents[1]/"scripts/prepare_microns.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_prepare_preserves_isolates_and_exact_ids():
    nodes=pd.DataFrame({"pt_root_id":["864691135000000001","864691135000000002","864691135000000003"],
        "is_column":[True,True,True],"broad_type":["excitatory","inhibitory","excitatory"]})
    syn=pd.DataFrame({"pre_pt_root_id":np.array([864691135000000001]*2,dtype=np.int64),
        "post_pt_root_id":np.array([864691135000000002]*2,dtype=np.int64),"id":[1,2]})
    payload,report=module.prepare(nodes,syn)
    assert report["nodes"]==3 and report["directed_pairs"]==1
    assert payload["synapse_count"].tolist()==[2]
    assert payload["signs"].tolist()==[1,-1,1]
    assert not report["layer_metadata_present"]


def test_blob_hash():
    assert module.git_blob_sha(b"hello\n")=="ce013625030ba8dba906f756967f9e9ca394464a"
