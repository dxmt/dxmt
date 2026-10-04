"""ann-benchmarks hdf5 -> raw little-endian arrays for tests/recall/calibration_transfer_test.rs.

Usage: python3 scripts/ann_to_raw.py glove-100-angular.hdf5 /data/ann/glove-100-angular
Writes <out>.train.f32, .test.f32, .neighbors.i32 (top 10) and .meta.json. Needs h5py and numpy.
"""
import sys, json, h5py, numpy as np
src, out = sys.argv[1], sys.argv[2]
with h5py.File(src) as f:
    train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
    test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
    nb = np.ascontiguousarray(f["neighbors"][:, :10], dtype=np.int32)
    metric = f.attrs.get("distance", "unknown")
    if isinstance(metric, bytes): metric = metric.decode()
train.tofile(out + ".train.f32"); test.tofile(out + ".test.f32"); nb.tofile(out + ".neighbors.i32")
json.dump({"n": int(train.shape[0]), "dim": int(train.shape[1]), "nq": int(test.shape[0]), "metric": str(metric)}, open(out + ".meta.json", "w"))
print(out, train.shape, test.shape, metric)
