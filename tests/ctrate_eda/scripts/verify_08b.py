import numpy as np, nibabel as nib, pandas as pd
from pathlib import Path
B=Path("/workspace/tests/ctrate_eda_bundle/files")
def load(grp,scan,r):
    v=np.asanyarray(nib.load(str(B/grp/f"{scan}_{r}.nii.gz")).dataobj).astype(np.float32)
    v[v<=-8192]=-1000.0; return np.clip(v,-1000,1000)
# 1016_b voxelwise
va=load("multi-abnormality","valid_1016_b",1); vb=load("multi-abnormality","valid_1016_b",2)
d=np.abs(va-vb)
print(f"1016_b voxelwise mean|d|={d.mean():.2f} max|d|={d.max():.0f} equal={np.array_equal(va,vb)} shapes {va.shape} {vb.shape}")
# kernels
meta=pd.read_csv("/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv")
meta["stem"]=meta["VolumeName"].str.replace(".nii.gz","",regex=False); meta=meta.set_index("stem")
for s in ["valid_1000_a","valid_1001_a","valid_1022_a","valid_1016_b","valid_1288_a"]:
    print(s, [meta.loc[f"{s}_{r}","ConvolutionKernel"] for r in (1,2)], "series", [str(meta.loc[f"{s}_{r}","SeriesDescription"]) for r in (1,2)])
# no_chest
md=Path("/workspace/datasets/datasets/CT-RATE/dataset/metadata")
import glob
for f in glob.glob(str(md/"no_chest*")):
    t=Path(f).read_text()
    for s in ["valid_1000_a","valid_1001_a","valid_1022_a","valid_1016_b","valid_1288_a"]:
        for r in (1,2):
            if f"{s}_{r}" in t: print("NO_CHEST",f,s,r)
print("done; no_chest files:", glob.glob(str(md/"no_chest*")))
