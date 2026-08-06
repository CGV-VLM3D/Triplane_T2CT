import sys; sys.path.insert(0,'/workspace')
import numpy as np, nibabel as nib, pandas as pd
from scipy import ndimage
from pathlib import Path

B=Path("/workspace/tests/ctrate_eda_bundle/files")
GRP={"valid_1000_a":"all-zero","valid_1001_a":"lung-nodule-only","valid_1022_a":"diffuse-low-burden","valid_1016_b":"multi-abnormality","valid_1288_a":"medical-material"}

def load(scan,r):
    p=B/GRP[scan]/f"{scan}_{r}.nii.gz"
    v=np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)
    v[v<=-8192]=-1000.0
    return np.clip(v,-1000,1000)

# independent noise: std of (vol - gaussian sigma2) in soft-tissue mask eroded 2
def noise(v):
    m=(v>=-50)&(v<=120); m=ndimage.binary_erosion(m,iterations=2)
    hp=v-ndimage.gaussian_filter(v,2.0)
    return float(hp[m].std()), int(m.sum())
def lapvar(v):
    body=v>-500
    return float(ndimage.laplace(v)[body].var())

print("scan recon noise nvox lapvar")
res={}
for scan in GRP:
    res[scan]={}
    for r in (1,2):
        v=load(scan,r); n,nv=noise(v); lv=lapvar(v)
        res[scan][r]=(n,nv,lv)
        print(f"{scan} {r} noise={n:.3f} nvox={nv} lapvar={lv:.1f}")

# voxelwise for 1016_b
va=load("valid_1016_b",1); vb=load("valid_1016_b",2)
d=np.abs(va-vb)
print(f"\n1016_b voxelwise mean|d|={d.mean():.2f} max|d|={d.max():.0f} equal={np.array_equal(va,vb)} shapes={va.shape}{vb.shape}")

# sharper recon by lapvar
print("\nsharper (by lapvar):")
for scan in GRP:
    s=1 if res[scan][1][2]>=res[scan][2][2] else 2
    print(f"  {scan}: recon-{s}")

# kernels from metadata CSV (independent read)
meta=pd.read_csv("/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv")
meta["stem"]=meta["VolumeName"].str.replace(".nii.gz","",regex=False)
meta=meta.set_index("stem")
print("\nkernels:")
for scan in GRP:
    for r in (1,2):
        print(f"  {scan}_{r}: kernel={meta.loc[f'{scan}_{r}','ConvolutionKernel']!r} series={meta.loc[f'{scan}_{r}','SeriesDescription']!r}")

# no_chest check for these volumes
for f in ["no_chest_valid.txt","no_chest_train.txt"]:
    p=Path("/workspace/datasets/datasets/CT-RATE/dataset/metadata")/f
    if p.exists():
        txt=p.read_text()
        for scan in GRP:
            for r in (1,2):
                if f"{scan}_{r}" in txt: print(f"NO_CHEST HIT: {scan}_{r} in {f}")
print("no_chest check done")
