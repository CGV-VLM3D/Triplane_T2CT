import sys, os, glob, json, random
import numpy as np, pandas as pd, nibabel as nib

DS = "/workspace/datasets/datasets/CT-RATE/dataset"
VF = os.path.join(DS, "valid_fixed")

# 1) on-disk valid_fixed count
disk = glob.glob(os.path.join(VF, "**", "*.nii.gz"), recursive=True)
print("valid_fixed on-disk count:", len(disk))

# 2) no_chest on-disk check (volume-level exclusion)
ncf = os.path.join(DS, "metadata", "no_chest_valid.txt")
nc = set()
if os.path.exists(ncf):
    with open(ncf) as f:
        for l in f:
            l=l.strip()
            if l: nc.add(os.path.basename(l).replace(".nii.gz",""))
print("no_chest_valid list size:", len(nc))
disk_names = set(os.path.basename(p).replace(".nii.gz","") for p in disk)
print("no_chest present on disk:", len(nc & disk_names))

# 3) metadata CSV load
md = pd.read_csv(os.path.join(DS,"metadata","validation_metadata.csv"))
md["vn"] = md["VolumeName"].str.replace(".nii.gz","",regex=False)
mdi = md.set_index("vn")

# 4) sample headers, compare to metadata; recompute stats over a sample + full-ish for zoom z
random.seed(0)
samp = random.sample(disk, 200)
dtypes=set(); axset=set(); ndims=set(); qforms=set(); sforms=set()
sp_diffs=[]; sh_diffs=[]; dets=[]; zx=[]; zz=[]
for p in samp:
    img = nib.load(p)
    hdr = img.header
    vn = os.path.basename(p).replace(".nii.gz","")
    zooms = np.array(hdr.get_zooms()[:3], dtype=float)
    shape = np.array(img.shape[:3])
    dtypes.add(str(hdr.get_data_dtype()))
    axset.add("".join(nib.aff2axcodes(img.affine)))
    ndims.add(int(img.ndim))
    qforms.add(int(hdr["qform_code"])); sforms.add(int(hdr["sform_code"]))
    dets.append(np.linalg.det(img.affine[:3,:3]))
    zx.append(zooms[0]); zz.append(zooms[2])
    if vn in mdi.index:
        r = mdi.loc[vn]
        xy=float(r["XYSpacing"]) if not isinstance(r["XYSpacing"],str) else float(str(r["XYSpacing"]).strip("[] ").split(",")[0])
        # XYSpacing may be like "[0.68, 0.68]"
        try:
            xyv = str(r["XYSpacing"]).strip("[] ")
            xy = float(xyv.split(",")[0])
        except: pass
        zs=float(r["ZSpacing"])
        # metadata spacing vs header zoom
        md_sp = np.array([xy, xy, zs])
        sp_diffs.append(np.max(np.abs(md_sp - zooms)))
        md_sh = np.array([int(r["Rows"]), int(r["Columns"]), int(r["NumberofSlices"])])
        sh_diffs.append(np.max(np.abs(md_sh - shape)))

print("dtypes:", dtypes, "| axcodes:", axset, "| ndims:", ndims)
print("qform_codes:", qforms, "| sform_codes:", sforms)
print("affine det: min %.4f max %.4f mean %.4f n_neg %d" % (min(dets),max(dets),np.mean(dets),sum(1 for d in dets if d<0)))
print("sample spacing diff max: %.3e (n=%d, n_gt_1e-2=%d)" % (max(sp_diffs), len(sp_diffs), sum(1 for d in sp_diffs if d>1e-2)))
print("sample shape diff max: %d (n_gt_0=%d)" % (max(sh_diffs), sum(1 for d in sh_diffs if d>0)))
print("sample zoom_x: min %.4f max %.4f | zoom_z: min %.4f max %.4f" % (min(zx),max(zx),min(zz),max(zz)))

# 5) full-set header zoom z stats to confirm median/min/max (load headers only, fast)
allzz=[]; allzx=[]
for p in disk:
    hdr = nib.load(p).header
    z=hdr.get_zooms()[:3]
    allzx.append(float(z[0])); allzz.append(float(z[2]))
print("FULL zoom_x: min %.4f median %.4f max %.4f" % (min(allzx), float(np.median(allzx)), max(allzx)))
print("FULL zoom_z: min %.4f median %.4f max %.4f" % (min(allzz), float(np.median(allzz)), max(allzz)))
