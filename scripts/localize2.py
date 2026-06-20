"""Localization v2: fair equal-volume distance zones + multi-threshold robustness + volume fractions.
定位 v2:用距離分位數切等體積區帶(公平比較)+ 多門檻穩健性 + 各區體積佔比。
"""
import glob
import os

import numpy as np
import scipy.io as sio
import yaml
from scipy import ndimage
from sklearn.metrics import roc_auc_score

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml")); raw = cfg["data"].get("raw_dir", "")
d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
files = [str(f) for f in d0["files"]]; y = d0["y"].astype(int); N = len(y)
sp = np.load("results/mil/spacing.npz", allow_pickle=True); XY = sp["xymm"].astype(float); ZZ = sp["zmm"].astype(float)
idx = {}
for r in [raw, "/home/yikuan/PE"]:
    if r and os.path.isdir(r):
        for p in glob.glob(os.path.join(r, "**", "*.mat"), recursive=True):
            idx.setdefault(os.path.basename(p), p)
print(f"indexed {len(idx)} mats", flush=True)

LO, HI = 50.0, 150.0; THR = [5, 10, 15]; NQ = 3
hf_q = np.full((N, NQ), np.nan); hf_all = np.full(N, np.nan)
hf_shell = np.full((N, len(THR)), np.nan); hf_core = np.full((N, len(THR)), np.nan); shellvol = np.full((N, len(THR)), np.nan)

for i, fn in enumerate(files):
    p = idx.get(fn) or idx.get(os.path.basename(fn))
    if not p:
        continue
    m = sio.loadmat(p, variable_names=["T00", "T00_Lobe"])
    hu = np.asarray(m["T00"]).astype(np.float32) - 1024.0
    lung = np.asarray(m["T00_Lobe"]).astype(np.int16) > 0
    if lung.sum() < 1000:
        continue
    a0 = np.where(lung.any((1, 2)))[0]; a1 = np.where(lung.any((0, 2)))[0]; a2 = np.where(lung.any((0, 1)))[0]
    sl = (slice(a0[0], a0[-1] + 1), slice(a1[0], a1[-1] + 1), slice(a2[0], a2[-1] + 1))
    hu, lung = hu[sl], lung[sl]
    xy = XY[i] if np.isfinite(XY[i]) else 0.98; zz = ZZ[i] if np.isfinite(ZZ[i]) else 2.5
    dist = ndimage.distance_transform_edt(lung, sampling=(float(xy), float(xy), float(zz)))
    hyp = (hu >= LO) & (hu < HI) & lung
    L = int(lung.sum()); hf_all[i] = hyp.sum() / L
    dv = dist[lung]; qs = np.quantile(dv, [1.0 / 3, 2.0 / 3])
    masks = [lung & (dist <= qs[0]), lung & (dist > qs[0]) & (dist <= qs[1]), lung & (dist > qs[1])]
    for z in range(NQ):
        mk = masks[z]; hf_q[i, z] = (hyp & mk).sum() / max(int(mk.sum()), 1)
    for j, t in enumerate(THR):
        sh = lung & (dist <= t); co = lung & (dist > t)
        hf_shell[i, j] = (hyp & sh).sum() / max(int(sh.sum()), 1)
        hf_core[i, j] = (hyp & co).sum() / max(int(co.sum()), 1)
        shellvol[i, j] = sh.sum() / L
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N}", flush=True)

np.savez("results/mil/localize2.npz", hf_q=hf_q, hf_all=hf_all, hf_shell=hf_shell, hf_core=hf_core, shellvol=shellvol, y=y)


def A(s):
    m = np.isfinite(s); return roc_auc_score(y[m], s[m])


print("\n=== equal-volume tertiles by distance-from-pleura (each ~1/3 of lung volume) ===", flush=True)
for z, nm in [(0, "OUTER 1/3 (subpleural)"), (1, "MIDDLE 1/3"), (2, "INNER 1/3 (central)")]:
    s = hf_q[:, z]
    print(f"  {nm:<24} AUC {A(s):.3f} | PE+ {np.nanmean(s[y == 1]):.4g} vs PE- {np.nanmean(s[y == 0]):.4g}", flush=True)

ci = hf_q[:, 2]; co = hf_q[:, 0]; mm = np.isfinite(ci) & np.isfinite(co)
yy = y[mm]; ci, co = ci[mm], co[mm]; rng = np.random.default_rng(0); dd = []
for _ in range(5000):
    ix = rng.integers(0, len(yy), len(yy))
    if len(set(yy[ix])) > 1:
        dd.append(roc_auc_score(yy[ix], ci[ix]) - roc_auc_score(yy[ix], co[ix]))
dd = np.array(dd); pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
print(f"  [paired INNER - OUTER] dAUC {dd.mean():+.3f} [{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}] p={pv:.3f}", flush=True)

print("\n=== fixed-shell robustness (shell = <=t mm from pleura; core = deeper) ===", flush=True)
for j, t in enumerate(THR):
    print(f"  t={t:>2}mm | shell vol {np.nanmean(shellvol[:,j]):.0%} | hf_shell AUC {A(hf_shell[:,j]):.3f} | hf_core AUC {A(hf_core[:,j]):.3f}", flush=True)
print(f"\n  whole-lung hyper_frac AUC {A(hf_all):.3f}", flush=True)
print("ALLDONE", flush=True)
