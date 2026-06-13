"""Feasibility: CT ventilation/perfusion map from inhale->exhale registration.
可行性：吸->吐形變配準導出的 CT 通氣/灌注圖。

Reproduces the FAMILY of methods used by Kuo et al. (npj Biomed Innov 2026): deformable
registration between exhale (T50) and inhale (T00), then a per-voxel ventilation surrogate
(Jacobian of the deformation = local volume change; and DIR-HU density change). Visualises a
few PE+ / PE- cases so we can judge (1) registration quality, (2) physiological plausibility,
(3) whether PE cases show regional perfusion/ventilation defects.
重建 Kuo et al. 的方法家族：吐->吸形變配準，再算每體素的通氣替代量（形變的 Jacobian = 局部體積變化；
以及 DIR-HU 密度變化）。視覺化幾個 PE 正/負案例，判斷配準品質、生理合理性、PE 是否有區域缺損。

    python -m scripts.vent_feasibility --config configs/swinunetr_ie.yaml \
      --cases case_01.mat,case_02.mat --shrink 2 --out results/vent_feas
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import SimpleITK as sitk
import yaml

HU_LO, HU_HI = -1000.0, 200.0


def locate(raw_dir, fname):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw_dir, sub, fname)
        if os.path.exists(p):
            return p
    return None


def np2sitk(a, spacing):
    """numpy (X,Y,Z) -> sitk image (spacing in mm). / numpy(X,Y,Z) 轉 sitk 影像。"""
    img = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    return img


def sitk2np(img):
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)


def register_exhale_to_inhale(hu_in, hu_ex, spacing, iters=40):
    """Diffeomorphic Demons: fixed=inhale(T00), moving=exhale(T50). / 形變配準（吐->吸）。"""
    fixed = np2sitk(np.clip(hu_in, HU_LO, HU_HI), spacing)
    moving = np2sitk(np.clip(hu_ex, HU_LO, HU_HI), spacing)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    moving_m = matcher.Execute(moving, fixed)
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iters)
    demons.SetStandardDeviations(1.2)
    disp = demons.Execute(fixed, moving_m)              # displacement field / 位移場
    jac = sitk2np(sitk.DisplacementFieldJacobianDeterminant(disp))
    tx = sitk.DisplacementFieldTransform(sitk.Cast(disp, sitk.sitkVectorFloat64))
    warped = sitk2np(sitk.Resample(np2sitk(hu_ex, spacing), fixed, tx, sitk.sitkLinear, -1000.0))
    return jac, warped


def process_case(path, shrink, iters):
    m = sio.loadmat(path, variable_names=["T00", "T00_Lobe", "T50", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]) if "xymm" in m else 1.0
    zmm = float(np.ravel(m["zmm"])[0]) if "zmm" in m else 1.0
    s = shrink
    hu00 = (np.asarray(m["T00"]).astype(np.float32) - 1024.0)[::s, ::s, ::s]
    hu50 = (np.asarray(m["T50"]).astype(np.float32) - 1024.0)[::s, ::s, ::s]
    lobe = np.asarray(m["T00_Lobe"]).astype(np.int16)[::s, ::s, ::s]
    spacing = (xymm * s, xymm * s, zmm * s)
    jac, warped = register_exhale_to_inhale(hu00, hu50, spacing, iters)
    dHU = np.clip(hu00, HU_LO, HU_HI) - np.clip(warped, HU_LO, HU_HI)   # inhale - warped exhale
    return hu00, hu50, warped, jac, dHU, lobe


def per_lobe_stats(jac, dHU, lobe):
    out = {}
    for i in range(1, 6):
        mk = lobe == i
        if mk.sum() > 50:
            out[f"lobe{i}"] = {"n": int(mk.sum()),
                               "jac_mean": float(jac[mk].mean()),
                               "dHU_mean": float(dHU[mk].mean())}
    mk = lobe > 0
    out["whole"] = {"n": int(mk.sum()),
                    "jac_mean": float(jac[mk].mean()) if mk.any() else float("nan"),
                    "dHU_mean": float(dHU[mk].mean()) if mk.any() else float("nan"),
                    "jac_p10": float(np.percentile(jac[mk], 10)) if mk.any() else float("nan"),
                    "dHU_p90": float(np.percentile(dHU[mk], 90)) if mk.any() else float("nan")}
    return out


def montage(case, label, hu00, hu50, warped, jac, dHU, lobe, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lung = lobe > 0
    z = int(np.argmax(lung.sum(axis=(0, 1))))           # slice with most lung / 肺面積最大切片
    msk = lung[:, :, z]
    jm = np.ma.masked_where(~msk, jac[:, :, z])
    dm = np.ma.masked_where(~msk, dHU[:, :, z])
    panels = [(hu00[:, :, z].T, "inhale T00", "gray", None),
              (hu50[:, :, z].T, "exhale T50", "gray", None),
              (warped[:, :, z].T, "warped exhale->inhale", "gray", None),
              (jm.T, "Jacobian (vent)", "jet", (0.7, 1.5)),
              (dm.T, "dHU (in-exWarp)", "jet", (-200, 50))]
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    for a, (im, ttl, cm, lim) in zip(ax, panels):
        kw = dict(cmap=cm)
        if lim:
            kw.update(vmin=lim[0], vmax=lim[1])
        elif cm == "gray":
            kw.update(vmin=-1000, vmax=200)
        h = a.imshow(im, origin="lower", **kw)
        a.set_title(ttl, fontsize=10); a.axis("off")
        if cm == "jet":
            fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle(f"{case}  label={label}  (1=PE+, 0=PE-)  slice z={z}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cases", required=True, help="comma list of .mat filenames / 逗號分隔")
    ap.add_argument("--shrink", type=int, default=2)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default="results/vent_feas")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_dir = cfg["data"]["raw_dir"]
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    lab = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for case in args.cases.split(","):
        case = case.strip()
        path = locate(raw_dir, case)
        if path is None:
            print(f"{case}: NOT FOUND"); continue
        label = int(lab.get(case, -1))
        print(f"processing {case} (label={label})...", flush=True)
        hu00, hu50, warped, jac, dHU, lobe = process_case(path, args.shrink, args.iters)
        stats = per_lobe_stats(jac, dHU, lobe)
        png = os.path.join(args.out, f"{os.path.splitext(case)[0]}.png")
        montage(case, label, hu00, hu50, warped, jac, dHU, lobe, png)
        summary[case] = {"label": label, "stats": stats}
        w = stats["whole"]
        print(f"  jac[mean={w['jac_mean']:.3f} p10={w['jac_p10']:.3f}] "
              f"dHU[mean={w['dHU_mean']:.1f} p90={w['dHU_p90']:.1f}] -> {png}", flush=True)

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print("\n=== per-case whole-lung ventilation summary ===")
    print(f"{'case':<14}{'label':>6}{'jac_mean':>10}{'jac_p10':>9}{'dHU_mean':>10}{'dHU_p90':>9}")
    for case, d in summary.items():
        w = d["stats"]["whole"]
        print(f"{case:<14}{d['label']:>6}{w['jac_mean']:>10.3f}{w['jac_p10']:>9.3f}"
              f"{w['dHU_mean']:>10.1f}{w['dHU_p90']:>9.1f}")
    print(f"\nsaved montages + summary -> {args.out}")


if __name__ == "__main__":
    main()
