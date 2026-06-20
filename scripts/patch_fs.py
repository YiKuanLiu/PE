"""Patch final_stats.py so the two per-lobe FEATURE models use mean-pooling MIL OOF
(matching the CNN's mean aggregation), instead of random-forest. Idempotent-ish.
把 final_stats.py 的兩個逐肺葉特徵模型(Features 21、V/Q)改用 mean-MIL 的 OOF,與 CNN 聚合一致。
"""
p = "scripts/final_stats.py"
s = open(p).read()
s = s.replace(
    'OOF["Features (21)"] = np.load("results/mil/tuned_bands.npz")["oof"]',
    'OOF["Features (21)"] = np.load("results/mil/oof_feat_mean.npz")["oof_feat"]')
s = s.replace(
    'OOF["V/Q features"] = rf_oof(bags[:, :, :3].reshape(N, -1))',
    'OOF["V/Q features"] = np.load("results/mil/oof_feat_mean.npz")["oof_vq"]')
open(p, "w").write(s)
print("patched final_stats.py (Features 21 + V/Q -> mean-MIL OOF)")
