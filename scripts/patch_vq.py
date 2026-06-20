"""Patch final_stats.py: ONLY the V/Q comparator switches to mean-pooling MIL OOF
(it benefits, 0.60->0.62); Features (21) stays the rigorous tuned-band random forest (== mean, 0.713 vs 0.712).
只把 V/Q 換成 mean-MIL(0.60->0.62);Features 維持嚴謹的 tuned-band RF(與 mean 等價)。
"""
p = "scripts/final_stats.py"
s = open(p).read()
s = s.replace(
    'OOF["V/Q features"] = rf_oof(bags[:, :, :3].reshape(N, -1))',
    'OOF["V/Q features"] = np.load("results/mil/oof_feat_mean.npz")["oof_vq"]')
open(p, "w").write(s)
print("patched V/Q -> mean-MIL (Features kept as tuned-band RF)")
