"""Re-render matplotlib manuscript figures to submission formats (vector PDF + 600dpi TIFF + 300dpi PNG),
named by figure number. Optional argv = which scripts to render (default all).
"""
import os
import sys
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as MF

OUT = "results/figs/hires"
os.makedirs(OUT, exist_ok=True)
NAME = {
    "fig1a_auc": "Figure2a", "fig1b_paired": "Figure2b", "fig_ablation": "Figure3",
    "fig4_band": "Figure4", "fig_roc": "Figure5",
    "figS1_spatial": "FigureS1", "figS2_learning_curve": "FigureS2", "figS3_threshold": "FigureS3",
}
_orig = MF.Figure.savefig


def multi(self, fname, **kw):
    base = NAME.get(os.path.basename(os.path.splitext(fname)[0]),
                    os.path.basename(os.path.splitext(fname)[0]))
    bbox = kw.get("bbox_inches", "tight")
    _orig(self, f"{OUT}/{base}.pdf", bbox_inches=bbox)
    _orig(self, f"{OUT}/{base}.png", dpi=300, bbox_inches=bbox)
    try:
        _orig(self, f"{OUT}/{base}.tiff", dpi=600, bbox_inches=bbox, pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:
        print("  [tiff skipped]", base, repr(e))
    print(f"  saved {base}  (pdf / tiff / png)")


MF.Figure.savefig = multi
SCRIPTS = sys.argv[1:] or ["scripts/make_fig1_split.py", "scripts/fig_ablation.py", "scripts/make_fig3.py",
                           "scripts/fig_roc.py", "scripts/make_fig_lc.py", "scripts/make_fig_threshold.py"]
for s in SCRIPTS:
    print("==", s, flush=True)
    runpy.run_path(s, run_name="__main__")
print("DONE ->", OUT)
