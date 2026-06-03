# 執行指南（自助跑 nested CV）

在 server `yikuan@10.157.174.124` 上，repo 位於 `~/PE_project`、conda 環境為 `PE`。
整個流程**可中斷續跑**：隨時 `Ctrl-C` 或斷線都沒關係，重跑同一指令會自動跳過已完成的 job。

---

## 一、進到環境（每次都要）
```bash
ssh yikuan@10.157.174.124          # 或直接用 VS Code 連到 server 的終端機
source /home/yikuan/miniconda3/etc/profile.d/conda.sh
conda activate PE
cd ~/PE_project
```

## 二、（可選）調整時間預算
編輯 `configs/swinunetr_i.yaml`：

| 想做的事 | 改哪裡 |
|----------|--------|
| 內層訓練短一點（省最多時間、對排序 HP 幾乎無損）| `hardware.inner_epochs: 100 → 50` |
| 內層折數少一點 | `cv.inner_folds: 5 → 3` |
| 少調一個超參數 | 從 `hpsearch.stages` 刪掉 `dropout` 那一段 |

粗估（4 張 A100、~120 秒/epoch）：完整 ~2 週；只降 cap→50 ~8-9 天；
再加內層 3 折 ~5 天；再砍 dropout 階段 ~3.5 天。

## 三、用 tmux 開跑（關鍵：斷線也不中斷）
```bash
tmux new -s pe                     # 建立名為 pe 的 session
# ── 進到 tmux 後 ──
bash run.sh                        # = make_splits + dry-run + 正式執行，並 tee 到 log
#   想用其他 config： bash run.sh configs/你的.yaml
```
- **卸離（讓它在背景繼續跑）**：先按 `Ctrl-b` 放開，再按 `d`。之後可關掉 ssh / VS Code。

> 不想用 `run.sh` 也可手動：
> ```bash
> python -m scripts.make_splits   --config configs/swinunetr_i.yaml
> python -u -m scripts.run_nested_cv --config configs/swinunetr_i.yaml 2>&1 | tee -a results/swinunetr_i/run.log
> ```

## 四、回來看進度
```bash
tmux attach -t pe                  # 重新接回
tmux ls                            # 列出所有 session
```
不接回也能監看（另開終端機）：
```bash
ls results/swinunetr_i/jobs/*.json | wc -l    # 已完成 job 數（總共 460）
tail -f results/swinunetr_i/run.log           # 即時看 orchestrator 輸出
watch -n5 nvidia-smi                           # 看 4 張卡是否都在跑
```

## 五、可中斷續跑
隨時 `Ctrl-C` / 關閉 / 重開機都沒關係 —— 再執行 `bash run.sh`（或同一行指令），
**已完成的 job（已有結果 JSON）會自動跳過**，從中斷處接著跑；崩掉沒產生 JSON 的 job 會自動重試。

## 六、隨時取結果
```bash
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --only-aggregate
cat results/swinunetr_i/summary.json          # AUC/Sens/Spec/PPV/NPV/F1 的 平均±95%CI
```

---

### tmux 速查
| 操作 | 按鍵 |
|------|------|
| 卸離（背景續跑）| `Ctrl-b` 然後 `d` |
| 重新接回 | `tmux attach -t pe` |
| 捲動畫面 | `Ctrl-b` 然後 `[`（按 `q` 離開）|
| 砍掉 session | `tmux kill-session -t pe` |

### 沒有 tmux 時的備援
```bash
nohup bash run.sh > results/swinunetr_i/run.log 2>&1 &
tail -f results/swinunetr_i/run.log
```
