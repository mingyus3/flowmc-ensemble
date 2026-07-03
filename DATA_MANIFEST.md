# 数据来源清单 / DATA MANIFEST

本包对应论文:`paper/flowmc_paper_final.md`
目标期刊:Entropy (MDPI, SCI, IF≈2.7) — 收诚实的方法学/负结果研究
代码仓库:github.com/mingyus3/flowmc-ensemble

---

## 一、论文每个表/图/关键数字 → 对应数据文件

| 论文位置 | 内容 | 数据来源 | 核验 |
|---|---|---|---|
| Table 1 (single@2k列) | 单链@2000, 均值0.104 | `data/results_equal_per_member.csv` (method=flowMC-single) | ✓ 实测0.1038 |
| Table 1 (ens@10k列) | full ensemble, 均值0.085 | `data/results_equal_per_member.csv` (method=...ensemble...) | ✓ 实测0.0849, per-config逐一吻合 |
| Table 1 (single@10k列) | 单链@10000, 均值0.065 | `data/results_single_10000.csv` | ✓ 实测0.0649 |
| Figure 1 | 三方对比柱状图(11配置) | 同Table 1三列 | ✓ |
| Figure 2 | noise floor曲线 + 有效样本数坍缩 | numpy复算(3分布平均); 关键发现:ensemble有效N≈2094 | ✓ |
| Figure 3 | raw-pool定位柱状图(11配置) | 同Table 3 | ✓ |
| Table 2 (full/no_cov/no_temp/no_w) | ablation均值 | `data/significance_equal_per_member.csv` + 三个ablation run | ✓ (见下"待确认") |
| Table 2 (single@10k行) | 0.0649 | `data/results_single_10000.csv` | ✓ |
| Table 3 (rawpool列) | 纯拼接, **11配置**均值0.062 | `data/results_rawpool.csv`(完整11配置×5重复 per-rep) | ✓ 均值0.0622, seed重跑复现一致 |
| §5.3 noise floor | 0.084–0.087@2k, 0.042–0.044@10k | 纯numpy从真分布采样复算(脚本可重跑) | ✓ 11配置全验, 比值1.95–2.06 |
| §5.4 homogeneous | 0.0840 ≈ diverse 0.0849 | `data/results_homogeneous_ensemble.csv` | ✓ 实测0.0840 |
| §5.6 equal-total | ensemble差11–64% | `data/results_equal_total.csv` | ✓ (之前核验) |
| §5.7 runtime | single@10k≈14s, ens≈95s | `data/results_single_10000.csv` (runtime_sec列) | ✓ 稳态14.1s |

---

## 二、⚠️ 需要你确认/处理的事项

1. **equal_per_member 有两个版本(再现性隐患)**
   你上传过 `results_equal_per_member.csv`(ensemble均值0.0842)和 `results_equal_per_member__1_.csv`(0.0849)。
   **论文用的是0.0849那版,per-config完全吻合。** 本包`data/`里放的就是这一版(已改名为标准名)。
   → **请确认你的GitHub repo里也用这一版**,或干脆重跑一次得到单一权威数据集,避免repo里两个数字打架。

2. **Table 2 ablation 的来源run要对齐**
   ablation的full ensemble基线是0.0849(与__1_版一致)。请确认三个ablation(no_cov/no_temp/no_w)与full来自同一批run。

3. **raw-pool 全11配置完成,完整per-rep CSV已就位** ✓
   `data/results_rawpool.csv`(55行=11配置×5重复)。seed固定重跑复现的均值与论文完全一致(0.0622),可复现性已验证。

4. **唯一剩下的limitation: joint分布指标(energy distance/MMD)**
   论文只用marginal JS。§6.3已说明:对bimodal够、对ring弱。这是最后一个审稿人可能要求补的点,但已作为明牌limitation写入。不补也能投。

5. **6条未带[verified]的引用**(Brooks, Goodman-Weare, Neal1996, Lin1991, Geyer1992)
   DOI已尽力填写,投稿前请快速核对一遍。

6. **affiliation**: 当前"Independent Researcher"。如需挂JHU校友身份请自行决定。

---

## 三、本包内容

```
paper/   flowmc_paper_final.md   — 最终论文(人味重写版)
figures/ fig_control_single10000 / fig_noise_floor / fig_rawpool (各png+pdf) — 论文Figure 1/2/3
data/    7个源CSV(权威版)
code/    run_controls.py, run_rawpool_resumable.py — 控制实验脚本
```

注:论文目前用Markdown。转LaTeX(MDPI模板)是下一步,尚未做。

---

## 四、论文核心结论一句话

固定总样本预算下,单链flowMC@10000比5成员ensemble又快7倍又准24%;diversity零收益;纯拼接能追平单链,是"加权重采样+coverage/tempering/weighting"那套聚合机制主动把质量从0.065砸到0.089。建议:单链跑满预算,别用那套聚合。

## Update (honest-mechanism revision)
- `code/run_controls.py`, `code/run_rawpool.py` — scripts that produced the single@10000, homogeneous, uniform-pool and equal-total control CSVs (ESS, runtime_sec, JS all logged).
- `code/noise_floor.py` — reproduces the perfect-sampler JS floor (Figure 2); writes `data/noise_floor_curve.csv`. Anchors: ~0.085 @2,000, ~0.043 @10,000.
- `data/noise_floor_curve.csv` — the regenerated floor curve.
- Mechanism note: the ensemble's measured ESS is ~9,800 (near nominal 10,000), so its JS deficit is a *bias* from the aggregation (reweighting + weighted resampling), not a loss of effective samples. Earlier "effective-N collapse" wording has been removed throughout. Runtime corrected to single@10k≈14s, single@2k≈18s, ensemble≈92s (≈6× faster).
