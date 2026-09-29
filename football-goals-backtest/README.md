# Football Goals Backtest

用于回测五套 Total Goals Engine 候选权重。

## 回测范围

测试赛季：
- 2023/24
- 2024/25
- 2025/26

热身/历史窗口：
- 2021/22
- 2022/23

联赛：
- EPL = E0
- LaLiga = SP1
- Serie A = I1
- Bundesliga = D1
- Ligue 1 = F1
- Primeira Liga = P1
- Eredivisie = N1

## 方法

严格按比赛日期排序，预测每一场比赛时只使用该场比赛之前已经发生的数据。

五套权重冻结在 `config/models.yml`：
- A_balanced
- B_xg_core
- C_tempo_tactical
- D_lineup_context
- E_conservative

## 数据说明

第一版使用 Football-Data 历史 CSV。

Football-Data 的标准比赛文件包含赛果、射门、射正、角球和赔率等字段，但不是完整的逐场 xG 数据。因此代码中的 `xg_proxy` 是“射门/射正代理”，不会冒充真实 xG。

## 输出

运行完成后：
- `results/predictions.csv`
- `results/summary.csv`
- `results/by_league.csv`
- `results/by_season.csv`
- `results/run_summary.md`

## GitHub Actions

进入仓库的 **Actions**：
1. 选择 `Football Goals Backtest`
2. 点击 `Run workflow`
3. 等运行结束
4. 在本次运行底部下载 Artifact：`football-goals-backtest-results`

## 下一阶段

拿到第一轮结果后，再接入真实 match-level xG 数据做第二轮回测，并在独立开发窗口上优化权重，不对测试窗口反复调参。
