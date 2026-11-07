# 给 Pro 的审阅入口：固定 Shared-A 的累计 B 双侧近端

请先看本页，再看[最终报告](FINAL_REPORT_ZH.md)、[每 seed 配对表](PAIRED_T5.csv)及源码。此次希望分析方法与证据，不预设近端必须有效。

## 已得到的结论

普通 AdamW + 原 selected route 保留。完成 seed3/4 两组前五任务四臂对照，实际 n=2；八条轨迹均正常退出、最终重载混淆一致。seed0 的四臂 T3 仅作开发窗口。所有 T3/T5 都是固定100类、每任务10类、共10任务划分的前缀，不是重新划分全部类别。

|方案|前5任务 AIA（两 seed 均值）|T5 最终准确率|
|---|---:|---:|
|E0 原 selected route|92.8040%|87.5700%|
|E1 自身 shadow-E3 幅值控制|92.7495%|87.4600%|
|E2 欧氏近端|92.7422%|87.5400%|
|E3 响应近端|92.8122%|87.5400%|

E3−E0 的最终差依 seed 为 −0.16/+0.10pp，均值 −0.03pp；AIA 差均值 +0.0082pp。E3−E2 最终差 −0.04/+0.04pp，均值0。E3−E1 最终差 −0.10/+0.26pp，不支持稳定方向优势。没有预定三 seed 结果，也没有当前版本的 T6–T10 结果。

## 源码导航

1. [方法与数学目标](../../research/response_bilateral/METHOD.md)、[来源适配边界](../../research/response_bilateral/SOURCE_ADAPTATION.md)、[冻结协议](../../research/response_bilateral/PROTOCOL.json)。SF-LoRA 的历史双侧对象与 InheritLoRA 的当前输入响应距离进入同一个二次目标；不是原论文完整复现。
2. [prox.py](../../research/response_bilateral/prox.py)：非对易右侧度量下的精确求解、E0/E1/E2/E3及诊断。
3. [pilot.py](../../research/response_bilateral/pilot.py)：固定历史谱与当前参考Gram、正式服务器累计B修正、探针与恢复标记。
4. [route response.py](../../research/route_compare_20261102/response.py)与[route pilot.py](../../research/route_compare_20261102/pilot.py)：在每短通信轮调用与正式部署相同的可微算子。
5. [head pilot.py](../../research/head_regmean/pilot.py)：`server_aggregate` hook 在保存端点之前执行；默认恒等。头聚合保留既有隐式导数代理，不宣称全链严格梯度认证。
6. [test_prox.py](../../research/response_bilateral/test_prox.py)、[test_lifecycle.py](../../research/response_bilateral/test_lifecycle.py)及[验证摘要](VALIDATION_SUMMARY.json)：独立dense oracle、梯度、零分支和真实小状态机恢复。
7. [supervisor.py](../../research/response_bilateral/supervisor.py)、[worker.py](../../research/response_bilateral/worker.py)、[运行控制适配](../../research/response_bilateral/operations/README.txt)。原冻结科学源码保持，资源门与已审计文件竞争恢复使用独立shim。

41个科学依赖文件按本轮实际冻结源码复制，指纹见[FROZEN_PROTOCOL_PUBLIC.json](FROZEN_PROTOCOL_PUBLIC.json)。其余依赖沿用本分支基础提交。公开清单是路径脱敏后的审阅元数据，不是原始服务器 `FROZEN_PROTOCOL.json` 的替代品。

## 请重点判断

1. 当前二次目标、固定 A 的约束以及非对易 R/J 求解是否正确；功能短训练和正式服务器规则是否确实一致？特别区分累计B修正和当轮增量修正。
2. 历史权重残差的双侧交叠是否是合理保护对象？当前输入的patch响应距离是否足以支持所期待的新旧类权衡？它们与旧类损失/准确率间有什么尚未验证的缺口？
3. 后续任务的E3单块相对修正最大约12.18%，不是全程恒等；目标和罚项下降却没有稳定精度收益，最合理的解释有哪些？哪些可以只读已有记录判断，哪些需要新的预注册证据？
4. E1仅匹配各矩阵累计有效BA范数，不匹配实际当轮步长，也不保证不同轨迹以后范数一致。现有结果能支持多大程度的方向归因？
5. 不同臂的functional response不同，后续选择的A也可能不同。完整轨迹是整套算法效应；[共同状态小诊断](COMMON_INTERVENTION_READBACK_ZH.md)样本很少，不能冒充长期独立因果证据。
6. [E0回读限制](BASELINE_READBACK_NOTE_ZH.md)：历史A1 seed0/T2为94.65%，本轮E0为94.35%。小状态机与ViT合成回归通过，完整真实轨迹逐值回归未建立。请评估该不确定性对结论的影响，而不是用历史高分替换当前基线。
7. 审核当前选基、分类头与评估口径是否一致；历史配置中保留的推理规则文字不能替代实际代码。当前结果按全部已见类raw argmax，无task id，不把query上的选择收益当独立效果证据。
8. [预算](BUDGET_FINAL.json)：计费66,699.436 GPU秒；当时剩余GPU约19,700.564秒、墙钟18,106.032秒。下一完整组固定估算为17×1.4×最慢任务3,505.010秒≈83,419.236秒，因此未启动seed5。请评价预算设计是否过度保守；这是原规则的准入结束，不是计算时间全部耗尽。
9. 在这些负/近零结果下，这个具体模块是否值得保留？若证据不足，请明确指出，而不是为得到正结果建议任意增加模块、挑seed或按已看测试成绩调参。

原始CIFAR测试集此前已用于开发；seed3/4仍属于同一benchmark，不是全新盲测试。冻结后没有调gamma/ridge/rank、降低epochs/clients或删选seed。

## 用户关心的完整十任务 LoRM 对照

本次整理时补查了已发布的 `codex/head-regmean-pilot-release@f0fca03`。该分支有早期Head-RegMean与LoRM的**两个seed完整十任务最终结果**，比旧工作树中“LoRM seed1还在训练”的快照更新。详见[两seed十任务最终报告](HISTORICAL_T10_TWO_SEED_FINAL_ZH.txt)和[结构化统计](HISTORICAL_T10_TWO_SEED_FINAL.json)。

|历史版本，完整10任务，两seed均值|早期Head-RegMean|LoRM|Head−LoRM|
|---|---:|---:|---:|
|raw AIA|86.9614%|85.6709%|+1.2905pp|
|双方预固定去均值 AIA|91.0139%|88.3271%|+2.6868pp|
|raw 最终准确率|79.5250%|80.7150%|−1.1900pp|
|双方预固定去均值最终准确率|86.1100%|83.9950%|+2.1150pp|

这些是**早期版本**，不是本轮AdamW响应选基/近端的十任务成绩。读出方式、任务长度和版本必须分开。早期方法的T2–T10平均新类学习仍落后LoRM，不能只选AIA或T10新类优势声称全面胜出。用户允许跨seed看历史参考，但未授权绕过本轮冻结的扩展门。

## 公开边界与运行状态

本轮已结束，原监测已移除，无活跃自身worker。2026-11-07用户明确要求推送源码和报告，此次发布授权覆盖原“默认仅本地交付”，但不改变实验冻结配置，也不授权新实验。

本分支发布科学依赖源码、测试、配置、聚合矩阵、配对表、成本和故障恢复摘要；不含数据、权重、客户端状态、逐样本预测、原始私有日志或第三方论文PDF。本地报告中的 `source.zip` 指原本地交付；公开分支以可浏览的源码目录替代。完整旧服务器账本/检查点与未用到的未提交研究支线没有一并发布。

CPU数学/小状态机测试入口为 `python -m unittest research.response_bilateral.test_prox research.response_bilateral.test_lifecycle -v`，需项目已有PyTorch环境。正式训练工厂依赖用户自有CIFAR、预训练文件和原始运维冻结账本；本公开包不是一个会自动下载私有资产或重新启动旧账本的训练入口。
