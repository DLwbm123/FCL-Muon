# 来源与迁移边界

三份用户本地 PDF 已按任务书附录 SHA256 核对；仅阅读指定方法/附录页，未声称复现原论文训练。以下页码均为这三份 PDF 的物理页。

| 来源 | 已读取依据 | 本轮采用 | 未采用与保证边界 |
|---|---|---|---|
| SF-LoRA，21页，SHA256 743ebfcacbb5b63ef6c7371f282746c38d94b2cd8e8b499f08f41eec6d0d24a5 | p4 Eq2–3；p5 Eq4–5；p6 Eq6–12；p18 Algorithm1 | 合并残差左右 SVD、相对谱权重、软近端思想 | 原目标含两个单侧项和 gamma_s^2 双侧项，且同时修改 A/B；本轮只在固定 A 内以 gamma 罚双侧项，按通信轮执行。q=64 非原密度阈值；gamma=.1 非原论文 lambda=150 的等价设置 |
| InheritLoRA，47页，SHA256 f7eec1f956d1499882f68ffeeb4f128c2e867aa258bef6bb4afd49d73f9a99e3 | p5 Eq8–10；p6 Eq14–15；p18 B.2–B.3 | 当前参考激活的 K=A C A^T，作为同一近端目标的距离 | 不实现 GradResp、历史 reader 投影、逆 Gram 梯度预条件、per-step AdamW reset、历史 adapter 保留。原局部 LS 的一步 writer-gap 结论不能移植到本轮 CE/联邦长程效果 |
| SPACE-LoRA，17页，SHA256 3bf656b4bcdcdeb384f8994dee85bd2766916377d36efd3574e9645e9fa4ec9e | p4–5 Eq5–16；p6 Algorithm1；p17 Algorithm2 | 识别固定 A 迁移边界 | 不实现 Fisher 分配、A 硬投影或历史分支。固定 A 时 BAV=0 且 AV 满行秩会迫使 B=0，原 k<=d-r 不能保证当前 row(A) 内有非零可行解 |

RRS 保护历史输入统计，新双侧项保护历史权重残差左右方向，二者对象不同。历史谱大方向只是重要性代理，交叠也可能有正迁移；需 E0/E1/E2/E3 及固定共同状态干预检验。方法构建使用上述思想，不把本轮适配结果冒称原论文标准实现或对比方法复现。
