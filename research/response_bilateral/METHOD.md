# 固定 Shared-A 的服务器双侧近端修正

本轮严格遵照 TASKBOOK.txt。主线保留普通 AdamW 和原 selected route；不是恢复 Muon。每通信轮在 B FedAvg 和 Head-RegMean 后、checkpoint/广播/评估前，对本任务累计 B 应用一次算子。保存、恢复、评估、task-end merge 不额外调用。

对已完成任务的合并权重减同一不可变预训练权重，得到完整 Q/V 残差 H。分别对 H 做 FP64 SVD，按 `max(m,d)*eps64*sigma1` 判数值秩；保留 min(64,rank) 个方向，alpha=(sigma/sigma1)^2。首任务空历史，不补任意零奇异方向。Q/V 分开历史谱；每层共享 A、当前 patch Gram C。C 复用原 RRS raw outer-product sum，客户端加和一次，不乘 N；排除 CLS，不使用 head Gram、query、test 或旧样本。

令 K=A C A^T，R=.99*r*K/trace(K)+.01I，K 精确为零时 R=I。固定相对负谱容差 1e-10，不增 ridge、不 clip 谱。E2 使用 R=I。解

`min_B .5 tr((B-Bhat)R(B-Bhat)^T) + gamma/2 ||sqrt(Lambda) U^T B A V sqrt(Lambda)||_F^2`。

J=A V Lambda V^T A^T，Z=U^T Bhat，一阶条件为 `(B-Bhat)R+gamma U Lambda U^T B J=0`。U 的正交补不变；U 内第 i 行解 `(R+gamma alpha_i J)y_i^T=R z_i^T`，恢复 `Bplus=Bhat+U(Y-Z)`。R 正定，目标严格凸、最优解唯一。非对易情形不能交换右乘次序，也不能断言 Euclidean ker(J) 分量或 Euclidean 范数不变/下降。零 B 仍执行 solve，保留正确 Jacobian。所有矩阵计算 FP64，正式与 functional 端点均 cast 回输入 dtype。

E0 为显式 gamma=0；E1 在自身状态计算 shadow E3，仅逐 Q/V 用其有效 BA 范数比缩放；E2 为欧氏近端；E3 为响应近端。E1 不匹配本轮增量，不是学习率控制；其最终输出不要求近端最优性或 FedAvg 可交换。若 E2 胜出，必要时补 E1E2 对照。E1 的 c 不 clip，零分母不加任意 epsilon。

新增 `Pilot.server_aggregate(proposal,head)` 在旧 fuse 保存前调用，缺省恒等。新子类调用同一个纯算子。原 `federated_response` 仅新增可选 `server_operator`，置于每短轮 B/head 聚合后、detach 前。原 RoutePilot 显式传递可选算子，原候选数、步长、历史/正交/信赖域和接受门不变。A->R/J->solve 全程保留 autograd，固定历史和参考 C detach。

完整 latest.pt 继承原状态机，额外保存本任务 U/V/alpha/C、已应用轮号及最近应用标记。新任务历史在选基前构建，pending C 建立后固定。W0 从同一预训练文件恢复；源文件 SHA 和 stat 冻结，模型 restore 不改变 W0。候选 A 改变时重新构造 J/R；当前实现不做 eig 缓存加速。基于旧客户端端点的 Head-RegMean 证书仅覆盖原求解，不覆盖后置 B 改动。

每块 SVD 成本 O(min(m,d)*m*d)，每次 map 为 O(mqr+drq+qr^3)，q<=64,r=32；额外存储 W0 和 O((m+d)q+d^2) 几何，不引入新逻辑通信字段。实际 SVD、solve/响应/训练/评估耗时及内存、数据访问和通信计数另记录，不能称零成本。

总划分始终 100 类/10 任务；`tasks=total_tasks=10`，`stop_after_task` 单独决定窗口。原本地学习率只依赖 round 与 local steps，T1 前缀复用不改变日程。各臂从同 seed T1/RNG 派生，冻结后评估全部已见类 argmax。T3 是开发窗口，T5 seeds3/4/5 仍属同一 benchmark。没有足够完整三组时不报告预定三 seed 均值。冻结决策阈值见 PROTOCOL.json；不按测试改参数、筛 seed 或降低训练量。

局部响应距离、历史交叠、近端目标和短响应只作机制证据；真实保留与塑性需看旧/新/总体准确率。现有 Head-RegMean 反向仍为 FP32 量化上的隐式代理，本轮算子 gradcheck 不构成整个 selector 的严格梯度认证。
