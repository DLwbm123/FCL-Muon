First-tier causal ablations for the completed UT64 protocol.
Statistics merge has explicit identity/full/mean_only modes. Identity forces both T=I and b=0 in deployed and selected forwards.
Pure RRS maintains independent chronological statistic histories, while reusing parameter training only after preflight proves exact isolation against standalone branches.
Frozen-LDA has zero parameter updates and processes each task's current training data once; dense and encoded scatter share features and FP32 means.
Evaluator-only old TEST summaries and conditional readouts are isolated from algorithm updates; diagnostic tasks are fixed to T2/T5/T10.
Current scope is CIFAR100 seed0. No new data downloads, H0/AUX/full-affine training, STSA or ImageNet-R are launched here.
