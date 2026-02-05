---
title: "References"
---

## Foundational Papers

- Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- Kaplan, J., et al. (2020). **Scaling Laws for Neural Language Models**. *arXiv preprint*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

- Hoffmann, J., et al. (2022). **Training Compute-Optimal Large Language Models**. *NeurIPS 2022*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

## Parallelism Strategies

- Shoeybi, M., et al. (2019). **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. *arXiv preprint*. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

- Narayanan, D., et al. (2021). **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. *SC'21*. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

- Huang, Y., et al. (2019). **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**. *NeurIPS 2019*. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)

- Narayanan, D., et al. (2019). **PipeDream: Generalized Pipeline Parallelism for DNN Training**. *SOSP'19*. [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)

- Rajbhandari, S., et al. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. *SC'20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

- Korthikanti, V., et al. (2022). **Reducing Activation Recomputation in Large Transformer Models**. *MLSys 2023*. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)

- Liu, H., et al. (2023). **Ring Attention with Blockwise Transformers for Near-Infinite Context**. *arXiv preprint*. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

- Jacobs, S. A., et al. (2023). **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**. *arXiv preprint*. [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)

## Mixture of Experts

- Shazeer, N., et al. (2017). **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. *ICLR 2017*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

- Fedus, W., et al. (2022). **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. *JMLR*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

- Lepikhin, D., et al. (2021). **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**. *ICLR 2021*. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)

## Efficient Training

- Dao, T., et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *NeurIPS 2022*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. *arXiv preprint*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

- Micikevicius, P., et al. (2017). **Mixed Precision Training**. *ICLR 2018*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

- Micikevicius, P., et al. (2022). **FP8 Formats for Deep Learning**. *arXiv preprint*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)

- Chen, T., et al. (2016). **Training Deep Nets with Sublinear Memory Cost**. *arXiv preprint*. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)

## Large-Scale Systems

- Dubey, A., et al. (2024). **The Llama 3 Herd of Models**. *arXiv preprint*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

- DeepSeek-AI (2024). **DeepSeek-V3 Technical Report**. *arXiv preprint*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

- Jiang, A., et al. (2024). **Mixtral of Experts**. *arXiv preprint*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

- Jiang, A., et al. (2023). **Mistral 7B**. *arXiv preprint*. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

- Brown, T., et al. (2020). **Language Models are Few-Shot Learners**. *NeurIPS 2020*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

## Communication and Distributed Systems

- **NCCL: The NVIDIA Collective Communication Library**. [GitHub](https://github.com/NVIDIA/nccl)

- Thakur, R., et al. (2005). **Optimization of Collective Communication Operations in MPICH**. *International Journal of High Performance Computing Applications*.

- Rabenseifner, R. (2004). **Optimization of Collective Reduction Operations**. *ICCS 2004*.

- Dean, J., et al. (2012). **Large Scale Distributed Deep Networks**. *NeurIPS 2012*.

## Automatic Parallelization

- Zheng, L., et al. (2022). **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**. *OSDI'22*. [arXiv:2201.12023](https://arxiv.org/abs/2201.12023)

- Jia, Z., et al. (2019). **Beyond Data and Model Parallelism for Deep Neural Networks**. *MLSys 2019*. [arXiv:1807.05358](https://arxiv.org/abs/1807.05358)

## Books and Guides

- **How to Scale Your Model** (Google DeepMind/JAX team, 2025). [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/)

- **The Ultra-Scale Playbook** (Hugging Face, 2025). [huggingface.co/spaces/nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

## Hardware References

- NVIDIA (2022). **H100 Tensor Core GPU Architecture Whitepaper**.
- NVIDIA. **DGX H100 System Architecture Guide**.
- InfiniBand Trade Association. **InfiniBand Architecture Specification**.

## Classic Distributed Computing

- Lamport, L. (1978). **Time, Clocks, and the Ordering of Events in a Distributed System**. *Communications of the ACM*.

- Little, J. D. C. (1961). **A Proof for the Queuing Formula: L = λW**. *Operations Research*.

- Williams, S., et al. (2009). **Roofline: An Insightful Visual Performance Model for Multicore Architectures**. *Communications of the ACM*.
