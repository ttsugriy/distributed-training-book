---
title: "References"
---

## Foundational Papers

- **Attention Is All You Need** (Vaswani et al., 2017) - The transformer architecture
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) - Original scaling laws
- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) - Chinchilla scaling

## Parallelism Strategies

- **Megatron-LM** (Shoeybi et al., 2019) - Tensor parallelism
- **GPipe** (Huang et al., 2019) - Pipeline parallelism
- **ZeRO** (Rajbhandari et al., 2020) - Memory optimization
- **Sequence Parallelism** (Korthikanti et al., 2022) - Extending tensor parallelism
- **Ring Attention** (Liu et al., 2023) - Long context via sequence sharding

## Efficient Training

- **FlashAttention** (Dao et al., 2022, 2023) - IO-aware attention
- **Mixed Precision Training** (Micikevicius et al., 2017) - FP16 training
- **FP8 Training** (Micikevicius et al., 2022) - 8-bit training

## Large-Scale Systems

- **Llama 3** (Meta, 2024) - State-of-the-art open model training
- **DeepSeek V3** (DeepSeek, 2024) - Cost-efficient MoE training
- **Mixtral** (Mistral, 2024) - Open MoE architecture

## Communication

- **NCCL** (NVIDIA) - Collective communication library
- **Gloo** (Facebook) - CPU collective library

## Books and Surveys

- **How to Scale Your Model** (Google DeepMind, 2025) - JAX-focused scaling guide
- **The Ultra-Scale Playbook** (Hugging Face, 2025) - Practical distributed training

*[Full bibliography to be completed with proper citations]*
