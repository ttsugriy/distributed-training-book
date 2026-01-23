---
title: "Preface"
---

::: {.chapter-opener}
Every parallelism strategy exploits a mathematical property. Every communication pattern has an algebraic structure. Every efficiency gain traces to a fundamental insight about what can be decomposed and what must be synchronized.
:::

## Why This Book

Training large models is no longer optional knowledge. What was once the domain of a few research labs is now the daily work of thousands of engineers. Yet most resources either stay at the surface ("use FSDP") or dive into implementation details without explaining *why* things work.

This book takes a different path: **derive, don't explain**.

We start from first principles—mathematical properties like associativity, linearity, and separability—and show how each parallelism strategy follows inevitably from these foundations. When you understand *why* tensor parallelism requires high-bandwidth interconnects (linearity of matrix multiplication, nonlinearity of activations), you can reason about new architectures that don't yet have tutorials.

## Who This Book Is For

This book is for **Capacity Engineers**—the people who make large-scale training actually work. You might be:

- An ML engineer scaling training beyond a single node
- A systems engineer designing infrastructure for AI workloads
- A researcher who needs to understand the systems beneath your models
- A student preparing for a career in large-scale ML

We assume you understand neural networks and have trained models on a single GPU. We'll take you from there to reasoning about thousand-GPU clusters.

## The Investigation-Based Approach

Each chapter begins with a **question**—a concrete problem that motivates the investigation:

> *"Our gradient tensor is 10GB. We have 256 GPUs. How do we synchronize without drowning in communication?"*

We don't hand you the answer. We explore the problem space, identify the mathematical structure that enables a solution, and derive the technique step by step. When we reach the standard algorithm, you'll understand not just *what* it does but *why* it must be that way.

This approach is inspired by Pólya's *How to Solve It* and Stepanov's *From Mathematics to Generic Programming*. The goal is not to memorize techniques but to develop the intuition to derive them yourself.

## What You'll Learn

By the end of this book, you'll be able to:

1. **Analyze** any distributed training setup using extended roofline models
2. **Derive** parallelism strategies from the mathematical properties they exploit
3. **Compose** multiple parallelism dimensions into efficient configurations
4. **Estimate** throughput, memory usage, and communication costs from first principles
5. **Debug** performance problems by identifying which ceiling you're hitting
6. **Design** training configurations for new models and hardware

## How to Read This Book

The book is structured in eight parts that build on each other:

- **Part I: Foundations** establishes the mental models—roofline, communication costs, estimation
- **Part II: Scaling Laws** connects compute budgets to model and data sizing
- **Part III: The Algebra of Collectives** treats communication primitives as algebraic operations
- **Part IV: Parallelism from Properties** derives each strategy from mathematical foundations
- **Part V: Memory as a Dimension** covers techniques that trade communication for memory
- **Part VI: Composition and Resilience** combines strategies and handles failures
- **Part VII: Efficiency Frontiers** explores compression, reduced precision, and overlapping
- **Part VIII: Synthesis** applies everything to real case studies

Read sequentially for the full derivation experience, or jump to specific chapters when you need them.

## Connection to *The Algebra of Speed*

This book is a companion to [*The Algebra of Speed: Mathematical Foundations of Computational Performance*](https://ttsugriy.github.io/performance-book/). That book establishes the core properties (associativity, separability, sparsity, locality, redundancy, symmetry) and applies them to single-machine optimization.

Here we extend those ideas to distributed systems, where communication costs introduce a new dimension to the optimization landscape. The thesis remains the same: **every optimization traces to a mathematical property**.

## Acknowledgments

This book builds on the work of many researchers and engineers who have developed and documented distributed training techniques. Particular thanks to:

- The JAX team for [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/)
- The Hugging Face team for the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- The authors of landmark papers on tensor parallelism, pipeline parallelism, ZeRO, and FlashAttention
- The open-source community that makes this knowledge accessible

---

Let's begin.
