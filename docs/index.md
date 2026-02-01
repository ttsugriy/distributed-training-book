---
title: "Preface"
---

**Read online: [ttsugriy.github.io/distributed-training-book](https://ttsugriy.github.io/distributed-training-book/)**

<figure>

![The Algebra of Distributed Training](images/distributed_training_book.png)

<figcaption>Cover illustration for <em>The Algebra of Distributed Training</em>.</figcaption>

</figure>

<div class="chapter-opener" markdown>
Every parallelism strategy exploits a mathematical property. Every communication pattern has an algebraic structure. Every efficiency gain traces to a fundamental insight about what can be decomposed and what must be synchronized.
</div>

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

## The Three Invariants

Every distributed training system obeys three invariants. All strategies are trade-offs between them.

1. **Memory**: What must fit where?
2. **Compute**: How much work per step can the hardware sustain?
3. **Communication**: What data must cross which links, how often?

Throughout the book we return to these invariants. When you get stuck, ask which invariant is violated and which lever fixes it.

## A Simple Decision Procedure

When designing or debugging a system, use this order:

1. **Fit**: Does the model state + activations fit? If not, add sharding or recomputation.
2. **Keep GPUs busy**: Are you compute-bound or memory-bound? If not, improve kernels, precision, or batch.
3. **Hide communication**: If comm dominates, increase intensity, overlap, or change topology.

Every chapter can be read as a response to a failure in one of these steps.

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

| Part | Title | Chapters | Focus |
|------|-------|----------|-------|
| I | Foundations | 1–6 | Mental models—roofline, communication costs, estimation |
| II | Scaling Laws | 7–10 | Compute budgets, model sizing, data sizing |
| III | The Algebra of Collectives | 11–13 | Communication primitives as algebraic operations |
| IV | Parallelism from Properties | 14–18 | Deriving each strategy from mathematical foundations |
| V | Memory as a Dimension | 19–22 | Techniques that trade communication for memory |
| VI | Composition and Resilience | 23–27 | Combining strategies and handling failures |
| VII | Efficiency Frontiers | 28–31 | Compression, reduced precision, and overlapping |
| VIII | Synthesis | 32–37 | Case studies and real-world applications |

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
