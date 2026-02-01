# The Algebra of Distributed Training

**Mathematical Foundations of Large-Scale Machine Learning**

> *Every parallelism strategy exploits a mathematical property. Every communication pattern has an algebraic structure.*

## What This Book Is

This is an investigation-based guide to distributed training. Rather than explaining techniques, we **derive** them from first principles—starting with the mathematical properties that make each approach possible.

The goal: develop the intuition to reason about any distributed training problem, not just memorize existing solutions.

## Read the Book

📖 **[Read online](https://ttsugriy.github.io/distributed-training-book/)** — Free, no login required

## Who This Is For

**Capacity Engineers** and ML practitioners who want deep understanding of:

- Why tensor parallelism requires high-bandwidth interconnects
- How pipeline bubbles arise from the algebra of sequential composition
- When ZeRO stages trade communication for memory
- What makes certain operations shardable and others not

We assume you've trained models on a single GPU. We'll take you from there to reasoning about thousand-GPU clusters.

## Book Structure

### Part I: Foundations
The mental models—extended roofline, α-β communication costs, estimation as discipline.

### Part II: Scaling Laws
How compute budgets connect to model size and data through Chinchilla optimality and phase transitions.

### Part III: The Algebra of Collectives
Communication primitives as algebraic operations with formal properties.

### Part IV: Parallelism from Properties
Each strategy derived from the mathematical property it exploits:
- **Data Parallelism** ← Associativity of gradient accumulation
- **Tensor Parallelism** ← Linearity of matrix multiplication
- **Pipeline Parallelism** ← Separability of layer composition
- **Sequence Parallelism** ← Decomposability of attention
- **Expert Parallelism** ← Sparsity of MoE routing

### Part V: Memory as a Dimension
ZeRO, activation recomputation, and offloading—techniques that trade communication for memory.

### Part VI: Composition and Resilience
Combining parallelism strategies on device meshes, handling failures, configuration search.

### Part VII: Efficiency Frontiers
Gradient compression, local SGD, reduced precision, overlapping communication.

### Part VIII: Synthesis
Profiling methodology and case studies (LLaMA 3, DeepSeek, Mistral).

## Connection to The Algebra of Speed

This book is a companion to [*The Algebra of Speed*](https://ttsugriy.github.io/performance-book/), which establishes the core mathematical properties for single-machine optimization. Here we extend those ideas to distributed systems.

## Local Development

### Prerequisites
- [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/)
- Python 3.10+
- Node.js 18+ (for interactive elements)

### Setup
```bash
git clone https://github.com/ttsugriy/distributed-training-book.git
cd distributed-training-book
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build
```bash
mkdocs serve      # Live development server
mkdocs build      # Build static site
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Types of contributions:
- 🐛 Issue reports and corrections
- 📝 Improved explanations and derivations
- 📊 Interactive visualizations
- 🌍 Translations

## License

- **Content**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code**: [MIT](LICENSE-CODE)

## Acknowledgments

Inspired by:
- Pólya's *How to Solve It*
- Stepanov's *From Mathematics to Generic Programming*
- The [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)
- The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

---

*"The right parallelization follows from understanding what can be decomposed and what must be synchronized."*
