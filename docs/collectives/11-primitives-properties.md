---
title: "Collective Primitives and Properties"
subtitle: "The Building Blocks of Distributed Communication"
---

::: {.chapter-opener}
Every distributed training algorithm is built from a small set of collective operations. Understanding their algebraic properties reveals why certain compositions work and others fail.
:::

::: {.investigation-question}
**The Question**: Why is AllReduce = ReduceScatter ∘ AllGather? What properties must hold for this decomposition to be valid? What happens when these properties are violated?
:::

## The Primitives

| Primitive | Input | Output | Volume |
|-----------|-------|--------|--------|
| Broadcast | 1 → N | same on all | n |
| Reduce | N → 1 | sum (or op) | n |
| AllReduce | N → N | sum on all | 2n |
| Scatter | 1 → N | different shards | n |
| Gather | N → 1 | concatenated | n |
| AllGather | N → N | all have all | n(P-1) |
| ReduceScatter | N → N | each has shard of sum | n |
| AlltoAll | N → N | transpose | n |

## Algebraic Properties

### Associativity

Reduction operations require associativity:
$$(a \oplus b) \oplus c = a \oplus (b \oplus c)$$

Floating-point addition is *not* perfectly associative. This causes reproducibility issues.

### Commutativity

Some algorithms require:
$$a \oplus b = b \oplus a$$

Important for determining valid reduction orders.

### The Decomposition Theorem

**Theorem**: For associative, commutative reduction operator ⊕:
$$\text{AllReduce} = \text{AllGather} \circ \text{ReduceScatter}$$

*[Proof to be completed]*

## Implementation Patterns

### Ring AllReduce

*[Algorithm and analysis to be completed]*

### Tree AllReduce

*[Algorithm and analysis to be completed]*

## Exercises

*[To be completed]*
