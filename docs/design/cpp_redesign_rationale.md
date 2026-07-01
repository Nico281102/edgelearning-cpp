# C++ Redesign Rationale

The C++ rewrite exists to test whether the embedded training runtime benefits
from moving network structure from runtime descriptors into the type system.
The original C runtime remains a useful semantic baseline: static arenas, flat
parameter buffers, sample-wise training, and explicit embedded constraints.

The C++ design keeps those constraints and changes where decisions are made:

- topology, tensor shapes, layer sizes, backend policy, precision policy, and
  arena size are compile-time facts;
- unsupported shapes or insufficient statically sized arenas fail during
  compilation when the information is available in the type;
- small Dense/Conv2D models can be specialized without a runtime layer factory
  or dynamic graph descriptor;
- backend selection is explicit in the model type, so a generic scalar path and
  an M55-specialized path can be compared with the same public API;
- model storage can be owned by the model or supplied as an external static
  arena for linker-controlled placement.

The main hypothesis is not that C++ is faster by itself. The hypothesis is that
a C++20 type-level model can remove some runtime interpretation overhead and
make backend specialization easier to express while preserving the static
memory discipline of the embedded C implementation.

This also makes ablation experiments cleaner:

- `EL-C M55` measures the legacy C runtime and M55 backend used as the private
  reference path;
- `EL++ generic scalar` measures the public C++ API without specialized M55
  kernels;
- `EL++ M55` measures the same public C++ API with the M55 backend policy;
- `RLTools Generic` provides an external TinyRL-style static C++ baseline with
  RLTools neural-network APIs, batch-256 tensors, and the same fast RLTools
  network selection used by the RL firmware path.

The comparison separates language/runtime structure from backend specialization.
If `EL++ generic scalar` is close to `EL-C M55`, the static C++ representation
already removed relevant overhead. If `EL++ M55` improves mainly over
`EL++ generic scalar`, the gain is primarily backend specialization. If RLTools
is competitive on small models, that gives a useful external check against an
independent static-template implementation.

Modern agentic AI tools make it easier to study advanced C++ features and use
them deliberately. In this library those features are justified only when they
reduce runtime state, improve static checking, or make embedded deployment more
predictable.
