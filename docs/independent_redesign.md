# Independent Redesign

EdgeLearning++ is an independent C++20 redesign of the pre-internship EdgeLearning C core.

The only allowed semantic reference baseline is:

- Repository: `Nico281102/EdgeLearning`
- Commit: `0085814908ca1b57ece4fe367361d084fd74aa3e`

This repository contains only newly written C++ source code and documentation for the redesign. It does not vendor, copy, translate, or republish the old C implementation.

The redesign excludes internship-specific code and artifacts, including PPO, reinforcement learning agents, CarRacing, Pendulum, STM32N6 application code, STAI integration, encoder runtime, host-MCU protocol, firmware probes, private datasets, generated models, thesis benchmark artifacts, post-baseline profiling extensions, and post-baseline optimized kernels.

The C baseline may be checked out locally outside this repository for regression benchmarking. Public benchmark artifacts must contain methodology and measurements only, not the old C source.

