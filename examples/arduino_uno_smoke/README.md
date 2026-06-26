# Arduino Uno Smoke Test

This sketch is a small EdgeLearning++ smoke test for Arduino Uno-class targets.
It uses a `2 -> 2 -> 1` dense model with fixed parameters and checks the
forward output plus all parameter gradients over Serial.

The network has two trainable layers on purpose:

- the output layer propagates an input gradient to the hidden layer
- the first layer accumulates parameter gradients without writing an external
  input gradient

That exercises the templated backward API in both modes:
`backward<true, Types>` and `backward<false, Types>`.

## Files

- `arduino_uno_smoke.ino`: sketch to build against the real EdgeLearning++
  headers.

## Expected Serial Output

The exact memory numbers may change when the model planner changes, but the
test should end with:

```text
RESULT: PASS
```

The checked numerical values are:

```text
output[0] = 3.925
grad[0]  = 6.0
grad[1]  = -9.0
grad[2]  = -8.0
grad[3]  = 12.0
grad[4]  = 3.0
grad[5]  = -4.0
grad[6]  = 3.7
grad[7]  = -0.9
grad[8]  = 2.0
```

## Build Notes

EdgeLearning++ is a C++20 header-only library. Arduino Uno/Tinkercad-style AVR
builds commonly default to older C++ modes, so the build must provide:

```text
-std=gnu++20
```

and must add this repository's `include` directory to the compiler include
path. With Arduino CLI, the shape is:

```sh
arduino-cli compile \
  --fqbn arduino:avr:uno \
  --build-property compiler.cpp.extra_flags="-std=gnu++20 -I/path/to/edgelearning-cpp/include" \
  examples/arduino_uno_smoke
```

Replace `/path/to/edgelearning-cpp` with the local checkout path.

## Tinkercad

Tinkercad Circuits is useful for wiring and simple sketches, but it is not the
right validation path for this repository as-is:

- the simulator exposes selected built-in Arduino libraries
- it does not provide a normal local include path for arbitrary project headers
- the EdgeLearning++ public headers require C++20 language support

For Tinkercad specifically, use either a tiny self-contained sketch that copies
only the minimal math needed for demonstration, or validate the real library
with Arduino CLI / a local AVR toolchain and use Tinkercad only as a wiring and
Serial-output reference.
