# StableHLO Build Workflow

1. Install LLVM 18 toolchain.
2. Build the custom pass shared library with `cmake --build build --target stablehlo_pass`.
3. Run `pytest tests/ir` to validate the bridge.
