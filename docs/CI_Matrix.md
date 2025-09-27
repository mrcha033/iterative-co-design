# CI Matrix

| Job | Frequency | Notes |
| --- | --- | --- |
| `pytest` | Every PR | Runs unit and integration suites |
| `stablehlo-build` | Nightly | Validates IR bridge |
| `lint` | Every PR | Formatting and static checks |
