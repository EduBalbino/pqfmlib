# XYZ numerical comparison: Aer and TileLang

## Validation status

All table rows passed the shape, hot-output, and Aer-comparison assertions before the test saved their artifacts.

## Validated implementation

- `xyz_tilelang.py` has 1,438 lines. The cleanup baseline was 1,757 lines (319 lines removed).
- Qubits 1 through 12 use the small-state kernel.
- Qubit count 18 uses a specialized 9+9 cut with Schmidt rank 8.
- Its first three crossing gates expand exactly from rank 1 to rank 8. The remaining crossing gates use structured rank-8 updates.
- Qubit count 18 measures each 4,096-value tile with two 64 by 64 Tensor Core transforms and retains only the requested coefficients.
- The generic q13+ fallback and the unused BigBang kernels were removed.
- Qubit counts 13 through 17 and 19 or greater are not supported.

## Experiment

The test uses the IBM Kingston layout. Aer uses 4,096 shots. Each input has 36 features. The XYZ map uses shared features and a diagonal Hamiltonian. The test measures all same-axis observables.

The Aer comparison uses an absolute tolerance of 0.08. The repeated hot output uses an absolute tolerance of 1e-06.

## Timing

The end-to-end times include setup. The hot time does not include setup or compilation. It is the median time for five runs of 128 rows. The hot batch repeats the eight validated input rows 16 times.

## Results

| Status | Qubits | Hot rows | Hot median (ms) | Hot rows/s | Rows | Outputs | Aer end-to-end (s) | TileLang end-to-end (s) | Max difference | Mean difference | RMSE |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS | 6 | 128 | 1.000 | 128033.9 | 8 | 63 | 12.248 | 7.941 | 0.048799 | 0.012386 | 0.015281 |
| PASS | 12 | 128 | 6.655 | 19233.9 | 8 | 234 | 11.051 | 7.454 | 0.051421 | 0.012258 | 0.015458 |
| PASS | 18 | 128 | 50.817 | 2518.8 | 8 | 513 | 15.887 | 47.520 | 0.055199 | 0.012985 | 0.016237 |
