# PQFMLib

PQFMLib is a Python library for Projected Quantum Feature Maps (PQFMs) for
tabular machine learning. It builds quantum circuits from numerical datasets,
executes them with Qiskit/Aer or IBM Quantum Runtime, and exports projected
expectation values as classical feature matrices that can be used by standard
machine-learning pipelines.

The central research object in this repository is `XYZProjectiveQFM`: a
feature-map Hamiltonian that combines ideas from CD-Ising encodings and
Heisenberg-style multi-axis interactions. The library also includes
`CDIsingProjectiveQFM` and `HeisenbergProjectiveQFM` as comparison maps.

## What The Library Does

PQFMLib turns a numeric CSV dataset into quantum features through this workflow:

1. Load a tabular dataset whose last column is the target.
2. Estimate pairwise feature relations with a normalized mutual-information
   matrix $J$.
3. Split features into one or more encoding blocks/layers.
4. Map features to qubits, optionally using QPU connectivity and edge quality.
5. Build a parametrized Hamiltonian circuit.
6. Measure one-local and two-local Pauli observables.
7. Save the resulting projected quantum features to CSV/NPY.

The library supports ideal Aer simulation, MPS simulation, fake-backend
simulation from IBM backends, resource estimation, and IBM Runtime execution.

## Installation

```bash
pip install -e .
```

or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Data Format

Datasets are loaded from:

```text
<data_dir>/<name_file>.csv
```

All columns must be numeric and finite. The last column is treated as the target
$y$; all previous columns are encoded as input features.

## Quick Example

```python
from pqfmlib import XYZProjectiveQFM

qfm = XYZProjectiveQFM(
    name_file="my_dataset",
    data_dir="./data",
    output_root="./results",
    ideal=True,
    q_enc=4,
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    measure_cross_observables=True,
)

result = qfm.run()
print(result["Xq_all_raw"].shape)
print(result["csv_path"])
```

Runnable examples using the `Toxicity_preprocessed_shuffled` dataset are
available in `examples/`.

## Hamiltonian Maps

### CD-Ising PQFM

`CDIsingProjectiveQFM` is a counterdiabatic-inspired Ising-glass feature map.
For each block of features, PQFMLib builds data-dependent local fields $h_i$
and couplings $J_{ij}$ from the input values and the mutual-information matrix.

A useful way to view the underlying Ising problem Hamiltonian is:

$$
H_{\mathrm{Ising}}(x)
= \sum_i h_i(x) Z_i
+ \sum_{(i,j)} J_{ij} Z_i Z_j .
$$

The implemented circuit applies a first-order counterdiabatic correction with
generators:

$$
H_{\mathrm{CD}}^{(1)}(x)
= \sum_i h_i(x) Y_i
+ \sum_{(i,j)} J_{ij} \left( Y_i Z_j + Z_i Y_j \right) .
$$

The coefficient is computed from the schedule functions $s(t)$, $\dot{s}(t)$,
and the first-order CD factor $\alpha_1$. This map is useful as a physically
motivated Ising baseline: it encodes features locally and uses feature-feature
relations to activate two-qubit CD terms on available edges.

### Heisenberg PQFM

`HeisenbergProjectiveQFM` follows the original notebook-style Heisenberg map.
It prepares one random single-qubit unitary per qubit and then applies repeated
even/odd nearest-neighbor chain layers. Each scalar feature drives an isotropic
two-qubit interaction:

$$
H_{\mathrm{Heisenberg}}(x)
= \sum_{(i,i+1)} \theta_e(x)
\left(
X_i X_{i+1}
+ Y_i Y_{i+1}
+ Z_i Z_{i+1}
\right) .
$$

Feature angles can be scaled as:

$$
\theta(x) = 2\pi \tanh\left(\frac{x}{3}\right) .
$$

The default observables are one-local $Z$, $X$, and $Y$ for every qubit, with an
option to include two-local diagonal observables $ZZ$, $XX$, and $YY$.

### XYZ PQFM: Three Isings In One Hamiltonian

`XYZProjectiveQFM` is the main map in PQFMLib. It generalizes the Ising idea
from one Pauli axis to several Pauli axes and can also include cross-axis
interactions. In its diagonal form, it is equivalent to placing up to three
Ising-like Hamiltonians in the same circuit:

$$
H_{\mathrm{diag}}(x)
= H_X(x) + H_Y(x) + H_Z(x) .
$$

$$
\begin{aligned}
H_X(x)
&= \sum_i x_{f_X(i)} X_i
+ \sum_{(i,j)} J_{f_X(i), f_X(j)} X_i X_j, \\
H_Y(x)
&= \sum_i x_{f_Y(i)} Y_i
+ \sum_{(i,j)} J_{f_Y(i), f_Y(j)} Y_i Y_j, \\
H_Z(x)
&= \sum_i x_{f_Z(i)} Z_i
+ \sum_{(i,j)} J_{f_Z(i), f_Z(j)} Z_i Z_j .
\end{aligned}
$$

With `axes=("x", "y", "z")` and `keep_diagonal_terms=True`, the map can use
$XX$, $YY$, and $ZZ$ terms together. This gives three axis-specific Ising
channels inside one feature map.

When `keep_cross_terms=True`, the map also enables interactions between
different axes:

$$
\begin{aligned}
H_{\mathrm{cross}}(x)
= \sum_{(i,j)} \Big[
&J_{f_X(i), f_Y(j)} X_i Y_j
+ J_{f_X(i), f_Z(j)} X_i Z_j \\
&+ J_{f_Y(i), f_X(j)} Y_i X_j
+ J_{f_Y(i), f_Z(j)} Y_i Z_j \\
&+ J_{f_Z(i), f_X(j)} Z_i X_j
+ J_{f_Z(i), f_Y(j)} Z_i Y_j
\Big] .
\end{aligned}
$$

The full XYZ Hamiltonian used by the circuit is:

$$
H_{\mathrm{XYZ}}(x)
= \sum_a \sum_i x_{f_a(i)} \sigma_i^a
+ \sum_{(i,j)} \sum_{(a,b)}
J_{f_a(i), f_b(j)} \sigma_i^a \sigma_j^b .
$$

where $a,b$ are selected Pauli axes from $x$, $y$, and $z$. The pair set
contains diagonal pairs such as $XX$, $YY$, $ZZ$, cross pairs such as $XY$,
$XZ$, $YZ$, or both, depending on the chosen flags.

## Axis Encoding Modes

`XYZProjectiveQFM` has two encoding modes.

### Multi-Axis Encoding

Use:

```python
encoding_mode="multi_axis"
features_per_qubit=3
axes=("x", "y", "z")
```

In multi-axis mode, each `(qubit, axis)` slot can receive a different tabular
feature:

$$
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f_X(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f_Y(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f_Z(i).
\end{aligned}
$$

This means the capacity per layer is:

$$
C_{\text{multi-axis}} = q_{\mathrm{enc}} \, |\mathrm{axes}| .
$$

For example, $q_{\mathrm{enc}} = 10$ and `axes=("x", "y", "z")` can encode up to 30
features per layer. This mode is useful when the goal is to compress many
features into a small qubit register while preserving axis-specific structure.

### Shared-Feature Encoding

Use:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y", "z")
```

In shared-feature mode, each qubit receives one feature and reuses that same
feature across all selected axes:

$$
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f(i).
\end{aligned}
$$

The capacity per layer is:

$$
C_{\text{shared-feature}} = q_{\mathrm{enc}} .
$$

This mode is useful when the experiment should compare or combine projections
of the same variable through different Pauli axes.

## Layer Encoding

PQFMLib can encode more features than fit in one layer. It builds blocks from
the mutual-information matrix and stacks them as Hamiltonian layers. If a
dataset has $n_{\mathrm{features}}$ and one layer has capacity $C$, the number of blocks is
approximately:

$$
n_{\mathrm{blocks}}
= \left\lceil \frac{n_{\mathrm{features}}}{C} \right\rceil .
$$

The layer mechanism can be mixed with axis encoding:

- `multi_axis` + layers: many different features per qubit per layer.
- `shared_feature` + layers: the same feature is reused across selected axes
  inside each layer, and additional features appear in later layers.
- diagonal + cross terms + layers: each layer can contain $XX$/$YY$/$ZZ$ Ising
  channels and cross-axis couplings.

The physical repetition parameter $m$ controls how many Trotter steps are used
per block. Internally, PQFMLib builds a total circuit depth proportional to:

$$
m_{\mathrm{total}} = m \, n_{\mathrm{blocks}} .
$$

## Important XYZ Options

```python
XYZProjectiveQFM(
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    n_keep_terms=None,
    measure_all_zz=False,
    measure_cross_observables=False,
    use_edge_error=True,
    rho_thr=0.0,
)
```

Key options:

- `axes`: choose any unique subset of `("x", "y", "z")`.
- `encoding_mode`: choose `"multi_axis"` or `"shared_feature"`.
- `keep_diagonal_terms`: include same-axis terms such as $XX$, $YY$, $ZZ$.
- `keep_cross_terms`: include mixed-axis terms such as $XY$, $XZ$, $YZ$.
- `n_keep_terms`: keep only the strongest interaction terms in a layer.
- `measure_cross_observables`: measure cross-axis observables in addition to
  default one-local and diagonal observables.
- `use_edge_error`: when running on IBM backends, include edge quality in the
  feature-to-qubit assignment heuristic.
- `use_fixed_blocks` and `use_fixed_phys_nodes`: reproduce a previously saved
  feature assignment or physical layout.

## IBM Quantum Runtime

For real IBM Quantum execution, set a token before running:

```bash
export QISKIT_IBM_TOKEN="your-token"
```

On Windows PowerShell:

```powershell
$env:QISKIT_IBM_TOKEN="your-token"
```

Then set:

```python
ideal=False
simulation=False
ibm_qpu="ibm_kingston"
```

Generated `job_meta.json` files can be retrieved with the scripts in
`scripts/`:

- `get_job_xyz.py`
- `get_job_cd_ising.py`
- `get_job_heisenberg.py`

## Repository Structure

```text
pqfmlib/
|-- examples/
|-- pqfmlib/
|   |-- core/
|   |-- hardware/
|   |-- maps/
|   |-- runners/
|   `-- utils/
|-- scripts/
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## GitHub Notes

The repository is configured to keep generated artifacts out of version
control. Local datasets, IBM tokens, notebooks, experiment outputs,
`results/`, `old_general_results/`, `.venv/`, caches, `.qpy` circuits, `.npy`
arrays, and large local research scripts are ignored by `.gitignore`.

The intended public package surface is:

- `pqfmlib/`: reusable library code.
- `examples/`: simple scripts showing how to run each map on Toxicity data.
- `scripts/`: optional IBM Runtime retrieval helpers.
- `pyproject.toml`, `requirements.txt`, and `README.md`.
