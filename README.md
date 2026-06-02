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
3. Split features into one or more encoding blocks/layers and axis.
4. Map features to qubits, optionally using QPU connectivity and edge quality.
5. Build a parametrized Hamiltonian circuit.
6. Measure one-local and two-local Pauli observables.
7. Save the resulting projected quantum features to CSV/NPY.

The library supports ideal Aer simulation, MPS simulation, fake-backend
simulation from IBM backends, resource estimation, and real IBM Quantum execution using IBM Runtime.

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

All columns must be numeric and finite, with no missing values. The last column
is treated as the target $y$; all previous columns are encoded as input
features.

PQFMLib assumes that the dataset has already been preprocessed. In particular,
features should be normalized before running the quantum feature maps. A common
choice is standard-score normalization:

```math
h_i = \bar{x}_{f(i)},
\qquad
\bar{x}_f = \frac{x_f - \mu_f}{w_f},
```

where $\mu_f$ is the feature mean and $w_f$ is the feature standard deviation.

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

## Execution Modes

PQFMLib separates the execution mode from the hardware topology used to build
the feature map:

- `ideal=True`: runs an ideal Aer simulation. The encoded qubits are treated as
  fully connected when the map allows it, without enforcing a specific IBM QPU
  coupling map.
- `ideal=False` and `simulation=True`: simulates the circuit while using the
  structure of the selected `ibm_qpu`. For example, a Heron r2 backend uses its
  heavy-hex topology, so the available two-qubit terms are limited by the QPU
  coupling map even though execution is simulated. This mode is also useful for
  MPS simulation with `mps=True`.
- `ideal=False`, `simulation=True`, and `fakebackend=True`: simulates from the
  selected IBM backend model through `AerSimulator.from_backend`, including the
  backend topology and fake-backend behavior.
- `ideal=False` and `simulation=False`: submits the job to the selected IBM
  Quantum backend through Runtime.

## Hamiltonian Maps

### CD-Ising PQFM

`CDIsingProjectiveQFM` is a counterdiabatic-inspired Ising-glass feature map
based on the approach presented in [1]. In the present implementation, it is extended to support multi-feature
encoding per qubit by increasing the circuit depth. Each circuit block, or
layer, encodes one feature per qubit, allowing multiple features to be assigned
sequentially to the same qubit across different layers.

For each feature block, PQFMLib constructs data-dependent local fields $h_i$ 
and couplings $J_{ij}$ from the input values and the mutual-information matrix. 
The local fields are interpreted as normalized feature values.

A useful way to view the underlying Ising problem Hamiltonian is:

```math
H_{\mathrm{Ising}}(x)
= \sum_i h_i(x) Z_i
+ \sum_{(i,j)} J_{ij} Z_i Z_j .
```

The implemented circuit applies a first-order counterdiabatic term:

```math
H_{\mathrm{CD}}(x,t) =
-2 \dot{\lambda}(t) \alpha_1(t)
[
\sum_i h_i(x)Y_i +
\sum_{i \lt j} J_{ij}(Y_i Z_j + Z_i Y_j)
].
```

The first-order CD coefficient is:

```math
\alpha_1(t) =
-\frac{\sum_i h_i^2 + \sum_{i \lt j} J_{ij}^2}{4R(t)}.
```

with:

```math
R(t) =
(1-\lambda(t))^2
(\sum_i h_i^2 + 4\sum_{i\ne j} J_{ij}^2)
+ \lambda(t)^2
(
\sum_i h_i^4 +
\sum_{i\ne j} J_{ij}^4 +
6\sum_{i\ne j} h_i^2J_{ij}^2 +
6\sum_{i \lt j \lt k}(J_{ij}^2J_{ik}^2 + J_{ij}^2J_{jk}^2 + J_{ik}^2J_{jk}^2)
).
```

In the implementation, the schedule is represented by $s(t)$ and
$\dot{s}(t)$, which play the role of $\lambda(t)$ and $\dot{\lambda}(t)$.
This map is useful as a physically motivated Ising baseline: it encodes
features locally and uses feature-feature relations to activate two-qubit CD
terms on available edges.

### Heisenberg PQFM

`HeisenbergProjectiveQFM` follows the Heisenberg-style projected quantum feature
maps used in [2,3]. It prepares one random single-qubit unitary per qubit and
then applies repeated even/odd nearest-neighbor chain layers. Each normalized
scalar feature drives an isotropic two-qubit interaction:

```math
H_{\mathrm{Heisenberg}}(x)
= \sum_{i} J_i
(
X_i X_{i+1}
+ Y_i Y_{i+1}
+ Z_i Z_{i+1}
) .
```

Here $J_i$ denotes the angle assigned to the chain edge $(i,i+1)$. Using the
normalized feature convention defined in **Data Format**, the Heisenberg map
scales each assigned feature as:

```math
J_i = 2\pi \tanh\left(\frac{\bar{x}_{f(i)}}{3}\right).
```

The default observables are one-local $Z$, $X$, and $Y$ for every qubit, with an
option to include two-local diagonal observables $ZZ$, $XX$, and $YY$.

### XYZ PQFM

`XYZProjectiveQFM` is the main map in PQFMLib. It generalizes the Ising idea
from one Pauli axis to several Pauli axes and can also include cross-axis
interactions. A compact way to write the full Hamiltonian is:

```math
H_f
= \sum_i
(
h_i^{(x)} X_i
+ h_i^{(y)} Y_i
+ h_i^{(z)} Z_i
)
+ \sum_{i \lt j}
[J_{ij}^{(xx)} X_i X_j
+ J_{ij}^{(yy)} Y_i Y_j
+ J_{ij}^{(zz)} Z_i Z_j
+ J_{ij}^{(xy)}(X_iY_j + Y_iX_j)
+ J_{ij}^{(xz)}(X_iZ_j + Z_iX_j)
+ J_{ij}^{(yz)}(Y_iZ_j + Z_iY_j)].
```

The local fields $h_i^{(x)}$, $h_i^{(y)}$, and $h_i^{(z)}$ are normalized
axis-encoded features:

```math
h_i^{(a)} = \bar{x}_{f_a(i)},
\qquad
\bar{x}_{f_a(i)}
= \frac{x_{f_a(i)} - \mu_{f_a(i)}}{w_{f_a(i)}} ,
\qquad
a \in \{x,y,z\}.
```

The pairwise couplings between qubits $i$ and $j$ form a full axis-correlation
matrix:

```math
J_{ij}
=
\begin{pmatrix}
J_{ij}^{(xx)} & J_{ij}^{(xy)} & J_{ij}^{(xz)} \\
J_{ij}^{(yx)} & J_{ij}^{(yy)} & J_{ij}^{(yz)} \\
J_{ij}^{(zx)} & J_{ij}^{(zy)} & J_{ij}^{(zz)}
\end{pmatrix}.
```

With `keep_diagonal_terms=True`, the map includes the three Ising-like channels
$XX$, $YY$, and $ZZ$. With `keep_cross_terms=True`, it also includes cross-axis
correlations such as $XY$, $XZ$, and $YZ$. In feature-map terms, each qubit has
a local feature vector:

```math
\mathbf{x}_i
=
(
x_i^{(x)},
x_i^{(y)},
x_i^{(z)}
).
```

and each pair of qubits can carry a full correlation matrix between their
axis-encoded features.

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

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f_X(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f_Y(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f_Z(i).
\end{aligned}
```

This means the capacity per layer is:

```math
C_{\text{multi-axis}} = q_{\mathrm{enc}} \, |\mathrm{axes}| .
```

For example, $q_{\mathrm{enc}} = 10$ and `axes=("x", "y", "z")` can encode up to 30
features per layer. This mode is useful when the goal is to compress many
features into a small qubit register while preserving axis-specific structure.

You can also choose only two axes. For example:

```python
encoding_mode="multi_axis"
features_per_qubit=2
axes=("x", "y")
```

In this case, each qubit encodes two different features per layer, one on the
$X$ axis and one on the $Y$ axis.

### Shared-Feature Encoding

Use:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y", "z")
```

In shared-feature mode, each qubit receives one feature and reuses that same
feature across all selected axes:

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f(i).
\end{aligned}
```

The capacity per layer is:

```math
C_{\text{shared-feature}} = q_{\mathrm{enc}} .
```

This mode is useful when the experiment should compare or combine projections
of the same variable through different Pauli axes.

Shared-feature mode can also use two axes:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y")
```

Here each qubit still carries one feature, but that same feature is encoded on
both the $X$ and $Y$ axes. Thus, the feature is shared across two axes instead
of three.

## Layer Encoding

PQFMLib can encode more features than fit in one layer. It builds blocks from
the mutual-information matrix and stacks them as Hamiltonian layers. If a
dataset has $n_{\mathrm{features}}$ and one layer has capacity $C$, the number of blocks is
approximately:

```math
n_{\mathrm{blocks}}
= \lceil \frac{n_{\mathrm{features}}}{C} \rceil .
```

The layer mechanism can be mixed with axis encoding:

- `multi_axis` + layers: many different features per qubit per layer.
- `shared_feature` + layers: the same feature is reused across selected axes
  inside each layer, and additional features appear in later layers.
- diagonal + cross terms + layers: each layer can contain the $XX$, $YY$, and
  $ZZ$ Ising channels plus cross-axis couplings.

The physical repetition parameter $m$ controls how many Trotter steps are used
per block. Internally, PQFMLib builds a total circuit depth proportional to:

```math
m_{\mathrm{total}} = m \, n_{\mathrm{blocks}} .
```

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

## References

[1] Anton Simen et al., "Digitized Counterdiabatic Quantum Feature Extraction,"
arXiv:2510.13807, 2025. https://arxiv.org/abs/2510.13807

[2] Axel Ciceri et al., "Enhanced fill probability estimates in institutional
algorithmic bond trading using statistical learning algorithms with quantum
computers," arXiv:2509.17715, 2025. https://arxiv.org/abs/2509.17715

[3] Andras Ferenczi et al., "Credit Default Prediction with Projected Quantum
Feature Models and Ensembles," arXiv:2510.01129, 2025.
https://arxiv.org/abs/2510.01129

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
