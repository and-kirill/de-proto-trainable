# Learning LDPC codes with quantized density evolution

PyTorch implementation of gradient-based LDPC protograph optimization using
quantized density evolution (DE) over a relaxed parity-check matrix.

The project accompanies the paper:

> G. Shutkov, D. Artemasov, A. Frolov, P. Rybin, and K. Andreev,
> **“Learning LDPC codes with quantized density evolution over relaxed
> protographs,”** arXiv:2607.08484, 2026.

- [Paper on arXiv](https://arxiv.org/abs/2607.08484)

## License

The source code is licensed under the [MIT License](LICENSE).

## Method overview

The optimized object is a relaxed protograph

```text
Ω ∈ [0, 1]^(m × n).
```

An entry `ωᵢⱼ` is interpreted as the Bernoulli probability that the
corresponding protograph edge is present. Thus, `Ω` represents an ensemble of
binary protographs rather than merely being an auxiliary continuous matrix.
An integer entry fixes an edge as absent or present; every non-integer entry
adds uncertainty to the final binary ensemble.

For a selected decoder, the code performs deterministic density evolution on
quantized LLR probability mass functions. The channel is binary-input AWGN
with BPSK, and all-zero-codeword symmetry is assumed. LLRs are clipped to
`[-llr_max, llr_max]` and discretized with resolution `1 / llr_scale`.

The training objective is

```text
loss(Ω) = log BER_DE(Ω),
```

where `BER_DE` is the bit-error rate predicted after a configured number of
flooding decoder iterations. PyTorch differentiates through the complete
unrolled DE computation.

The relaxed protograph is updated outside the autograd graph by projected
gradient descent:

```text
Ω(t + 1) = clip_[0,1](Ω(t) - learning_rate × ∇Ω loss(Ω(t))).
```

For normalized min-sum (NMS), the paper shows that relaxed DE computes the
ensemble-averaged DE performance of the associated Bernoulli protograph
ensemble. The SPA implementation and joint optimization of NMS column scales
included in this repository are experimental extensions; the corresponding
NMS ensemble-average theorem does not directly apply to interpolated SPA.

## Training pipeline

One optimizer epoch has the following stages:

1. `ParameterUpdater` applies the previous detached gradient to the last
   accepted parameters and enforces their constraints. PCM entries are clipped
   to `[0, 1]`; trainable NMS scales are constrained to be non-negative.
2. A fresh SPA or NMS density-evolution instance is created. This creates a new
   PyTorch computation graph for the current parameters.
3. DE propagates quantized message PMFs for `n_iter` flooding iterations and
   returns `log(BER)`.
4. If BER is at or below `ber_threshold`, the SNR is decreased by `snr_step`.
   The epoch is not committed, the learning rate is reset, and gradients from
   the previous SNR are discarded.
5. Otherwise, `loss.backward()` computes gradients for the relaxed PCM and,
   in joint NMS mode, the column normalization scales.
6. Changes in PCM gradient signs are passed to `LearningRateScheduler`. A
   sharp change can reject the candidate and roll back the learning rate.
7. An accepted candidate is detached from PyTorch, committed by
   `ParameterUpdater`, and written to the history directory.
8. `ConvergenceCriterion` evaluates the accepted state.

The optimizer stops when any convergence condition is met:

- the PCM is at an integer boundary and its gradient points outside the
  feasible box;
- the configured maximum number of accepted epochs is reached;
- the effective ensemble size does not decrease for the configured patience.

The effective ensemble size is monitored through

```text
log2(# matrices) = Σᵢⱼ h₂(ωᵢⱼ),
```

where `h₂` is binary entropy. It is zero for a fully binary protograph and
increases with the number and uncertainty of relaxed entries.

## Supported training modes

| Configuration | DE decoder | Optimized parameters |
|---|---|---|
| `sample_spa.json` | Flooding sum-product | Relaxed PCM |
| `sample_nms.json` | Flooding normalized min-sum | Relaxed PCM; fixed column scales |
| `sample_nms_joint.json` | Flooding normalized min-sum | Relaxed PCM and column scales |

Puncturing is supported: the first `punctured` variable-node types are
initialized with a zero-LLR distribution.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The sample configurations currently use CPU. To use CUDA, set, for example,
`"device": "cuda:0"` and install a CUDA-enabled PyTorch build.

## Running experiments

Run all three sample experiments sequentially:

```bash
bash example.sh
```

Run one configuration:

```bash
python3 main.py configs/sample_nms.json
```

The optional second argument selects a saved epoch from which to restart:

```bash
python3 main.py configs/sample_nms.json 100
```

For a restart, `pcm_soft_100.txt` and, for NMS, `scales_100.txt` are loaded
from the configured history directory.

## Configuration

Each JSON file contains the following sections.

### `protograph`

```json
"protograph": {
  "init_pcm": "input/init.txt",
  "punctured": 0
}
```

- `init_pcm`: whitespace-separated relaxed PCM. Every entry must be in
  `[0, 1]`.
- `punctured`: number of punctured columns, starting from the first column.

### `grid`

```json
"grid": {
  "llr_max": 20,
  "llr_scale": 20,
  "device": "cpu",
  "dtype": "float64"
}
```

- `llr_max`: clipping magnitude for quantized LLR values.
- `llr_scale`: number of grid intervals per unit LLR.
- `device`: PyTorch device.
- `dtype`: PMF data type; `float64` is recommended for small BER values.

The grid contains `2 × llr_max × llr_scale + 1` points. Its size has a major
effect on runtime and memory usage, especially for SPA and trainable NMS
scaling interpolation.

### `decoder`

```json
"decoder": {
  "type": "nms",
  "col_scales": "input/scales.txt",
  "train_scales": 0
}
```

- `type`: `"nms"` or `"spa"`.
- `col_scales`: per-column NMS normalization coefficients.
- `train_scales`: `1` to optimize NMS scales jointly with the PCM; `0` to keep
  them fixed. It is ignored by SPA.

### `optimization`

- `snr_db`: initial training SNR in dB.
- `n_iter`: number of flooding DE iterations per optimizer epoch.
- `ber_threshold`: BER at which the current SNR is considered solved.
- `snr_step`: amount subtracted from `snr_db` after reaching the threshold.

Learning-rate scheduling is controlled by gradient-sign changes:

```json
"lr_schedule": {
  "initial_rate": 0.01,
  "acceleration_factor": 1.1,
  "deceleration_factor": 0.75,
  "acceleration_flip_threshold": 1,
  "deceleration_flip_threshold": 3,
  "rollback_flip_threshold": 6
}
```

For `F` changed PCM gradient signs:

```text
F < acceleration_flip_threshold  → LR *= acceleration_factor
F > deceleration_flip_threshold  → LR *= deceleration_factor
F > rollback_flip_threshold      → reject candidate and reset LR
```

Deceleration never reduces LR below `initial_rate`. Rollback is active only
after LR has grown above its initial value.

Convergence settings:

```json
"convergence": {
  "max_epochs": 1000,
  "ensemble_size_patience": 20
}
```

`ensemble_size_patience` counts consecutive accepted epochs during which
`log2(# matrices)` does not improve on its absolute minimum over the training
history.

### Output directory

`data_dir` is interpreted relative to `output/`:

```json
"data_dir": "history_nms"
```

## Generated data

For each accepted epoch `N`, the optimizer writes:

```text
output/<data_dir>/pcm_soft_N.txt   accepted relaxed PCM
output/<data_dir>/pcm_grad_N.txt   PCM gradient
output/<data_dir>/ber_N.txt        DE-predicted BER
output/<data_dir>/scales_N.txt     NMS scales, when applicable
output/<data_dir>.log              complete training log
```

Rejected rollback candidates and SNR-reduction epochs are not added to the
history.

After training stops, the relaxed PCM from the last accepted epoch is rounded
by evaluating every integer matrix in its remaining ensemble at the current
working SNR. The same DE grid, decoder type, puncturing, iteration count, and
(for NMS) column scales from that epoch are used. The best matrix is written
to:

```text
output/<data_dir>/final.txt
```

## Postprocessing

Generate entropy plots, integer-fraction and BER plots, scale trajectories,
gradient statistics, and a training GIF:

```bash
python3 output/postproc.py output/history_nms
```

The GIF is produced directly with Matplotlib/Pillow without creating
intermediate PNG frames.

## Project structure

```text
main.py                         command-line entry point
config.py                       JSON parsing and validation
implementation/
  density_evolution.py          quantized SPA/NMS density evolution
  node_operations.py            differentiable VN/CN PMF operations
optimization/
  optimizer.py                  training-loop orchestration
  parameter_updater.py          detached GD steps and history commits
  lr_scheduler.py               adaptive learning-rate policy
  convergence.py                stopping criteria
configs/                        sample experiments
input/                          initial PCM and NMS scales
output/postproc.py              history visualization
tools/                          standalone analysis utilities
```

## Scope and limitations

- DE uses the standard independence/tree-like-neighborhood assumption.
- The objective is protograph-level BER, not finite-length FER or BLER.
- Short cycles, girth, lifting constraints, and rate-adaptation constraints
  are not part of the current loss.
- The formal relaxed-ensemble interpretation from the paper is established
  for the min-sum family. SPA support in this code should be treated as an
  experimental differentiable relaxation.
- A final relaxed PCM may contain non-integer entries. These can be interpreted
  probabilistically during lifting or enumerated to select a binary
  protograph, as discussed in the paper.
