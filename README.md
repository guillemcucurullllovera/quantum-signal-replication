# Quantum Signal Paper Replication: Projected Kernels

This release retains the first-party implementation of projected quantum
RBF kernels and OCSVM evaluation from the Pranjic et al. anomaly-detection
replication (arXiv:2411.16970). The paper describes a method; no upstream code
was copied according to the source notices. The 10-qubit CX/ECR/RXX feature maps
and Frobenius-distance kernel are real circuit/statevector implementations:
K(x,y)=exp(-gamma*sum_q ||rho_q(x)-rho_q(y)||_F^2).

The source's RDM helper needed two semantic corrections: Qiskit qubit-to-tensor
axis order and complex density-matrix orientation. Independent Qiskit partial
traces now validate each physical RDM. These transformations leave the summed
Frobenius kernel invariant, but physical per-qubit RDMs must still be correct.
The release CV wrapper also fits angle scaling on each training fold only;
global pre-scaling or a globally encoded feature tensor is not accepted as
evidence of leakage-safe evaluation. Hyperparameters and feature map are fixed
before testing; no model is selected on test scores.

The original literature-replication attempt did not validate the paper's reported
advantage. This release does not reuse that dataset's metrics as new evidence.
Synthetic controls validate code and expose kernel concentration; they do not
replicate a paper table or establish a finance/trading signal.

## Reproduce

```sh
uv sync --locked --extra dev --python 3.12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync pytest -q
uv run --no-sync python experiments/synthetic.py
```

## Limitations

Simulator-limit exact statevectors only, not tomography or hardware. No Sparkov dataset, historical result archive, vendor code or credentials is distributed. Shuffled stratified CV is for IID anomaly experiments, not financial time-series forecasting. QRC paper replication and trading-signal validation are outside this scoped release.

See THEORY.md, PROVENANCE.md and SOURCE_FILES.json.
