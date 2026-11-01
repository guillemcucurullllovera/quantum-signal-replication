import numpy as np
import pytest
from qiskit.quantum_info import Statevector,partial_trace
from quantum_signals.kernels import FEATURE_MAPS,state_to_1q_rdms,N_QUBITS
from quantum_signals.replication import fit_angle_scaler,run_quantum_cv


@pytest.mark.parametrize('name',['CX','ECR','RXX'])
def test_each_qubit_rdm_matches_independent_qiskit_partial_trace(name):
    x=np.random.default_rng(2026).uniform(-np.pi,np.pi,20)
    state=Statevector(FEATURE_MAPS[name](x));actual=state_to_1q_rdms(state.data)
    for qubit in range(N_QUBITS):
        oracle=partial_trace(state,[i for i in range(N_QUBITS) if i!=qubit]).data
        assert np.allclose(actual[qubit],oracle,atol=1e-12)


def test_training_scaler_is_independent_of_held_out_extrema():
    train=np.random.default_rng(4).normal(size=(12,20))
    scaler=fit_angle_scaler(train);before=scaler.transform(train)
    scaler.transform(np.full((3,20),1e9))
    assert np.array_equal(before,scaler.transform(train))
    assert np.array_equal(scaler.data_min_,train.min(axis=0))


def test_fixed_hyperparameters_reproduce_fold_scores():
    X=np.random.default_rng(7).normal(size=(12,20));y=np.array([0]*8+[1]*4)
    a=run_quantum_cv(X,y,'CX',k=2,seed=41)
    b=run_quantum_cv(X,y,'CX',k=2,seed=41)
    assert a==b
    assert a['scaling']=='training fold only'
