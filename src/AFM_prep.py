# https://docs.pasqal.com/pulser/tutorials/optimization/
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip
from pulser import Pulse, Register, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipBackendV2
from skopt import gp_minimize

T = 1000  # in µs
device = AnalogDevice
R_interatomic = 5  # µm


def create_interp_pulse(amp_params, det_params):
    return Pulse(
        InterpolatedWaveform(T, [1e-9, *amp_params, 1e-9]),
        InterpolatedWaveform(T, [delta_0, *det_params, delta_f]),
        0,
    )


def occupation(j, N):
    up = qutip.basis(2, 0, dtype="CSR")
    prod = [qutip.qeye(2, dtype="CSR") for _ in range(N)]
    prod[j] = up * up.dag()
    return qutip.tensor(prod)


def get_corr_pairs(k, N):
    corr_pairs = [[i, (i + k) % N] for i in range(N)]
    return corr_pairs


def get_corr_function(k, N, state):
    corr_pairs = get_corr_pairs(k, N)
    operators = [occupation(j, N) for j in range(N)]
    covariance = 0
    for qi, qj in corr_pairs:
        covariance += qutip.expect(operators[qi] * operators[qj], state)
        covariance -= qutip.expect(operators[qi], state) * qutip.expect(
            operators[qj], state
        )
    return covariance / len(corr_pairs)


def get_neel_structure_factor(reg, state):
    N = len(reg.qubits)
    st_fac = 0
    for k in range(-N // 2, N // 2 + 1):
        kk = np.abs(k)
        st_fac += 4 * (-1) ** kk * get_corr_function(k, N, state)
    return st_fac


def score(params):
    """Computes the score associated with a set of parameters."""
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(create_interp_pulse(params[:m], params[m:]), "ising")

    simul = QutipBackendV2(seq)
    results = simul.run()

    sampling = results.final_state.sample(num_shots=N_samples)
    sampled_state = sum(
        [
            np.sqrt(sampling[k] / N_samples) * qutip.ket(k, dtype="CSR")
            for k in sampling.keys()
        ]
    )

    F = get_neel_structure_factor(reg, sampled_state) / S_max

    return 1 - F


N_side = 4
# generates coords, square without middle, N_side per side
coords = (
    [R_interatomic * np.r_[x, 0] for x in range(N_side - 1)]
    + [R_interatomic * np.r_[N_side - 1, y] for y in range(N_side - 1)]
    + [R_interatomic * np.r_[N_side - 1 - x, N_side - 1] for x in range(N_side - 1)]
    + [R_interatomic * np.r_[0, N_side - 1 - y] for y in range(N_side - 1)]
)

reg = Register.from_coordinates(coords, prefix="q")
N = len(coords)
reg.draw()

# this will check if the register is compatible with device, otherwise an exception will occur when ran
seq = Sequence(reg, device)

# declare the channel
seq.declare_channel("ising", "rydberg_global")

# adding the pulse
# Size of the parameter space
m = 3

# Amp and detuning must respect the channel's specifications
tol = 1e-6  # within a certain tolerance

max_amp = seq.declared_channels["ising"].max_amp * (1 - tol)
Omega_max = min(max_amp, device.rabi_from_blockade(1.2 * R_interatomic))

U = 2 * Omega_max
max_det = seq.declared_channels["ising"].max_abs_detuning * (1 - tol)
delta_0 = -min(max_det, U)
delta_f = min(max_det, U)

# Random instance of the parameter space
amp_params = np.random.uniform(0, Omega_max, m)
det_params = np.random.uniform(delta_0, delta_f, m)

seq.add(create_interp_pulse(amp_params, det_params), "ising")
seq.draw()

N_samples = 1000
simul = QutipBackendV2(seq)
results = simul.run()
final_state = results.final_state
counts = final_state.sample(num_shots=N_samples)
print(counts)
# Let's plot the histogram associated to the measurements
# Let's select only the states that are measured more than 10 times
most_freq = {k: v for k, v in counts.items() if v > 10}
plt.bar(list(most_freq.keys()), list(most_freq.values()))
plt.xticks(rotation="vertical")
plt.show()

# Create antiferromagnetic state as the superposition of the two checkerboard patterns:
AFM1 = qutip.tensor([qutip.basis(2, k % 2) for k in range(N)])
AFM2 = qutip.tensor([qutip.basis(2, (k + 1) % 2) for k in range(N)])
AFM_state = (AFM1 + AFM2).unit()

t1 = time.process_time()
S_max = get_neel_structure_factor(reg, AFM_state)
print("S_Neel(AFM state) =", S_max)
t2 = time.process_time()
print("computed in", (t2 - t1), "sec")

print(score(np.r_[amp_params, det_params]))
