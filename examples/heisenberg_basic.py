"""Example: notebook-based Heisenberg PQFM."""

from pqfmlib import HeisenbergProjectiveQFM


if __name__ == "__main__":
    qfm = HeisenbergProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=False,
        simulation=False,
        fakebackend=False,
        shots=4096,
        ibm_qpu="ibm_kingston",
        q_enc=150,
        R=2,
        alpha=0.1,
        use_tanh_scaling=True,
        save_circuit_drawings=False,
    )
    print(qfm.run())
