"""Example: CD-Ising PQFM."""

from pqfmlib import CDIsingProjectiveQFM


if __name__ == "__main__":
    qfm = CDIsingProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=False,
        simulation=False,
        fakebackend=False,
        shots=4096,
        ibm_qpu="ibm_kingston",
        q_enc=156,
        measure_all_zz=False,
        mps=True,
    )
    print(qfm.run())
