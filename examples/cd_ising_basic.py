"""Example: CD-Ising PQFM on the Toxicity dataset."""

from pqfmlib import CDIsingProjectiveQFM


if __name__ == "__main__":
    qfm = CDIsingProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        data_dir="./data",
        output_root="./results",
        seed=42,
        ideal=True,
        shots=4096,
        q_enc=13,
        m=1,
        tau=0.005,
        k_max=2,
        measure_all_zz=False,
    )
    print(qfm.run())
