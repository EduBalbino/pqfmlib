"""Example: Heisenberg PQFM on the Toxicity dataset."""

from pqfmlib import HeisenbergProjectiveQFM


if __name__ == "__main__":
    qfm = HeisenbergProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        data_dir="./data",
        output_root="./results",
        seed=42,
        ideal=True,
        shots=4096,
        q_enc=14,
        R=2,
        alpha=0.1,
        use_tanh_scaling=True,
        measure_2local_diagonal=False,
        save_circuit_drawings=False,
    )
    print(qfm.run())
