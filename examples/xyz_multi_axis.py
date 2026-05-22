"""Example: XYZ PQFM with multi-axis encoding on the Toxicity dataset."""

from pqfmlib import XYZProjectiveQFM


if __name__ == "__main__":
    qfm = XYZProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        data_dir="./data",
        output_root="./results",
        seed=42,
        ideal=True,
        shots=4096,
        q_enc=52,
        features_per_qubit=3,
        axes=("x", "y", "z"),
        encoding_mode="multi_axis",
        keep_diagonal_terms=True,
        keep_cross_terms=True,
        measure_cross_observables=False,
        m=1,
        tau=1.0,
    )
    print(qfm.run())
