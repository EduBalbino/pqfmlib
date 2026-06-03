"""Example: XYZ PQFM with shared-feature encoding on Toxicity."""

from pqfmlib import XYZProjectiveQFM


if __name__ == "__main__":
    qfm = XYZProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=False,
        simulation=False,
        fakebackend=False,
        shots=4096,
        ibm_qpu="ibm_kingston",

        q_enc=156,
        features_per_qubit=1,
        axes=("x", "y", "z"),
        encoding_mode="shared_feature",

        keep_cross_terms=False,
        keep_diagonal_terms=True,
    )
    print(qfm.run())