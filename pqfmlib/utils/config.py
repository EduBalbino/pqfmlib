"""Configuration helpers.

This module intentionally does not hard-code any private IBM Quantum token.
Use the QISKIT_IBM_TOKEN environment variable instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration read from environment variables."""

    ibm_token_env: str = "QISKIT_IBM_TOKEN"

    @property
    def QXToken(self) -> str:
        token = os.getenv(self.ibm_token_env, "")
        if not token:
            raise RuntimeError(
                f"IBM Quantum token not found. Set the {self.ibm_token_env} environment variable."
            )
        return token
