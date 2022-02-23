from dataclasses import dataclass

@dataclass(frozen=True)
class Data():
    sample_size_pct: float
    binary_output: bool
