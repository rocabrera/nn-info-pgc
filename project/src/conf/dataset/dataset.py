from dataclasses import dataclass

@dataclass(frozen=True)
class Data():
    sample_size: int
    binary_output: bool
