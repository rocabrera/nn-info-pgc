from dataclasses import dataclass

@dataclass(frozen=True)
class Architecture():
    epochs: int
    learning_rate: float
    hidden_layer_sizes: list
    binary_classification: bool