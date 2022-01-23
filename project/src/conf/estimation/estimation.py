from dataclasses import dataclass

@dataclass(frozen=True)
class Discrete():
    bins:int

@dataclass(frozen=True)
class Continuos():
    kernel:str

@dataclass(frozen=True)
class Estimation():
    discrete:Discrete
    continuos:Continuos