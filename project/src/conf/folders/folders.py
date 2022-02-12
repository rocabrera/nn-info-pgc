from dataclasses import dataclass



@dataclass(frozen=True)
class Results():
    discrete:str
    continuos:str

@dataclass(frozen=True)
class Figures():
    discrete:str
    continuos:str
@dataclass(frozen=True)
class Gifs():
    discrete:str
    continuos:str
@dataclass(frozen=True)
class Folders():
    data:str
    results:Results
    figures:Figures
    gifs:Gifs
