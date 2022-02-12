from dataclasses import dataclass

@dataclass(frozen=True)
class ScatterplotAesthetics:
    alpha:float
    edgecolor:str
    palette:str
    linewidth:float

@dataclass(frozen=True)
class FiguresAesthetics:
    scatterplot_aesthetics: ScatterplotAesthetics