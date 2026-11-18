"""Directed procedural knowledge; unexpected execution is observable, not blocked."""
from pcw.datasets.schema import validate_procedure

class ProcedureGraph:
    def __init__(self,procedure):
        self.procedure=validate_procedure(procedure)
        self.edges=None if "transitions" not in procedure else {tuple(edge) for edge in procedure["transitions"]}
    def expected(self,previous,current):
        k=len(self.procedure["steps"])
        if not 0<=previous<k or not 0<=current<k: raise ValueError("Unknown graph step")
        return self.edges is None or (previous,current) in self.edges
