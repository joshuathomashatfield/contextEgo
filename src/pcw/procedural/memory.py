"""Bounded predicted history, separate from procedural knowledge."""
from collections import deque
import numpy as np

class Memory:
    def __init__(self, classes, mode="none", size=16):
        self.k=classes; self.mode=mode; self.history=deque(maxlen=size)
    @property
    def dimension(self):
        return {"none":0,"previous":self.k,"history":2*self.k+self.k*self.k}[self.mode]
    def reset(self): self.history.clear()
    def update(self, prediction):
        if not 0 <= prediction < self.k: raise ValueError("Unknown predicted step")
        self.history.append(int(prediction))
    def vector(self):
        if self.mode=="none": return np.zeros(0,dtype=np.float32)
        prev=np.zeros(self.k,dtype=np.float32)
        if self.history: prev[self.history[-1]]=1
        if self.mode=="previous": return prev
        visits=np.zeros(self.k,dtype=np.float32); counts=np.zeros((self.k,self.k),dtype=np.float32)
        h=list(self.history)
        for s in h: visits[s]+=1/max(1,len(h))
        for a,b in zip(h,h[1:]): counts[a,b]+=1/max(1,len(h)-1)
        return np.concatenate([prev,visits,counts.ravel()])

class Decoder:
    def __init__(self, mode="none", size=3):
        self.mode=mode; self.size=size; self.values=deque(maxlen=size); self.current=None
    def update(self,p):
        self.values.append(int(p))
        if self.mode=="none": self.current=p
        elif self.mode=="majority":
            counts={v:list(self.values).count(v) for v in self.values}
            self.current=max(reversed(self.values),key=lambda v:counts[v])
        elif self.current is None or (len(self.values)==self.size and len(set(self.values))==1): self.current=p
        return int(self.current)
