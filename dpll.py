from typing import Dict, NamedTuple, List, Tuple, NewType, Optional
from itertools import groupby

Lit = NewType('Lit', int)
Asn = Tuple[Lit, ...]

class Clause():
    def __init__(self, xs: List[Lit]):
        self.xs = xs
    def __len__(self): return len(self.xs)
    def __str__(self): return str(self.xs)
    __repr__ = __str__
    def __getitem__(self, i): return self.xs[i]
    def contains(self, x): return x in self.xs
    def remove(self, x) -> 'Clause': return Clause([y for y in self.xs if x != y])
    def assign(self, v: Lit, b: bool) -> Optional['Clause']:
        new_xs = []
        for x in self.xs:
            if abs(x) == abs(v):
                if (x > 0) == b: return None
            else: new_xs.append(x)
        return Clause(new_xs)

class Formula():
    def __init__(self, cs: List[Clause]):
        self.cs = cs
        self.allVars = list(set([item for sub in [c.xs for c in cs] for item in sub]))
        # TODO: pure variables lost original information
        self.pureVars = [k for k, g in groupby(self.allVars, abs) if len(list(g))==1]
        self.unitVars = [c[0] for c in cs if len(c) == 1]
    def __str__(self): return str(self.cs)
    __repr__ = __str__
    def pick(self): return self.cs[0][0]
    def isEmpty(self) -> bool: return len(self.cs) == 0
    def assign(self, v: Lit, b: bool) -> 'Formula':
        assigned = [c.assign(v, b) for c in self.cs]
        assigned = [c for c in assigned if c != None]
        return Formula(assigned)
    def elimUnit(self) -> Tuple['Formula', Asn]:
        v = self.unitVars[0]
        #print("elim unit {}".format(v))
        return (Formula([c.remove(-v) for c in self.cs if not c.contains(v)]), (v,))
    def addUnit(self, v: Lit) -> 'Formula':
        return Formula([Clause([v])] + self.cs.copy())
    def hasUnsat(self) -> bool:
        for c in self.cs: 
            if len(c) == 0: return True
        return False
    def hasUnit(self) -> bool: return len(self.unitVars) != 0
    def hasPure(self) -> bool: return len(self.pureVars) != 0

class Cont(NamedTuple):
    var: Lit
    formula: Formula
    assignment: Asn

class State(NamedTuple):
    formula: Formula
    assignment: Asn
    cont: Tuple[Cont, ...]

def apply_backtrack(s: State) -> State:
    #print("backtrack {}".format(s))
    top = s.cont[0]
    v, f, asn = top
    return State(f.assign((v, False)), (-v,)+asn, s.cont[1:])

def apply_unit(s: State) -> State:
    #print("unit {}".format(s))
    f, asn, cont = s
    new_f, new_asn = f.elimUnit()
    return State(new_f, new_asn + asn, cont)

def apply_pure(s: State) -> State:
    #print("pure {}".format(s))
    f, asn, cont = s
    return State(f.addUnit(f.pureVars[0]), asn, cont)

def dpll_step(s: State) -> State:
    f, asn, cont = s
    if f.hasUnsat():  return apply_backtrack(s)
    elif f.hasUnit(): return apply_unit(s)
    elif f.hasPure(): return apply_pure(s)
    else:
        v = f.pick()
        return State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont)

def drive(s: State) -> Optional[Asn]:
    print(s)
    f, asn, cont = s
    if f.isEmpty(): return asn
    elif len(cont) == 0 and f.hasUnsat(): return None
    else: return drive(dpll_step(s))

def inject(f: Formula) -> State: return State(f, (), ())

def solve(s: Formula) -> State: return drive(inject(f))

if __name__ == '__main__':
    f = Formula([Clause([1, 2]),
                 Clause([-1])])
    assert(solve(f) == (2, -1))
    f = Formula([Clause([1,2,3]),
                 Clause([-1, -2, -3])])
    assert(solve(f) != None)
    f = Formula([Clause([1,2,3]),
                 Clause([-1]),
                 Clause([-2]),
                 Clause([-3])])
    assert(solve(f) == None)
