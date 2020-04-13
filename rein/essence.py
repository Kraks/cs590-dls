from typing import Dict, NamedTuple, List, Tuple, NewType, Optional, Set, Union, Literal
from dataclasses import dataclass, field
from itertools import groupby
import sys
import glob

# I will explain the algorithm DPLL and present a functional, object-oriented implementation
# of DPLL in state-transtion style, all written in Python with type annotations.

Lit = NewType('Lit', int)
Asn = Tuple[Lit, ...]

class Clause:
    def __init__(self, xs: List[Lit]):
        self.xs = xs

    def __len__(self): return len(self.xs)

    def __getitem__(self, i): return self.xs[i]

    def contains(self, x: Lit) -> bool: return x in self.xs

    def remove(self, x: Lit) -> 'Clause': 
        return Clause([y for y in self.xs if x != y])

    def assign(self, y: Lit, v: bool) -> Union[Literal[True], 'Clause']:
        xs = []
        for x in self.xs:
            if abs(x) == abs(y):
                if (x > 0) == v: return True
            else: xs.append(x)
        return Clause(xs)

class Formula:
    def __init__(self, cs: List[Clause]):
        self.cs = cs
        self.compute_unit_vars()
        self.compute_pure_vars()

    def compute_unit_vars(self):
        self.unit_vars = [c[0] for c in self.cs if len(c) == 1]

    def compute_pure_vars(self):
        distinct_vars = list(set([it for sub in [c.xs for c in self.cs] for it in sub]))
        # first sort all variables, where the key equality ignores the signs
        # then group those variables also according to their signs
        # Note that the groupby operation can only group consecutive elements
        grouped_vars = [list(g) for k, g in groupby(sorted(distinct_vars, key=abs), abs)]
        self.pure_vars = [g[0] for g in grouped_vars if len(g) == 1]

    def pick(self): return self.cs[0][0]

    def assign(self, y: Lit, v: bool) -> 'Formula':
        assigned_cs = [c.assign(y, v) for c in self.cs]
        return Formula([c for c in assigned_cs if c != True])

    def elim_unit(self) -> Tuple['Formula', Asn]:
        x = self.unit_vars[0]
        cs = [c.remove(-x) for c in self.cs if not c.contains(x)]
        return (Formula(cs), (x,))

    def is_empty(self) -> bool: return len(self.cs) == 0

    def has_unit(self) -> bool: return len(self.unit_vars) != 0

    def has_pure(self) -> bool: return len(self.pure_vars) != 0

    def add_unit_clause(self, x: Lit) -> 'Formula':
        return Formula([Clause([x])] + self.cs)

    def has_unsat_clause(self) -> bool:
        for c in self.cs:
            if len(c) == 0: return True
        return False


"""
@dataclass
class Cont:
    var: Lit
    formula: Formula
    assignment: Asn

@dataclass
class State:
    formula: Formula
    assignment: Asn
    cont: Tuple[Cont, ...]
"""

Cont = NamedTuple('Cont', [('var', Lit),
                           ('formula', Formula),
                           ('assignment', Asn)])

State = NamedTuple('State', [('formula', Formula),
                             ('assignment', Asn),
                             ('cont', Tuple[Cont, ...])])

def apply_backtrack(s: State) -> State:
    v, f, asn = s.cont[0]
    return State(f.assign(v, False), (-v,)+asn, s.cont[1:])

def apply_unit(s: State) -> State:
    f, asn, cont = s
    new_f, new_asn = f.elim_unit()
    return State(new_f, new_asn + asn, cont)

def apply_pure(s: State) -> State:
    f, asn, cont = s
    return State(f.add_unit_clause(f.pure_vars[0]), asn, cont)

def dpll_step(s: State) -> State:
    f, asn, k = s
    if f.has_unsat_clause(): return apply_backtrack(s)
    if f.has_unit(): return apply_unit(s)
    if f.has_pure(): return apply_pure(s)
    x = f.pick()
    return State(f.assign(x, True), (x,)+asn, (Cont(x, f, asn),)+k)

def drive(s: State) -> Optional[Asn]:
    f, asn, k = s
    if f.is_empty(): return asn
    if len(k) == 0 and f.has_unsat_clause(): return None
    return drive(dpll_step(s))

def inject(f: Formula) -> State:
    return State(f, (), ())

def solve(f: Formula) -> State:
    return drive(inject(f))

def parse_line(line) -> Clause: return Clause([int(x) for x in line.split(" ")][:-1])

def parse_dimacs(filename: str) -> Formula:
    def valid(l): return not (len(l)==0 or l[0]=='c' or l[0]=='p' or l[0]=='0' or l[0]=='%')
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [parse_line(l) for l in lines if valid(l)]
        return Formula(lines)

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    
    sats = glob.glob('/home/kraks/research/sat/src/main/resources/uf50-218/*.cnf')
    for filename in sats:
        print(filename)
        formula = parse_dimacs(filename)
        #assert(solve(formula) != None)
    unsats = glob.glob('/home/kraks/research/sat/src/main/resources/uuf50-218/*.cnf')
    for filename in unsats:
        print(filename)
        formula = parse_dimacs(filename)
        assert(solve(formula) == None) 
