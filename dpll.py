from typing import Dict, NamedTuple, List, Tuple, NewType, Optional, Set
from itertools import groupby
from common import *

def apply_backtrack(s: State) -> State:
    #print("backtrack {}".format(s))
    v, f, asn = s.cont[0]
    return State(f.assign(v, False), (-v,)+asn, s.cont[1:])

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
    #elif f.hasPure(): return apply_pure(s)
    else:
        v = f.pick()
        return State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont)

def drive(s: State) -> Optional[Asn]:
    #print(s)
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
