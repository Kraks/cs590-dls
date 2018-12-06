from typing import Dict, NamedTuple, List, Tuple, NewType, Optional, Set
from itertools import groupby
from common import *

def apply_backtrack(s: State) -> Set[State]:
    #print("backtrack {}".format(s))
    v, f, asn = s.cont[0]
    return set(State(f.assign(v, False), (-v,)+asn, s.cont[1:]))

def apply_units(s: State) -> Set[State]:
    #print("unit {}".format(s))
    f, asn, cont = s
    states = []
    for v in f.unitVars:
        new_f, new_asn = f.elimUnit(v)
        states.append(State(new_f, new_asn + asn, cont))
    return set(states)

def dpll_steps(s: State) -> Set[State]:
    f, asn, cont = s
    if f.hasUnsat():  return apply_backtrack(s)
    elif f.hasUnit(): return apply_units(s)
    else:
        states = []
        for v in f.allVars:
            states.append(State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont))
            states.append(State(f.assign(v. False), (-v,)+asn, (Cont(-v, f, asn),)+cont))
        return set(states)

def is_done(s: State) -> Tuple[bool, Optional[Asn]]:
    f, asn, cont = s
    if f.isEmpty(): return (True, asn)
    elif len(cont) == 0 and f.hasUnsat(): return (True, None)
    else: return (False, None)
