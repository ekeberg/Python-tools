import constants as _constants

def ev_to_m(ev):
    return _constants.h_evs*_constants.c/ev
    
def m_to_ev(m):
    return _constants.h_evs*_constants.c/m
    
def ev_to_nm(ev):
    return ev_to_m(ev)*1.0e9

def nm_to_ev(nm):
    return m_to_ev(nm*1.e-9)

def J_to_ev(J):
    return J/_constants.e

def ev_to_J(ev):
    return ev*_constants.e

