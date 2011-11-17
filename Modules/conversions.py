h = 4.13566733e-15 # eV s
c = 299792458.0 # m / s

def ev_to_m(ev):
    return h*c/ev
    
def m_to_ev(m):
    return h*c/nm
    
def ev_to_nm(ev):
    ev_to_m*1.0e9

def nm_to_ev(nm):
    return m_to_ev(nm*1.e-9)

