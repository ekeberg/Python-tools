"""Conversions between commonly used units."""
#import constants as _constants
from . import constants as _constants

def ev_to_m(electronvolt):
    """Photon energy in electronvolt to wavelength in meter."""
    return _constants.h_evs*_constants.c/electronvolt

def m_to_ev(meter):
    """Photon wavelength in meter to energy in electronvolt."""
    return _constants.h_evs*_constants.c/meter

def ev_to_nm(electronvolt):
    """Photon energy in electronvolt to wavelength in nanometer."""
    return ev_to_m(electronvolt)*1.0e9

def nm_to_ev(nanometer):
    """Photon wavelength in nanometer to energy en electronvolt."""
    return m_to_ev(nanometer*1.e-9)

def J_to_ev(joule):
    """Joule to electronvolt."""
    return joule/_constants.e

def ev_to_J(electronvolt):
    """Electronvolt to Joule."""
    return electronvolt*_constants.e
