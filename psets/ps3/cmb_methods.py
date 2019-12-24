"""
Methods to model CMB in the context of this assignment
"""
import camb


def get_spectrum(p, lmax_acc=2000, lmax_ret=1200):
    """Get power spectrum with CAMB
    Args:
        p: parameters H0, ombh2, omch2, tau, As, ns
        lmax_acc: maximum l index for the accuracy of calculations
        lmax_ret: maximum l index for the returned power spectrum
    Returns:
        tt: TT power spectrum for multipole indices 2 (quadrupole) to lmax_ret
    """
    H0 = p[0]
    ombh2 = p[1]
    omch2 = p[2]
    tau = p[3]
    As = p[4]
    ns = p[5]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=0.06,
                       tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(lmax_acc, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_ret)
    cmb = powers['total']
    tt = cmb[:, 0][2:]  # don't need first to indices

    return tt
