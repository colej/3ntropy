import numpy as np

def sort_on_x(x,y,z=None):
    if z is not None:
        zipp = list(zip(x,y,z)).sort(key=lambda x:x[0])
        x,y,z = list(zip(*zipp))
    else:
        zipp = list(zip(x,y)).sort(key=lambda x:x[0])
        x,y = list(zip(*zipp))
        z = None
    yield x
    yield y
    yield z


def run_aliasing(x,y,z=None):
    ax,ay,az = [],[],[]
    for ii in range(len(x)):
        if x[ii] < -0.4:
                ax.append(x[ii]+1)
                ay.append(y[ii])
                if z is not None:
                    az.append(z[ii])
        if x[ii] > 0.4:
                ax.append(x[ii]-1)
                ay.append(y[ii])
                if z is not None:
                    az.append(z[ii])

    if z is not None:
        x,y,z = list(sort_on_x(np.append(x,ax),np.append(y,ay),np.append(z,az)))
    else:
        x,y,z = list(sort_on_x(np.append(x,ax),np.append(y,ay),None))

    yield x
    yield y
    yield z

def time_to_ph(time, period=None, t0=0.):
    '''
    converts time to phase from input ephemeris
    DOES NOT ACCOUNT FOR BARYCENTRIC OR HELIOCENTRIC CORRECTION

    hjd_to_ph(time, period, t0)

    input: time (float or array)
    input: period (float)
    input: t0 (float)
    output: phase (float or array)
    '''

    ph = np.array( [ -0.5+( ( t-t0-0.5*period ) % period ) / period for t in time ] )
    return ph


def ph_to_time (phase, period, t0, norb):
    '''
    converts phase to time from an input ephemeris and number of orbit norb
    time = ph_to_time(phase, period, t0, norb)
    input: phase (float or array)
    input: period (float)
    input: t0 (float)
    input: n (int)
    output: time (float or array)
    '''
    n = int(n)

    time = np.array( [ ph*period + t0 + norb*period for ph in phase ] )
    #~ (phase+0.5)*period+period/2.0+hjd0+n*period
    return time


def calc_sup_conj_phase(omega,eps):
    '''
    routine to calculate the phase of superior conjuction
    and phase of periastron

    '''

    ecc_anomoly    = 2.*np.arctan( np.tan( 0.25*np.pi - 0.5*omega )*((1.+eps)/(1.-eps))**-0.5 )
    mean_anomoly   = ecc_anomoly-eps*np.sin(ecc_anomoly)

    phase_sup_conj = 0.5*(mean_anomoly+omega)/np.pi - 0.25
    phase_periast  = phase_sup_conj - 0.5*mean_anomoly/np.pi

    return phase_sup_conj,phase_periast
