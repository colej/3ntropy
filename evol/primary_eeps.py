import numpy as np
from scipy.interpolate import PchipInterpolator
import h5py

from entropy3.mesa import hdf5_io_support as h5io

def locate_primary_eeps(track,keys):
    pms_to_zams_eep = list(locate_pms_to_zams_eep(track,keys))[0]
    zams_to_mams_eep = list(locate_zams_to_mams_eep(track,keys))[0]
    mams_to_tams_eep = list(locate_mams_to_tams_eep(track,keys))[0]
    tams_to_rgbtip_eep = list(locate_tams_to_rgbtip_eep(track,keys))[0]

    yield pms_to_zams_eep
    yield zams_to_mams_eep
    yield mams_to_tams_eep
    yield tams_to_rgbtip_eep


def locate_pms_to_zams_eep(track,keys):

    idx_pms = np.where( track['log_center_T']>=5.0 )[0]
    idx = np.where( track['log_Lnuc'][idx_pms]/track['log_L'][idx_pms] )

    pms_to_zams_eep = {}

    for key in keys:
        pms_to_zams_eep[key] = track[key][idx]

    yield pms_to_zams_eep


def locate_zams_to_mams_eep(track,keys):

    # idx = np.where( ((track['center_h1']>=0.3) &
    #                  (track['log_Lnuc']/track['log_L'] > 0.9999)) )[0]
    idx = np.where( (track['center_h1']>=0.3) )[0]

    zams_to_mams_eep = {}

    for key in keys:
        zams_to_mams_eep[key] = track[key][idx]

    yield zams_to_mams_eep


def locate_mams_to_ntams_eep(track,keys):

    # idx = np.where( ((track['center_h1']<0.3) &
    #                  (track['center_h1']>0.01) &
    #                  (track['log_Lnuc']/track['log_L'] > 0.9999)) )[0]
    idx = np.where( ((track['center_h1']<0.3) &
                     (track['center_h1']>0.01)) )[0]

    mams_to_ntams_eep = {}

    for key in keys:
        mams_to_ntams_eep[key] = track[key][idx]

    yield mams_to_ntams_eep


def locate_ntams_to_tams_eep(track,keys):

    # idx = np.where( ((track['center_h1']<0.01) &
    #                  (track['center_h1']>1e-12) &
    #                  (track['log_Lnuc']/track['log_L'] > 0.9999)) )[0]
    idx = np.where( ((track['center_h1']<0.01) &
                     (track['center_h1']>1e-12)) )[0]

    ntams_to_tams_eep = {}

    for key in keys:
        ntams_to_tams_eep[key] = track[key][idx]

    yield ntams_to_tams_eep


def locate_tams_to_rgbtip_eep(track,keys):

    idx = np.where( ((track['center_h1']<1e-12) &
                     (track['center_h1']>1e-12) &
                     (track['log_Lnuc']/track['log_L'] > 0.9999)) )[0]

    tams_to_rgbtip_eep = {}

    for key in keys:
        tams_to_rgbtip_eep[key] = track[key][idx]

    yield tams_to_rgbtip_eep
