import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator,interp1d,Akima1DInterpolator
from progress.bar import Bar

from entropy3.general.smoothing import smooth

def sort_tracks_on_mass(tracks):

    masses = [track['star_mass'][0] for track in tracks]
    zipp = list(zip(masses,tracks))
    zipp.sort(key=lambda x:x[0])
    masses,tracks = list(zip(*zipp))
    yield tracks


def get_keys_dict(keys):
    key_dict = {key: [] for key in keys}
    yield key_dict

def make_isochrones_dict(i_ages,keys):

    key_dict = list(get_keys_dict(keys))[0]
    isochrones = { 'age-{}'.format(cc): key_dict for cc,i_age in enumerate(i_ages) }

    yield isochrones


def get_subsample(n,tracks,keys):

    # eep_subsample = { key: [ track[key][n] for track in tracks  ] for key in keys }
    eep_subsample = list(get_keys_dict(keys))[0]
    for track in tracks:
        for key in keys:
            eep_subsample[key].append(track[key][n])
    eep_masses = eep_subsample['star_mass']
    eep_ages = eep_subsample['star_age']

    yield eep_subsample
    yield eep_masses
    yield eep_ages


def make_interpolation_function(x, y):
    # This function creates an interpolation object to be called later
    # i_func = PchipInterpolator(x, y, extrapolate=False)
    # i_func = interp1d(x,y,kind='cubic')
    i_func = Akima1DInterpolator(x,y)

    yield i_func


def sort_y_on_x(x,y):

    zipp = list(zip(x,y))
    zipp.sort(key=lambda x:x[0])
    x,y = list(zip(*zipp))
    yield x
    yield y


def loop_over_subsample(i_age,cc,subsample,masses,ages,keys,isochrones):

    sorted_ages,sorted_masses = list(sort_y_on_x(ages,masses))
    i_age_func = list(make_interpolation_function(sorted_ages,sorted_masses))[0]
    mass_coord = i_age_func(i_age)
    isochrones = list(populate_isochrones(cc,mass_coord,masses,subsample,keys,isochrones))[0]

    plt.plot(sorted_masses,sorted_ages,'k--')
    plt.axhline(i_age,color='blue')
    # plt.plot(isochrones['age-{}'.format(cc)]['log_Teff'],isochrones['age-{}'.format(cc)]['log_g'])
    # plt.xlim(plt.xlim()[::-1])
    # plt.ylim(plt.ylim()[::-1])
    # plt.show()

    yield isochrones


def populate_isochrones(cc,mass_coord,masses,subsample,keys,isochrones):

    for key in keys:
        cval = subsample[key]
        sorted_masses,sorted_cval = list(sort_y_on_x(masses,cval))
        i_cval_func = list(make_interpolation_function(sorted_masses,sorted_cval))[0]
        # isochrones['age-{}'.format(cc)][key].append(i_cval_func(mass_coord))
        isochrones['age-{}'.format(cc)][key].append(i_cval_func(mass_coord))

    yield isochrones

def get_tracks_per_eep(tracks,npoints,keys):

    tracks_per_eep = get_keys_dict(keys)
    # for n_eep in range(npoints):]):



def construct_isochrones(eep_tracks,i_ages,keys,savename,pars):

    #-> Sort tracks based on mass to ensure ease later
    tracks = list(sort_tracks_on_mass(eep_tracks))[0]
    ntracks = len(tracks)
    npoints = len( tracks[0]['star_age'] )
    nages = len(i_ages)

    #-> create a dictionary where each age is a dictionary for all quantities listed above
    isochrones = list(make_isochrones_dict(i_ages,keys))[0]
    isochrones = {'age-{0:04d}'.format(ii): {key: np.empty(npoints) for key in keys} for ii in range(len(i_ages)) }
    eep_tracks = {'eep-{}'.format(n):{key: np.empty(ntracks) for key in keys} for n in range(npoints)}
    eep_mass_age_relations = {'eep-{}'.format(n):{'mass-init': np.empty(ntracks), 'age':np.empty(ntracks)} for n in range(npoints)}


    print('--> Building tracks in all quantities per EEP')
    print('--> Building M_init - Age relation per EEP')
    for n in range(npoints):
        for ii,track in enumerate(tracks):
            eep_mass_age_relations['eep-{}'.format(n)]['mass-init'][ii] = track['star_mass'][0]
            eep_mass_age_relations['eep-{}'.format(n)]['age'][ii] = track['star_age'][n]

            for key in keys:
                eep_tracks['eep-{}'.format(n)][key][ii] = track[key][n]

    print('\t--> All tracks and M_init - Age relations built')
    print('--> Creating interpolation functions for all tracks and M_init - Age relations')
    interpolated_mass_age_relations = {}
    interpolated_eep_tracks = {'eep-{}'.format(n): { key: [] for key in keys} for n in range(npoints)}
    for n in range(npoints):
        x = eep_mass_age_relations['eep-{}'.format(n)]['age']
        y = eep_mass_age_relations['eep-{}'.format(n)]['mass-init']
        x,y = list(sort_y_on_x(x,y))
        interpolated_mass_age_relations['eep-{}'.format(n)] = Akima1DInterpolator(x,y)
        for key in keys:
            xx = eep_tracks['eep-{}'.format(n)]['star_mass']
            xy = eep_tracks['eep-{}'.format(n)][key]
            xx,xy = list(sort_y_on_x(xx,xy))
            ai_func = Akima1DInterpolator(xx,xy)
            interpolated_eep_tracks['eep-{}'.format(n)][key] = ai_func
    print('\t--> All interpolation functions constructed')

    print('--> Looping over all ages to interpolate isochrones')
    ibar = Bar('Calculating...',max=nages)
    for ii,i_age in enumerate(i_ages):
        # print('Constructing isochrone at {:.1f} Myr'.format(i_age*1e-6))
        mass_values = [ interpolated_mass_age_relations['eep-{}'.format(n)](i_age) for n in range(npoints) ]
        for key in keys:
            for n,m_val in enumerate(mass_values):
                isochrones['age-{0:04d}'.format(ii)][key][n] = interpolated_eep_tracks['eep-{}'.format(n)][key](m_val)
        ibar.next()
    ibar.finish()
    print('\t--> All isochrones constructed')

    fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))
    for track in tracks:
        ax.plot(track['log_Teff'],track['log_L'],'k-')
    for ii in range(len(i_ages)):
        ax.plot(isochrones['age-{0:04d}'.format(ii)]['log_Teff'],isochrones['age-{0:04d}'.format(ii)]['log_L'],'r--',alpha=0.5)

    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$\log T_{\rm eff}/K$',fontsize='14')
    ax.set_ylabel(r'$\log L/{L_{\odot}}$',fontsize='14')
    grid_type,ov_str,zini_str,mlt_str,fov_str,ldm_str,plot_save_path = pars
    zini,mlt,fov,ldm = float(zini_str)*1e-4,float(mlt_str)*1e-2,float(fov_str)*1e-4,float(ldm_str)*1e-2
    ax.set_title(r'$Z_{\rm ini}=%f ; \alpha_{\rm MLT}=%f ; %s_{\rm ov}=%f ; \log(D_{\rm ext})=%f$'%(zini,mlt,ov_str,fov,ldm),fontsize=12)
    fig.savefig('{0}{1}_Zini{2}_MLT{3}_ov{4}_logDext{5}-isochrones.png'.format(plot_save_path,grid_type,zini_str,mlt_str,fov_str,ldm_str))
    fig.clear()



    with open(savename,'w') as fout:
        header = '# AGE[Myr]  %s'%' '.join( [ '%s'%key for key in keys ] )
        fout.write(header+'\n')
        for ii,i_age in enumerate(i_ages):
            for n in range(len(isochrones['age-{0:04d}'.format(ii)]['star_mass'])):
                # print(n,'/',len(isochrones['age-%i'%cc]['star_mass']))
                fout.write( '%f %s \n'%(i_age,' '.join( [ '%.8f'%isochrones['age-{0:04d}'.format(ii)][key][n] for key in keys] )) )

    return
