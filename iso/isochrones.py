import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator,interp1d,Akima1DInterpolator

from entropy3.general.smoothing import smooth

def sort_tracks_on_mass(tracks):

    masses = [track['star_mass'][0] for track in tracks]
    zipp = list(zip(masses,tracks))
    zipp.sort(key=lambda x:x[0])
    masses,tracks = list(zip(*zipp))
    yield tracks


def make_isochrones_dict(i_ages,keys):

    key_dict = {key: [] for key in keys}
    isochrones = { 'age-%i'%cc: key_dict for cc,i_age in enumerate(i_ages) }

    yield isochrones


def get_subsample(n,tracks,keys):

    # eep_subsample = { key: [ track[key][n] for track in tracks  ] for key in keys }
    eep_subsample = { key: [] for key in keys }
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

    # plt.plot(sorted_ages,sorted_masses,'r--')
    # sorted_ages = smooth(np.array(sorted_ages),3,'flat')
    # plt.plot(sorted_ages,sorted_masses,'k-o')
    # plt.axvline(i_age,color='blue')
    # plt.show()

    i_age_func = list(make_interpolation_function(sorted_ages,sorted_masses))[0]
    mass_coord = i_age_func(i_age)
    isochrones = list(populate_isochrones(cc,mass_coord,masses,subsample,keys,isochrones))[0]

    yield isochrones


def populate_isochrones(cc,mass_coord,masses,subsample,keys,isochrones):

    for key in keys:
        cval = subsample[key]
        sorted_masses,sorted_cval = list(sort_y_on_x(masses,cval))
        i_cval_func = list(make_interpolation_function(sorted_masses,sorted_cval))[0]
        isochrones['age-{}'.format(cc)][key].append(i_cval_func(mass_coord))

    yield isochrones


def construct_isochrones(eep_tracks,i_ages,keys,savename):

    #-> Sort tracks based on mass to ensure ease later
    tracks = list(sort_tracks_on_mass(eep_tracks))[0]
    npoints = len( tracks[0]['star_age'] )

    #-> create a dictionary where each age is a dictionary for all quantities listed above
    isochrones = list(make_isochrones_dict(i_ages,keys))[0]

    #-> Loop through all ages and create the isochrone for that age
    for cc,i_age in enumerate(i_ages):

        print('ISOCHRONE FOR: ',i_age/1e6,' Myrs')

        #-> Loop over each track at the same EEP to generate an isochrone
        for n_eep in range(npoints):

            eep_subsample,eep_masses,eep_ages = list(get_subsample(n_eep,tracks,keys))

            if ( ( min(eep_ages) < i_age < max(eep_ages) ) & (len(eep_masses) > 3) ) :

                isochrones = list(loop_over_subsample(i_age,cc,eep_subsample,eep_masses,eep_ages,keys,isochrones))[0]
        # plt.show()
        #         plt.figure(1)
        #         plt.plot( eep_masses, np.log10(eep_ages), 'ko-')
        #         plt.axvline(i_age_func(i_age),color='red',alpha=0.5)
        #         plt.axhline(np.log10(i_age),color='red',alpha=0.5)
        #
        # plt.show()

                # plt.figure(2)
                # plt.plot(isochrones['age-%i'%cc]['log_Teff'],isochrones['age-%i'%cc]['log_g'],'k-',alpha=0.7)

    for cc,i_age in enumerate(i_ages):
        for key in keys:
            isochrones['age-%i'%cc][key] = np.hstack(isochrones['age-%i'%cc][key])

    # plt.figure(2)
    # plt.ylim(plt.ylim()[::-1])
    #
    # plt.show()

    with open(savename,'w') as fout:
        header = '# AGE[Myr]  %s'%' '.join( [ '%s'%key for key in keys ] )
        fout.write(header+'\n')
        for cc,i_age in enumerate(i_ages):
            for n in range(len(isochrones['age-%i'%cc]['star_mass'])):
                # print(n,'/',len(isochrones['age-%i'%cc]['star_mass']))
                fout.write( '%f %s \n'%(i_age,' '.join( [ '%.8f'%isochrones['age-%i'%cc][key][n] for key in keys] )) )




            #valid_eeps.append(np.hstack(valid_masses))

    #for mass in masses:
        #plt.axvline(mass,color='red',alpha=0.5)
    #for i_age in i_ages:
        #plt.axhline(np.log10(i_age),color='blue',alpha=0.5)

    #plt.show()

    return
