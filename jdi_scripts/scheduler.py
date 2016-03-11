import optimise_kde_bandwidth_network as okbn
import optimise_kde_bandwidth_planar as okbp
import cris_planar_prediction as cpp
import cris_network_prediction as cnp
from analyse_bandwidth_optimisation import compute_and_save_aggregated_likelihood_grids_one_crime
from jdi.data import consts


if __name__ == "__main__":
    
    exclude_boroughs = ('ek', 'cw', 'ni', 'yr', 'qk', 'sx')
    boroughs = list(set([t.lower() for t in consts.BOROUGH_CODES]) - set(exclude_boroughs))
    minor_crime_type = 'Burglary In A Dwelling'
    planar_subdir = 'planar_bandwidth_linearexponential'
    net_subdir = 'network_bandwidth_linearexponential'
    
    for bo in boroughs:
        try:
            okbp.run_one(bo, minor_crime_type=minor_crime_type)
        except Exception as exc:
            # move past exceptions as well as possible
            print repr(exc)
            
        try:
            okbn.run_one(bo, minor_crime_type=minor_crime_type)
        except Exception as exc:
            # move past exceptions as well as possible
            print repr(exc)                      

    try:
        compute_and_save_aggregated_likelihood_grids_one_crime(minor_crime_type, planar_subdir)
    except Exception as exc:
        # move past exceptions as well as possible
        print repr(exc)                                          

    try:
        compute_and_save_aggregated_likelihood_grids_one_crime(minor_crime_type, net_subdir)
    except Exception as exc:
        # move past exceptions as well as possible
        print repr(exc)
             
    for bo in boroughs:
        try:
            cpp.validate_one(bo, minor_crime_type=minor_crime_type)
        except Exception as exc:
            # move past exceptions as well as possible
            print repr(exc)
                    
        try:
            cnp.validate_one(bo, minor_crime_type=minor_crime_type)
        except Exception as exc:
            # move past exceptions as well as possible
            print repr(exc)