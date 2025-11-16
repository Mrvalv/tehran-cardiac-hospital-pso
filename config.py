CONFIG = {
    'LAT_MIN': 35.65,
    'LAT_MAX': 35.95,
    'LON_MIN': 51.20,
    'LON_MAX': 51.60,

    # Problem constraints (in km)
    'MIN_DISTANCE_FROM_HOSPITAL_KM': 4.0,
    'MAX_POPULATION_DISTANCE_KM': 8.0,   # avg KNN 
    'STRICT_DISTANCE_KM': 3.0,
    'MIN_POPULATION_CENTERS_IN_RANGE': 3,
    'MIN_COVERAGE_DISTANCE_KM': 7.0,

    # PSO hyperparameters
    'N_PARTICLES': 30,
    'MAX_ITERATIONS': 50,
    'INERTIA_WEIGHT': 0.7,
    'COGNITIVE_PARAM': 2.0,
    'SOCIAL_PARAM': 2.0,
}


def get_params():
    """
    Convert CONFIG (km) to internal units (degrees) where needed.
    """
    return {
        'LAT_MIN': CONFIG['LAT_MIN'],
        'LAT_MAX': CONFIG['LAT_MAX'],
        'LON_MIN': CONFIG['LON_MIN'],
        'LON_MAX': CONFIG['LON_MAX'],

        'MIN_DISTANCE_FROM_HOSPITAL': CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM'] / 111,
        'MAX_POPULATION_DISTANCE': CONFIG['MAX_POPULATION_DISTANCE_KM'],  # km
        'STRICT_DISTANCE': CONFIG['STRICT_DISTANCE_KM'] / 111,
        'MIN_POPULATION_CENTERS_IN_RANGE': CONFIG['MIN_POPULATION_CENTERS_IN_RANGE'],
        'MIN_COVERAGE_DISTANCE': CONFIG['MIN_COVERAGE_DISTANCE_KM'] / 111,
    }


def print_config_summary():
    print("=" * 110)
    print("PSO CONFIGURATION SUMMARY")
    print("=" * 110)
    for k, v in CONFIG.items():
        print(f"{k:35s}: {v}")
    print("=" * 110 + "\n")
