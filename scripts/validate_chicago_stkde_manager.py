__author__ = 'gabriel'


REGIONS = (
    # 'chicago_central',
    # 'chicago_southwest',
    # 'chicago_south',
    # 'chicago_far_southwest',
    'chicago_far_southeast',
    'chicago_west',
    'chicago_northwest',
    'chicago_north',
    'chicago_far_north',
)

CRIME_TYPES = (
    'burglary',
    'assault',
)

if __name__ == '__main__':
    
    from scripts.validate_chicago_stkde import main
    for r in REGIONS:
        for ct in CRIME_TYPES:
            main(r, ct)