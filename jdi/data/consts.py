import re


BOROUGH_CODES = [
    'TX',
    'TW',
    'WW',
    'FH',
    'BS',
    'NI',
    'YR',
    'YE',
    'JI',
    'JC',                    
    'HT',
    'LX',
    'PY',
    'RY',
    'ZD',    
    'RG',
    'PL',
    'XH',               
    'EK',
    'VW',
    'VK',
    'CW',
    'MD',
    'GD',
    'SX',
    'ZT',
    'XB',
    'KG',
    'QK',
    'XD',
    'ID',
    'KF',
    'KD',
    'QA',
]


BOROUGH_NAME_MAP = {
    'BS': 'Kensington & Chelsea',
    'CW': 'Westminster',
    'EK': 'Camden',
    'FH': 'Hammersmith & Fulham',
    'GD': 'Hackney',
    'HT': 'Tower Hamlets',
    'ID': 'id',  ## FIXME
    'JC': 'Waltham Forest',
    'JI': 'Redbridge',
    'KD': 'Havering',
    'KF': 'Newham',
    'KG': 'Barking & Dagenham',
    'LX': 'Lambeth',
    'MD': 'Southwark',
    'NI': 'Islington',
    'PL': 'Lewisham',
    'PY': 'Bromley',
    'QA': 'Harrow',
    'QK': 'Brent',
    'RG': 'Greenwich',
    'RY': 'Bexley',
    'SX': 'Barnet',
    'TW': 'Richmond Upon Thames',
    'TX': 'Hounslow',
    'VK': 'Kingston Upon Thames',
    'VW': 'Merton',
    'WW': 'Wandsworth',
    'XB': 'Ealing',
    'XD': 'xd',  ## FIXME
    'XH': 'Hillingdon',
    'YE': 'Enfield',
    'YR': 'Haringey',
    'ZD': 'Croydon',
    'ZT': 'Sutton',
}

MAJOR_CRIME_TYPES = [
    'Burglary',
    'Criminal Damage',
    'Drugs',
    'Other Notifiable Offences',
    'Robbery',
    'Theft & Handling',
    'Violence Against The Person',
]

MINOR_CRIME_TYPES = [
    'Assault With Injury',
    'Burglary In A Dwelling',
    'Burglary In Other Buildings',
    'Business Property',
    'Common Assault',
    'Criminal Damage To Dwelling',
    'Criminal Damage To Motor Vehicle',
    'Criminal Damage To Other Building',
    'Drug Trafficking',
    'Going Equipped',
    'Handling Stolen Goods',
    'Harassment',
    'Motor Vehicle Interference & Tampering',
    'Murder',
    'Offensive Weapon',
    'Other Criminal Damage',
    'Other Drugs',
    'Other Notifiable',
    'Other Theft',
    'Other Violence',
    'Personal Property',
    'Possession Of Drugs',
    'Theft From Motor Vehicle',
    'Theft From Shops',
    'Theft Person',
    'Theft/Taking Of Motor Vehicle',
    'Theft/Taking Of Pedal Cycle',
    'Wounding/GBH'
]


# make file-friendly versions of the crime types
CRIME_TYPE_NAME_MAP = {}
for k in (list(MAJOR_CRIME_TYPES) + list(MINOR_CRIME_TYPES)):
    x = re.sub(r'&', 'and', k.lower())
    x = re.sub(r'/', ' or ', x)
    x = re.sub(r' +', ' ', x)
    CRIME_TYPE_NAME_MAP[k] = x.replace(' ', '_')
    

# make file-friendly versions of the borough names
BOROUGH_FILENAME_MAP = {}
for k, v in BOROUGH_NAME_MAP.items():
    x = re.sub(r'&', 'and', v.lower())
    x = re.sub(r' +', '_', x)
    BOROUGH_FILENAME_MAP[k] = x