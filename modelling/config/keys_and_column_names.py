COLUMN_TRANSLATION_PHYTOPLANKTON = {
    'Gewässer': 'water_body',
    'Messstelle': 'measurement_point',
    'Mst_Nr_Bio_Ch': 'measurement_number_bio_chem',
    'Probe_Nr_Bio': 'sample_number_bio',
    'Probe_Nr_Ch': 'sample_number_chem',
    'Datum': 'date',
    'Klasse': 'class',
    'DV': 'diversity_index',
    'ID': 'species_id',
    'Taxon': 'taxon',
    'Zellzahl_ml': 'cell_count_ml',
    'Biovol_cm³_m³': 'biovolume_cm3_m3'
}

IDENTIFIER_KEYS = [
    'water_body',
    'measurement_number_bio_chem',
    'date',
]

COLUMN_TRANSLATION_CHEMIE = {
    'PN-Nummer': 'pn_number',
    'Datum': 'date',
    'MST-Nr.': 'measurement_number_bio_chem',
    'Gewässer': 'water_body',
    'Probenahmestelle': 'sampling_location',
    'ST in m': 'depth_m',
    'T in °C': 'temperature_celsius',
    'O2 in mg/l': 'oxygen_mg_per_l',
    'pH in ': 'ph',
    'Lf in µS/cm': 'conductivity_us_per_cm',
    'NH4-(FIA) in mg/l': 'ammonium_fia_mg_per_l',
    'NH4 in mg/l': 'ammonium_mg_per_l',
    'NO3 in mg/l': 'nitrate_mg_per_l',
    'TN in mg/l': 'total_nitrogen_mg_per_l',
    'o-PO4-(FIA) in mg/l': 'orthophosphate_fia_mg_per_l',
    'o-PO4 in mg/l': 'orthophosphate_mg_per_l',
    'P ges. in mg/l': 'total_phosphorus_mg_per_l',
    'P ges.-(FIA) in mg/l': 'total_phosphorus_fia_mg_per_l',
    'SiO2 gel. in mg/l': 'dissolved_silica_mg_per_l',
    'SiO2 gel.-(FIA) in mg/l': 'dissolved_silica_fia_mg_per_l',
    'Chl.-a in µg/l': 'chlorophyll_a_ug_per_l',
    'Phae in µg/l': 'phaeophytin_ug_per_l'
}

MEASUREMENT_COLS = [
    'depth_m',
    'temperature_celsius',
    'oxygen_mg_per_l',
    'ph',
    'conductivity_us_per_cm',
    'ammonium_fia_mg_per_l',
    'ammonium_mg_per_l',
    'nitrate_mg_per_l',
    'total_nitrogen_mg_per_l',
    'orthophosphate_fia_mg_per_l',
    'orthophosphate_mg_per_l',
    'total_phosphorus_mg_per_l',
    'total_phosphorus_fia_mg_per_l',
    'dissolved_silica_mg_per_l',
    'dissolved_silica_fia_mg_per_l',
    'chlorophyll_a_ug_per_l',
    'phaeophytin_ug_per_l',
]

GENUS_COLUMNS = [
    'Actinocyclus', 'Anabaena', 'Aphanizomenon', 'Asterionella', 'Aulacoseira',
    'Ceratium', 'Chlamydomonas', 'Chrysochromulina', 'Chrysoflagellaten', 'Coelastrum',
    'Cryptomonas', 'Cyclostephanos', 'Cyclotella', 'Diatoma', 'Dinobryon',
    'Fragilaria', 'Gymnodinium', 'Nitzschia', 'Oocystis', 'Peridinium',
    'Planktothrix', 'Pseudanabaena', 'Rhodomonas', 'Scenedesmus',
    'Stephanodiscus', 'Tabellaria', 'Zentrale'
]

SITE_KEYS = ["water_body", "measurement_number_bio_chem"]