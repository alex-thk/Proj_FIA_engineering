import pandas as pd


class DataReduction:
    def __init__(self, df):
        self.df = df

    def remove_insignificant_columns(self):
        # removing all records with missing values in the column 'comune_residenza'
        # a 0.03% of the total records will be removed (not a big deal)
        self.df.dropna(subset=['comune_residenza'], inplace=True)

        # 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns are not useful anymore
        # since now we have the 'duration' column
        self.df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)  # inplace=True to modify the original dataframe

        # now dropping other columns that are not significant for the analysis
        list_of_cols = ['id_prenotazione', 'id_paziente', 'codice_regione_residenza', 'codice_asl_residenza', 'codice_provincia_residenza',
                        'codice_comune_residenza', 'tipologia_servizio', 'data_contatto', 'codice_regione_erogazione', 'codice_asl_erogazione',
                        'codice_provincia_erogazione', 'struttura_erogazione', 'codice_struttura_erogazione', 'id_professionista_sanitario',
                        'tipologia_professionista_sanitario', 'data_nascita']
        self.df.drop(columns=list_of_cols, inplace=True)
