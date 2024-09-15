import pandas as pd

# Carichiamo il dataset
file_path = 'dataset_per_post_proc.csv'  # Sostituisci con il percorso corretto
df = pd.read_csv(file_path)

# Rimuoviamo le colonne non necessarie
df_clean = df.drop(columns=['semester', 'year', 'duration_minutes', 'age'])

# Selezioniamo le colonne che identificano i professionisti sanitari e il sesso
columns_profession = [col for col in df_clean.columns if 'codice_tipologia_professionista_sanitario' in col]
columns_sesso = ['sesso_female', 'sesso_male']

# Creiamo una nuova colonna che descrive la combinazione di tipologia professionista sanitario e sesso
df_clean['combinazione'] = df_clean[columns_profession + columns_sesso].apply(lambda row: '-'.join([col.split('_')[-1] for col, val in zip(columns_profession + columns_sesso, row) if val]), axis=1)

# Raggruppiamo per combinazione e cluster, calcoliamo i conteggi
grouped = df_clean.groupby(['combinazione', 'cluster']).size().unstack(fill_value=0)

# Calcoliamo le percentuali per ciascun cluster
grouped['percentuale_cluster_0'] = (grouped[0] / grouped[0].sum()) * 100
grouped['percentuale_cluster_1'] = (grouped[1] / grouped[1].sum()) * 100

# Creiamo un nuovo DataFrame con la colonna combinazione e le percentuali per cluster
result = pd.DataFrame({
    'combinazione': grouped.index,
    'percentuale_cluster_0': grouped['percentuale_cluster_0'],
    'percentuale_cluster_1': grouped['percentuale_cluster_1']
}).reset_index(drop=True)

# Stampiamo il DataFrame con le percentuali
print("Percentuali delle combinazioni nei cluster:")
print(result)

# Calcolo della purezza di ciascun cluster (distribuzione delle classi di incremento)
purezza_cluster = df_clean.groupby('cluster')['incremento'].value_counts(normalize=True).unstack(fill_value=0) * 100
purezza_cluster.columns = [f'classe_{col}_percentuale' for col in purezza_cluster.columns]

# Creiamo un DataFrame separato per la purezza
print("\nPurezza di ogni cluster:")
print(purezza_cluster)

# Raggruppiamo per combinazione e classe incremento, calcoliamo i conteggi
grouped_incremento = df_clean.groupby(['combinazione', 'incremento']).size().unstack(fill_value=0)

# Calcoliamo le percentuali per ciascuna classe di incremento
grouped_incremento['percentuale_incremento_0'] = (grouped_incremento[0] / grouped_incremento[0].sum()) * 100
grouped_incremento['percentuale_incremento_1'] = (grouped_incremento[1] / grouped_incremento[1].sum()) * 100
grouped_incremento['percentuale_incremento_3'] = (grouped_incremento[3] / grouped_incremento[3].sum()) * 100

# Creiamo un nuovo DataFrame con la colonna combinazione e le percentuali per incremento
result_incremento = pd.DataFrame({
    'combinazione': grouped_incremento.index,
    'percentuale_incremento_0': grouped_incremento['percentuale_incremento_0'],
    'percentuale_incremento_1': grouped_incremento['percentuale_incremento_1'],
    'percentuale_incremento_3': grouped_incremento['percentuale_incremento_3']
}).reset_index(drop=True)

# Stampiamo il DataFrame con le percentuali per incremento
print("\nPercentuali delle combinazioni per classe di incremento:")
pd.set_option('display.max_rows', None)  # Mostra tutte le righe
pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Non troncare la visualizzazione delle colonne
print(result_incremento)
