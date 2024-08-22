import pandas as pd

# Specifica il percorso del file .parquet
file_path = 'challenge_campus_biomedico_2024.parquet'

# Leggi il file .parquet
df = pd.read_parquet(file_path)

# Visualizza le prime righe del DataFrame
print(df.head())

# Visualizza le informazioni del DataFrame
print(df.info())
