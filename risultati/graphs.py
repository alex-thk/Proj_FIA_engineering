import json
import matplotlib.pyplot as plt
import numpy as np

# Caricamento del JSON
with open('yearly_stats.json') as f:
    data = json.load(f)

# Estrazione dei dati
anni = list(data.keys())
regioni = list(data['2019']['residenza_sum'].keys())

# 1. Creazione di istogrammi per la distribuzione delle residenze per regione per ogni anno
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

for i, anno in enumerate(anni):
    ax = axs[i // 2, i % 2]
    residenze = [data[anno]['residenza_sum'][regione] for regione in regioni]
    ax.bar(regioni, residenze, color='skyblue')
    ax.set_xticklabels(regioni, rotation=90)
    ax.set_title(f'Distribuzione delle teleassistenze per regione ({anno})')
    ax.set_ylabel('Numero di teleassistenze')

plt.tight_layout()
plt.show()

# 2. Creazione dei grafici a torta per la distribuzione di genere per ogni anno
sesso_per_anno = {anno: data[anno]['sesso_sum'] for anno in anni}

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

for i, anno in enumerate(anni):
    labels = sesso_per_anno[anno].keys()
    sizes = sesso_per_anno[anno].values()
    ax = axs[i // 2, i % 2]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightskyblue'])
    ax.set_title(f'Distribuzione di genere {anno}')

plt.tight_layout()
plt.show()

# 3. Creazione di un grafico a linee per i codici tipologia (2019-2022)
codici = list(data['2019']['codice_tipologia_sum'].keys())
codici_totali = {codice: [] for codice in codici}

for anno in anni:
    for codice in codici:
        codici_totali[codice].append(data[anno]['codice_tipologia_sum'][codice])

plt.figure(figsize=(10,6))
for codice, valori in codici_totali.items():
    plt.plot(anni, valori, label=codice)



plt.title('Andamento delle tipologie di professioni sanitarie (2019-2022)')
plt.xlabel('Anno')
plt.ylabel('Numero di teleassistenze')
plt.legend()
plt.tight_layout()
plt.show()

# Estrazione dei dati per la somma di maschi e femmine per ogni anno
anni = list(data.keys())
totale_teleassistenze = []

for anno in anni:
    totale = data[anno]['sesso_sum']['female'] + data[anno]['sesso_sum']['male']
    totale_teleassistenze.append(totale)

# Creazione del grafico a linee per il numero totale di teleassistenze
plt.figure(figsize=(10,6))
plt.plot(anni, totale_teleassistenze, marker='o', color='purple', label='Totale Teleassistenze')
plt.title('Andamento del numero totale di teleassistenze (2019-2022)')
plt.xlabel('Anno')
plt.ylabel('Numero di Teleassistenze')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
