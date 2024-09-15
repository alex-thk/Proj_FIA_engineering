# Analisi e Clustering dei Dati sulle Teleassistenze dal 2019 al 2022

## Descrizione

Questo progetto si propone di analizzare e clusterizzare i dati sulle teleassistenze dal 2019 al 2022. In particolare l'obiettivo è quello di identificare caratteristiche comuni fra i campioni che hanno determinato negli anni un incremento nel numero di teleassistenze. Attraverso una serie di passaggi che includono la pulizia dei dati, la riduzione della dimensionalità, la trasformazione, la standardizzazione, la rilevazione degli outlier, la costruzione della variabile incremento, il clustering supervisionato con target incremento e la post-elaborazione, si mira a identificare pattern significativi e segmenti all'interno dei dati raccolti.

## Dataset

### Descrizione

Il dataset utilizzato in questo progetto è denominato `challenge_campus_biomedico_2024.parquet` e contiene informazioni relative agli appuntamenti biomedici nel campus per l'anno 2024. I dati includono dettagli sui pazienti, sui professionisti sanitari, sulle strutture erogatrici e sulle caratteristiche degli appuntamenti.

### Caratteristiche Principali

| Nome Variabile                           | Descrizione                                                                 | Tipo    |
|------------------------------------------|-----------------------------------------------------------------------------|---------|
| **id_prenotazione**                      | Identificativo univoco di una singola teleassistenza                       | String  |
| **id_paziente**                          | Codice identificativo univoco del paziente                                  | String  |
| **data_nascita**                         | Data di nascita del paziente                                                | String  |
| **sesso**                                | Genere del paziente                                                        | String  |
| **regione_residenza**                    | Regione di residenza del paziente                                          | String  |
| **codice_regione_residenza**             | Codice della regione di residenza del paziente                              | String  |
| **asl_residenza**                        | ASL di residenza del paziente                                              | String  |
| **codice_asl_residenza**                 | Codice dell'ASL di residenza del paziente                                  | String  |
| **provincia_residenza**                  | Provincia di residenza del paziente                                        | String  |
| **codice_provincia_residenza**           | Codice della provincia di residenza del paziente                            | String  |
| **comune_residenza**                     | Città di residenza del paziente                                            | String  |
| **codice_comune_residenza**              | Codice della città di residenza del paziente                                | String  |
| **tipologia_servizio**                   | Tipologia del servizio offerto dalla piattaforma di telemedicina           | String  |
| **descrizione_attivita**                 | Descrizione dell'attività svolta durante l'appuntamento                    | String  |
| **codice_descrizione_attivita**          | Codice associato alla descrizione dell'attività svolta                      | String  |
| **data_contatto**                        | Data di contatto per l'appuntamento                                         | String  |
| **regione_erogazione**                   | Regione di erogazione del servizio                                          | String  |
| **codice_regione_erogazione**            | Codice della regione di erogazione del servizio                              | String  |
| **asl_erogazione**                       | ASL responsabile dell'erogazione del servizio                               | String  |
| **codice_asl_erogazione**                | Codice dell'ASL responsabile dell'erogazione del servizio                    | String  |
| **provincia_erogazione**                 | Provincia in cui viene erogato il servizio                                  | String  |
| **codice_provincia_erogazione**          | Codice della provincia in cui viene erogato il servizio                      | String  |
| **struttura_erogazione**                 | Nome della struttura sanitaria che eroga il servizio                         | String  |
| **codice_struttura_erogazione**          | Codice identificativo della struttura sanitaria che eroga il servizio        | String  |
| **tipologia_struttura_erogazione**        | Tipologia della struttura sanitaria che eroga il servizio                     | String  |
| **codice_tipologia_struttura_erogazione** | Codice della tipologia della struttura sanitaria che eroga il servizio         | String  |
| **id_professionista_sanitario**          | Codice identificativo univoco del professionista sanitario erogatore         | String  |
| **tipologia_professionista_sanitario**    | Tipologia del professionista sanitario erogatore                            | String  |
| **codice_tipologia_professionista_sanitario** | Codice della tipologia del professionista sanitario erogatore                | String  |
| **data_erogazione**                      | Data in cui è stato erogato il servizio                                      | String  |
| **ora_inizio_erogazione**                | Timestamp di inizio dell'erogazione del servizio (se già effettuata)        | String  |
| **ora_fine_erogazione**                  | Timestamp di fine dell'erogazione del servizio (se già effettuata)          | String  |
| **data_disdetta**                        | Timestamp di cancellazione dell'erogazione del servizio (se visit cancellata)| String  |


## Funzionalità

- **Pulizia dei Dati**: Gestione dei valori mancanti, rimozione delle colonne non rilevanti e gestione degli appuntamenti cancellati.
- **Riduzione dei Dati**: Eliminazione delle colonne insignificanti per ottimizzare l'analisi.
- **Trasformazione dei Dati**: Mappatura delle regioni di residenza e creazione di variabili dummy.
- **Rilevazione degli Outlier**: Identificazione e rimozione di valori anomali nelle variabili chiave.
- **Standardizzazione**: Normalizzazione delle feature per migliorare le prestazioni degli algoritmi di clustering.
- **Analisi di Clustering er la creazione di Incremento**: Determinazione del numero ottimale di cluster, valutazione della stabilità delle feature e clustering per ogni semestre di ogni anno. Infine confronto tra raggruppamenti relativi allo stesso semestre di anni successivi per valutare l'incremento delle teleassistenze.
- **Clustering Supervisionato**: Applicazione di un algoritmo di clustering supervisionato con target incremento per individuare le caratteristiche simili fra campioni che hanno pari incremento.
- **Post-Elaborazione**: Creazione di statistiche per ogni cluster e salvataggio dei risultati finali.


## Installazione

1. **Clona il repository**

   ```bash
   git clone https://github.com/tuo-username/tuo-repo.git
   cd tuo-repo

2. **Crea un ambiente virtuale**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
3. **Installa i requisiti**

   ```bash
    pip install -r requirements.txt
   
## Utilizzo

1. **Posiziona il file dei dati**

    Assicurati che il file `challenge_campus_biomedico_2024.parquet` sia nella cartella `data/`.

2. **Esegui lo script principale**

    ```bash
    python main.py
    ```

    Questo script eseguirà tutti i passaggi di pulizia, trasformazione, clustering e post-elaborazione, salvando i risultati finali in `final_dataset_with_categories.csv` e le statistiche dei cluster in `stats_by_cluster.json`.

