# PAMAP2 – Human Activity Recognition with RNN and SVM

Questo progetto implementa una pipeline completa di Machine Learning per la classificazione delle attività umane sul dataset **PAMAP2**, confrontando modelli deep (GRU e LSTM) con un approccio classico basato su **SVM + feature engineering**.

L’obiettivo è mostrare come, a parità di preprocessing e suddivisione dei dati, modelli diversi possano ottenere prestazioni differenti in funzione della quantità e della struttura dei dati disponibili.

---

## Struttura del progetto
```bash
src/
├── main.py
└── pamap2_project/
├── config/
│ └── default.yaml
├── data/
│ ├── make_windows.py
│ ├── split_windows.py
│ ├── normalize.py
│ ├── window_features.py
│ └── npz_dataset.py
├── models/
│ └── rnn.py
├── train/
│ ├── train_rnn.py
│ └── train_svm.py
├── eval/
│ ├── evaluate_rnn.py
│ └── evaluate_svm.py
└── utils/
├── config_loader.py
├── metrics.py
└── run_manager.py
```
Cartelle generate automaticamente:
```bash
data/processed/
├── windows/
├── splits/
├── normalized/
└── features_svm/

reports/runs/
└── run_001/
└── run_002/
```

Ogni cartella `run_XXX` contiene:
- `config.yaml`
- modello migliore salvato
- metriche di validazione e test

---

## Installazione

Clona il repository e installa le dipendenze:

```bash
pip install -r requirements.txt
```

## Configurazione
Tutti i parametri del progetto sono definiti in:
src/pamap2_project/config/default.yaml

-Da qui è possibile configurare:
- attività considerate
- parametri di finestratura
- suddivisione train / validation / test
- iperparametri dei modelli RNN
- configurazione SVM
- early stopping
- learning rate scheduler

## Pepeline generale
Il progetto segue un pipeline modulare:
```bash
windows → split → normalize
```
Successivamente la pipeline si biforca:
Modelli RNN (GRU/LSTM)
```bash
train-rnn → evaluate
```
Modello SVM:
```bash
svm-features → train-svm → evaluate-svm
```

## Comandi principali:
```bash
python -m src.main --stage <nome_stage>
```

Preprocesssing comune:
```bash
python -m src.main --stage windows
python -m src.main --stage split
python -m src.main --stage normalize
```

RNN:
```bash
python -m src.main --stage train-rnn
python -m src.main --stage evaluate
```

SVM:
```bash
python -m src.main --stage svm-features
python -m src.main --stage train-svm
python -m src.main --stage evaluate-svm
```

Per la SVM, ogni finestra temporale viene trasformata in un vettore tramite:
- media (mean)
- deviazione standard (std)
- minimo (min)
- massimo (max)
Quindi:
```bash
(T, F) → (4F,)
```

## Metriche
Per tutti i modelli vengono calcolate:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Loss

La metrica principale utilizzata per la selezione del miglior modello è il F1-score, più robusto in presenza di classi sbilanciate.

## Obiettivo del confronto
Il progetto dimostra che:
- i modelli deep (GRU e LSTM) sfruttano direttamente la dinamica temporale dei segnali
- la SVM utilizza una rappresentazione compatta basata su feature statistiche
- con dataset di dimensioni ridotte e pochi soggetti, un approccio classico ben progettato può superare modelli deep più complessi

## Riproducibilità
Ogni esperimento viene salvato in:
```bash
reports/runs/run_XXX/
```
contenente:
- configurazione completa del run (config.yaml)
- modello migliore
- metriche di validazione e test
- Questo garantisce la completa riproducibilità degli esperimenti

## Dataset
Il dataset PAMAP2 deve essere posizionato in:
```bash
data/raw/Protocol/
```
contenente i file .dat relativi ai soggetti.

## Conclusione
Questo progetto fornisce una pipeline completa, modulare e riproducibile per la Human Activity Recognition, permettendo un confronto tra:
- modelli deep sequence-based (GRU, LSTM);
- modelli classici basati su feature engineering e SVM.
Il progetto è pensato come lavoro accademico per l’esame di Machine Learning.
