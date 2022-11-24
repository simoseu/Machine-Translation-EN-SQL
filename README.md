# Machine-Translation-EN-SQL

Il seguente progetto prevede l'implementazione del fine tuning di due modelli per il task di machine translation:
* Da Inglese a Linguaggio SQL & da Linguaggio SQL a Inglese.

## Modelli di riferimento

Di seguito i modelli utilizzati:
* [Da Inglese a SQL](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL);
* [Da SQL a Inglese](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL-sql-to-en).

Questi sono modelli ***T5*** [Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned sul dataset ***WikiSQL*** [WikiSQL](https://github.com/salesforce/WikiSQL).

## Fine Tuning su Spider

Il progetto consiste nell'esecuzione di un ulteriore fine-tuning, dei modelli sopraccitati, sul dataset [Spider](https://github.com/jkkummerfeld/text2sql-data/blob/master/data/spider.json)

*Il dataset è stato diviso in train/test rispettivamente con una percentuale 0.8/0.2*

Gli script:
* `EN_SQL_training.py`: esegue il fine tuning del modello per il task di machine translation da Inglese a SQL;
* `SQL_EN_training.py`: esegue il fine tuning del modello per il task di machine translation da SQL a Inglese.

Rispettivamente sono stati usati i modelli pre-finetuned analizzati precedentemente.

### How to run
* Aver scaricato il file ***.json*** del dataset [Spider](https://github.com/jkkummerfeld/text2sql-data/blob/master/data/spider.json);
* Modificare la variabile in riga ***8*** `db_path` inserendo il path in cui è presente il file del dataset;
* Modificare la variabile in riga ***7*** `save_location` inserendo il path in cui si desidera salvare il nuovo modello;
* **(Optional)** E' possibile scegliere i parametri di training andando a modificare le righe ***32***-***44***;
* Eseguire lo script.

## Evaluation

Per valutare i modelli creati a seguito del fine tuning è stata usata la metrica [Meteor](https://huggingface.co/spaces/evaluate-metric/meteor).

Gli script:
* `EN_SQL_evaluation.py`: esegue l'evaluation del modello ***EN_SQL*** creato attraverso l'esecuzione del precedente script, calcolando la media dello score ***Meteor*** del test set.
* `SQL_EN_evaluation.py`: esegue l'evaluation del modello ***SQL_EN*** creato attraverso l'esecuzione del precedente script, calcolando la media dello score ***Meteor*** del test set.

### How to run
* Aver scaricato il file ***.json*** del dataset [Spider](https://github.com/jkkummerfeld/text2sql-data/blob/master/data/spider.json);
* Modificare la variabile in riga ***8*** `db_path` inserendo il path in cui è presente il file del dataset;
* Modificare la variabile in riga ***28*** `model_path` inserendo il path in cui è presente il modello fine tuned;
* Eseguire lo script.

## Risultati ottenuti:
Di seguito i risultati ottenuti dai due modelli:
* Il modello **EN_SQL** (il cui training è stato di 8 epoche) ha una media score ***Meteor*** di 48%;
* Il modello **SQL_EN** (il cui training è stato di 5 epoche) ha una media score ***Meteor*** di 44%;

## Problemi

Il limite principale che si è presentato è legato all'architettura su cui sono stati effettuati i test.
Avendo una 'scarsa' potenza di calcolo a disposizione, non è stato possibile:
1. Utilizzare la GPU per eseguire il training, questo ha comportato un dilatamento nei tempi di esecuzione (e.g. 10 ore per effettuare il training di 8 epochs).

2. Usare la totalità dei dataset a nostra disposizione, fare il training su più dati avrebbe potuto portare a una migliore accuracy del modello fine tuned.

