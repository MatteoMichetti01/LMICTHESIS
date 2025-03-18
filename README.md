# Sperimentazioni

In questo capitolo verranno mostrati i risultati dall’addestramento di
un Transformer 200k con l’obiettivo di riprodurre gli esiti ottenuti
nello studio presentato nel capitolo precedente, in particolare gli
script sono forniti dalla repository , collegata direttamente ad esso.
L’addestramento è stato reso possibile grazie all’utilizzo del cluster
universitario Caliban.

## Flusso logico della repository

Le operazioni principali per ricreare il caso di studio sono tre:

- Addestrare il Transformer.

- Utilizzare i pesi acquisiti mediante l’addestramento per ottenere le
  probabilità del prossimo simbolo.

- Utilizzare lo script di codifica aritmetica con le probabilità
  ottenute per comprimere.

## Panoramica

La repository si compone di $11$ script in Python. Troviamo nella
cartella *compressor* $4$ file: **compressor.py**, **flac.py**,
**language_model.py** e **png.py**. Lo script **compressor.py**
definisce un’interfaccia per diversi tipi di compressori. Importa vari
algoritmi di compressione, tra cui FLAC, gzip, LZMA e quello basato sul
modello, successivamente li organizza in categorie "classical" e
"arithmetic_coding". Il protocollo Compressor specifica un’interfaccia
callable per le funzioni di compressione, mentre il dizionario
COMPRESS_FN_DICT mappa i nomi dei compressori alle loro rispettive
funzioni di compressione.

<div class="mintedbox">

python class Compressor(Protocol):

def \_\_call\_\_(self, data: bytes, \*args, \*\*kwargs) -\> bytes \|
tuple\[bytes, int\]: """Returns the compressed version of ‘data‘, with
optional padded bits."""

COMPRESSOR_TYPES = ’classical’: \[’flac’, ’gzip’, ’lzma’, ’png’\],
’arithmetic_coding’: \[’language_model’\],

COMPRESS_FN_DICT: Mapping\[str, Compressor\] = ’flac’: flac.compress,
’gzip’: functools.partial(gzip.compress, compresslevel=9),
’language_model’: language_model.compress, ’lzma’: lzma.compress, ’png’:
png.compress,

</div>

Le ulteriori $3$ diciture rappresentano le implementazioni degli
algoritmi di compressione specificati: **flac.py** per i file audio,
**png.py** per il suono e **language_model.py** per utilizzare il
modello con la codifica aritmetica. Ci concentreremo nella descrizione
di quest’ultimo, in quanto i primi due sono di scarso interesse per lo
studio compiuto.

Troviamo poi il codice **arithmetic_coder.py**, che implementa la
codifica aritmetica utilizzata dal modello e **compress.py** che ha lo
scopo di valutare la compressione andando a specificare il tipo di
compressore, il dataset e il numero di chunks da comprimere. Anche
questi verranno descritti in seguito.

Lo script **constants.py** definisce delle costanti utili ed adoperate
in quasi tutti gli script della repository. Ad esempio, mostra il numero
di chunks in cui è diviso enwik9, la dimensione di essi, la taglia
dell’alfabeto usato dal modello e la precisione della codifica
aritmetica.

<div class="mintedbox">

python NUM_CHUNKS = 488281 CHUNK_SIZE_BYTES = 2048 CHUNK_SHAPE_2D = (32,
64) ALPHABET_SIZE = 256 NUM_CHUNKS_CALGARY = 957

ARITHMETIC_CODER_BASE = 2 ARITHMETIC_CODER_PRECISION = 32

</div>

Un ulteriore codice di importanza cruciale è **data_loader.py**, esso
definisce degli iteratori per i dataset che verranno utilizzati sia nel
training che nella compressione. Mi limiterò a mostrare quello di enwik9
perchè di nostro principale interesse rispetto agli altri. I parametri
di iterazione come i chunk e la lunghezza di sequenza sono le costanti
definite in **costants.py**. Dopo aver scaricato enwik9, viene definita
una list di chunk, dove si appenderanno ognuno dei chunk di enwik9
iterati.

<div class="mintedbox">

python def get_enwik9_iterator( num_chunks: int = constants.NUM_CHUNKS,
sequence_length: int = constants.CHUNK_SIZE_BYTES, ) -\>
Iterator\[bytes\]: """Returns an iterator for enwik9 data.""" if not
os.path.exists(’enwik9’): \# Downloading and extracting the dataset.
urllib.request.urlretrieve( ’https://mattmahoney.net/dc/enwik9.zip’,
’enwik9.zip’, ) with zipfile.ZipFile(’enwik9.zip’, ’r’) as zip_ref:
zip_ref.extract(’enwik9’)

all_chunks = \[\] with open(’enwik9’, ’rb’) as file: for \_ in
range(num_chunks): all_chunks.append(file.read(sequence_length)) return
iter(all_chunks)

</div>

Ulteriori codici che meritano una descrizione più dettagliata sono
**train.py**, che addestra il modello, e **transformer.py** che
definisce l’architettura del Transformer. Per ultimo troviamo
**utils.py**, che contiene funzioni importanti per gli script come
quella che consiste nell’azzerare il bit più significativo per poter
rendere i caratteri che non lo sono codificabili in ASCII.

<div class="mintedbox">

python

def zero_most_significant_bit_if_not_ascii_decodable( data: bytes, ) -\>
tuple\[bytes, int\]: """Returns ascii-decodable data & the number of
zeroed most significant bits. """ masked_bits = 0 masked_data = list()

for byte in data: if chr(byte).isascii(): masked_data.append(byte) else:
masked_bits += 1 masked_data.append(byte & 0x7F)

return bytes(masked_data), masked_bits

</div>

### Training

Il processo di training è, appunto, implementato all’interno dello
script **train.py**. Lo script addestra il language model basandosi
sull’architettura Transformer definita nello script **transformer.py**.
La configurazione viene importata con *TransformerConfig*, fornendo cosi
gli iperparametri principali come il numero di strati, il numero di
teste di attenzione e la dimensione degli embedding.

<div class="mintedbox">

class TransformerConfig: """Hyperparameters used in the Transformer
architectures.""" \# Vocabulary size. vocab_size: int \# The dimension
of the first embedding. embedding_dim: int = 64 \# The number of
multi-head attention layers. num_layers: int = 4 \# The number of heads
per layer. num_heads: int = 8 \# The parameter initialization scale for
the embeddings. emb_init_scale: float = 0.02 \# How much larger the
hidden layer of the feedforward network should be \# compared to the
‘embedding_dim‘. widening_factor: int = 4

</div>

L’addestramento inizia con la preparazione del dataset mediante
l’iteratore di dati per enwik9, contenuto in **dataloader.py**,
descritto in precedenza.

<div class="mintedbox">

python data_generator = data_loaders.get_enwik9_iterator(
num_chunks=constants.NUM_CHUNKS // 10, sequence_length=sequence_length,
) dataset = list(data_generator)

</div>

Utilizzando il numero di chunk e la lunghezza delle sequenze
inizializzate in **constant.py**, definiti per enwik9, dividendo per
$10$, otteniamo i chunk necessari per l’addestramento su enwik8. La
funzione di loss (perdita) viene definita in `_make_loss_fn`, che prende
sia i parametri del modello che le sequenze di input e calcola la
probabilità condizionata di predizione per ogni token della sequenza. La
funzione di perdita viene poi usata per aggiornare i parametri del
modello:

<div class="mintedbox">

python loss_fn = \_make_loss_fn(model) grad_fn =
jax.value_and_grad(loss_fn, has_aux=False)

</div>

Vengono definiti inoltre il numero di training steps e la dimensione del
batch, che di default sono rispettivamente $100$ e $128$. In ogni
iterazione del training, si seleziona un batch casuale di sequenze dal
dataset e si calcola la log-loss, con conseguente retropropagazione per
l’aggiornamento dei parametri.

<div class="mintedbox">

python params, opt_state, logs = \_update_parameters( params=params,
opt_state=opt_state, sequences=batch, grad_fn=grad_fn,
optimizer=optimizer, )

</div>

Al termine del processo di training, il modello salva i parametri in un
file chiamato **params.npz**, che verrà utilizzato per il calcolo delle
probabilità, come si vedrà nella successiva sottosezione
<a href="#compressione" data-reference-type="ref"
data-reference="compressione">1.2.2</a>.

### Compressione

La codifica aritmetica è implementata in **arithmetic_coder.py**. Le due
classi principali sono:

- **Encoder**: Codifica una sequenza usando la probabilità condizionale
  di un simbolo.

- **Decoder**: Ricostruisce la sequenza originale compressa.

La funzione **Encoder.encode()**, prende la **PDF** per un simbolo e
comprime in base alla probabilità.

<div class="mintedbox">

python class Encoder(\_CoderBase): def encode(self, pdf: np.ndarray,
symbol: int) -\> None: self.\_process(pdf, symbol)

</div>

**pdf** è l’array di probabilità per ciascun simbolo, **symbol** è il
carattere da codificare, mappato nell?intervallo corretto in base alla
**PDF**.

La codifica aritmetica viene quindi utilizzata data
**language_model.py**, lo script presentato in precedenza che si trova
all’interno del cartella *compressor*. Per prima cosa, vengono
recuperati i parametri di addestramento prodotti durante il training. Il
file **params.npz**, viene caricato attraverso la funzione
*\_retrieve_model_params()*:

<div class="mintedbox">

python def \_retrieve_model_params() -\> hk.Params: try: with
np.load(’params.npz’, allow_pickle=True) as data: return key:
data\[key\].item() for key in data.files except FileNotFoundError as
exc: raise FileNotFoundError( ’You must train a model first, the
parameters file params.npz does not’ ’ exist yet.’ ) from exc

</div>

Dopo aver recuperato il file di parametri, viene generata la funzione di
previsione la quale prende la sequenza in input e calcola la
log-probabilità per ogni simbolo successivo.

<div class="mintedbox">

python def \_retrieve_predict_fn(params: hk.Params) -\>
Callable\[\[np.ndarray\], np.ndarray\]: config =
transformer.TransformerConfig (vocab_size=constants.ALPHABET_SIZE) model
= hk.transform( functools.partial(transformer.transformer_decoder,
config=config) ) return lambda x: model.apply(params, None, x)

</div>

La funzione *compress()* istanzia: il encoder definito nello script del
codificatore aritmetico, la funzione di caricamento dei parametri e la
funzione di previsione. Vengono calcolate le log-probabilità con la
funzione di previsione, passando a quest’ultima l’array di input
converitito in un array di interi. Le log probabilità vengono poi
convertite in probabilità reali. L’encoder viene istanziato e, tramite
un ciclo, scorre le probabilità e i simboli comprimendo con
*encoder.encode()*, facendo sì che poi vengano resituiti i byte
compressi.

<div class="mintedbox">

python def compress( data: bytes, return_num_padded_bits: bool = False,
use_slow_lossless_compression: bool = False, ) -\> bytes \|
tuple\[bytes, int\]: """Compresses the ‘data‘ using arithmetic coding
and a pretrained model.

params = \_retrieve_model_params() predict_fn =
\_retrieve_predict_fn(params)

\# Convert the ‘data‘ into an array of integers (representing the
bytes). sequence_array = np.frombuffer(data, dtype=np.uint8)

log_probs = predict_fn(sequence_array\[None\])\[0, ...\] probs =
np.exp(log_probs)

output = list() encoder = arithmetic_coder.Encoder(
base=constants.ARITHMETIC_CODER_BASE,
precision=constants.ARITHMETIC_CODER_PRECISION, output_fn=output.append,
) for pdf, symbol in zip(probs, sequence_array):
encoder.encode(utils.normalize_pdf_for_arithmetic_coding (pdf), symbol)
encoder.terminate()

compressed_bits = ”.join(map(str, output)) compressed_bytes,
num_padded_bits = utils.bits_to_bytes(compressed_bits)

return compressed_bytes

</div>

La valutazione della compressione avviene con lo script **compress.py**
dove si va a specificare quale compressore utilizzare, i dati da
comprimere e il numero di chunk.

<div class="mintedbox">

python \_COMPRESSOR = flags.DEFINE_enum( ’compressor’, ’gzip’,
compressor.COMPRESS_FN_DICT.keys(), ’Compressor to use.’, ) \_DATASET =
flags.DEFINE_enum( ’dataset’, ’enwik9’,
data_loaders.GET_DATA_GENERATOR_FN_DICT.keys(), ’Dataset to use.’, )
\_NUM_CHUNKS = flags.DEFINE_integer( ’num_chunks’, constants.NUM_CHUNKS,
’Number of chunks.’, )

</div>

Vengono definite all’interno dello script sia la valutazione della
compressione "chunked" che quella "unchunked". La prima è utilizzata per
valutare il tasso di compressione del modello, in quanto non è possibile
passare l’intero dataset al compressore data la limitatezza della
finestra di contesto a 2048 byte del modello, come visto nella
sottosezione <a href="#2048" data-reference-type="ref"
data-reference="2048">[2048]</a> del precedente capitolo. La seconda è
quella utilizzata dai compressori classici che hanno finestre di
contesto molto ampie e possono quindi processare un numero maggiore di
dati. Nulla vieta l’utilizzo della valutazione della compressione
chunked anche per i compressori diversi dal language model, basta
spiecificare il numero di chunk tra i parametri di avvio dello script,
tuttavia tale valutazione limita la reale capacità di compressione per i
compressori classici che sfruttano ridondanze e pattern lungo tutto il
dataset e non solo per una finestra di 2048 byte.

Per comprimere si necessita inoltre di una conversione per tutti i
caratteri non ASCII in quanto essi rappresentano l’input valido per il
transformer. Tale conversione si ottiene azzerando il bit piu
significativo in modo da rientrare nella codifica ascii; il bit "perso"
viene poi recuperato appendendolo alla fine della sequenza compressa,
per tenerne conto nel calcolo del compression rate, che viene stampato
al termine dell’esecuzione dello script.

<div class="mintedbox">

python def evaluate_compressor_chunked( compress_fn:
compressor.Compressor, get_data_generator_fn: Callable\[\[\],
Generator\[bytes, None, None\]\], num_chunks: int,
count_header_only_once: bool = True, mask_fn: Callable\[\[bytes\],
tuple\[bytes, int\]\] \| None = None, use_tqdm: bool = True, ) -\>
tuple\[float, float\]:

num_missed_bits = running_time = raw_length = compressed_length = 0

data_generator = get_data_generator_fn() if use_tqdm: data_generator =
tqdm.tqdm(data_generator, total=num_chunks)

for data in data_generator: if mask_fn is not None: data, missed_bits =
mask_fn(data) num_missed_bits += missed_bits

t0 = time.perf_counter() compressed_data = compress_fn(data) t1 =
time.perf_counter()

running_time += t1 - t0 raw_length += len(data) compressed_length +=
len(compressed_data)

if mask_fn is not None: num_bits = 8 \* num_chunks \*
constants.CHUNK_SIZE_BYTES compressed_length \*= num_bits / (num_bits -
num_missed_bits)

return compressed_length / raw_length, running_time

</div>

## Fase sperimentale

In questa sezione andrò a mostrare i risultati ottenuti, le modifiche
effettuate al codice e le criticità riscontrate.

I primi tentativi di allenamento li ho effettuati con il mio computer
personale. In primis ho clonato la repository e seguito le indicazioni
date dagli sviluppatori, ovvero:

- Installare il package manager Conda.

- Creare ed attivare l’ambiente Conda per poter installare le dipendenze
  necessarie all’esecuzione degli script. Si consigliava inoltre, se si
  fosse avuta la possibilità di una GPU disponibile, di installare JAX
  con supporto CUDA. La prima è una libreria Python progettata per il
  calcolo numerico ad alte performance e machine learing su larga scala.
  CUDA è invece un’architettura hardware per l’elaborazione parallela
  creata da NVIDIA. Purtroppo questa libreria ha dato diversi problemi
  di incompatibilità sia nel mio pc personale che nel cluster. Essa non
  risulta indispensabile per i nostri scopi ma avrebbe velocizzato i
  calcoli rendendo il training più rapido.

- Eseguire lo script di training e successivamente di compressione come
  segue:

  \|python train.py \#script per il training\|

Come era facilmente prevedibile, l’hardware del mio pc non è risultato
sufficiente a gestire una tale computazione. Sotto consiglio dei miei
relatori, ho fatto richiesta per accedere al super computer
universitario, Caliban.

## Sfide e risultati ottenuti

La prima cosa che ho fatto è stata addestrare il modello, così come mi
veniva fornito dalla repository. Il training era veloce però le capacità
di compressione lasciavano a desiderare. Ciò, dopo riflessione, era
dovuto al fatto che probabilmente gli iperparametri non fossero quelli
corretti. Difatti, poi, sono riuscito a scoprire che nella sezione
issues della repository vi era la configurazione corretta per il
Transformer da $200$k parametri, fornita dagli stessi sviluppatori:

<div class="mintedbox">

python "training_steps": "1000000", "batch_size": "32", "seq_length":
"2048", "embedding_dim": "64", "num_heads": "4", "num_layers": "4",
"positional_encodings": "ROTARY",

</div>

Ho per cui proceduto a cambiare gli iperparametri di deafult, senza
modificare il positional_encoding, che nello script fornito è quello
sinusoidal, descritto anche nel paper originale "Attention is all you
need" . Ho dunque ripetuto il training col nuovo script con "sinusoidal"
e non "ROTARY". Al termine di esso la "loss" (o anche perdita) perdita
ammontava a circa $2552$ con un tempo di addestramento di oltre $122$
ore, con convergenza alla loss finale ottenuta intorno alle $100$ ore.
Ho proseguito quindi alla compressione di enwik9, ottenendo il risultato
mostrato nella figura <a href="#fig:log" data-reference-type="ref"
data-reference="fig:log">1.1</a>.

<figure id="fig:log">
<img src="immagini/log enwik9 top.png" style="width:100.0%" />
<figcaption>Tasso di compressione di enwik9</figcaption>
</figure>

Ovvero il modello comprime 3.85 chunks al secondo, ottenendo in 35 ore
circa il tasso di compressione chunked del 23.5 per cento. Per cui un
tasso di compressione ben al di sotto (e dunque migliore) del 30.9
percento riportato dallo studio. Ho contattato quindi gli sviluppatori
su GitHub, aprendo una issues proprio su GitHub. Gli sviluppatori stessi
di Google DeepMind mi hanno risposto e mi hanno confermato la validità
del risultato ottenuto. Hanno anche aggiunto che e il fatto che il
discostamento rispetto al compression rate dello studio potrebbe essere
dovuto alla configurazione del Transformer, la quale non risulta essere
esattamente la stessa per il diverso positional_encoding. Dunque
sembrerebbe che, almeno per il transformer relativamente piccolo con
200K parametri che ho allenato, il positional_encoding "sinusoidal"
dell’articolo originale "attention is all you need" e non "ROTARY"
sembra essere molto più efficiente anche se, presumibilmente ci saranno
altri motivi validi per usare "ROTARY" su transformers più grandi
(velocità e/o efficienza in compressione).

### Criticità incontrata

L’addestramento di un Transformer di grandi dimensioni su un dataset
altrettanto ampio ha presentato diverse difficoltà, principalmente
legate al tempo di addestramento molto lungo data una discrepanza tra la
versione del driver NVIDIA (CUDA 12.4) e il compilatore PTX (CUDA 12.6).
Questo ha causato la disabilitazione della compilazione parallela da
parte di XLA, rallentando l’addestramento. Tale errore è probabilmente
dovuto ai problemi di compatibilità di JAX descritti in precedenza.

## Compressione su altri dataset

Ho deciso poi, di testare le capacità di compressione su altri
benchmarks. In particolare il Calgary Corpus e il Large Canterbury
Corpus. Questi, come vedremo nelle sezioni successive, hanno dimensione
minore rispetto ad enwik8. Si è pensato allora di addestrare ulteriori
due Transformer su due versioni minori di enwik9, nello specifico enwik7
e enwik6, per osservare il tasso di compressione che si otteneva con
dataset minori. Per fare ciò ho modificato l’iteratore del dataset nel
file **train.py** dividendo rispettivamente per 100 e 1000 i chunks di
enwik9, ottenendo in tal modo i dataset desiderati. Questi, avendo
dimensioni esigue, hanno bisogno di un *fine-tuning* specifico per poter
adattare il modello a processare una minor quantità di dati; ho quindi
ridotto il numero di layer e head a $1$ e i training steps a $100000$.

### Compressione del Corpus di Calgary

Il Corpus di Calgary è una raccolta di 14 file per un totale di
3,141,622 bytes, per lo più testuali e un’immagine bitmap. Per includere
il nuovo dataset tra quelli disponibili, ho modificato il codice di
**data_loader.py**, aggiungendo un iteratore per esso. Essendo il Corpus
Clagary composto da file diversi, l’iteratore non poteva leggerli. Per
risolvere il problema ho creato un array che contiene tutti i files e,
uno per volta, mediante un ciclo, venivano estratti i chunks e appesi in
un altro vettore. Ho successivamente calcolato il numero di chunks da
2048 byte in cui suddividere il dataset e tale costante è stata poi
aggiunta in **costants.py**. Provando a comprimere non ottenevo
risultati soddisfacenti. Rianalizzando il dataset e ho concluso che i
pessimi risultati erano dovuti al fatto che la maggior parte dei file
testuali di cui è composto non sono codificati in ASCII, per cui
necessitavano nell’atto di compressione, per tutta la loro dimensione,
della conversione in questa codifica, facendo così aumentare di molto il
compression rate. Ho provato quindi a comprimere solamente i 2 maggiori
file codificati in ASCII, ovvero: BOOK1 e BOOK2, per un totale di
$1379627$ bytes. Dopo aver modificato l’iteratore e le costanti, potevo
procedere alla compressione, tramite cui ho ottenuto i seguenti
risultati:

<div id="tab:compression_results">

| **Dataset** |  **Compressore**   | **Compression Rate (%)** |
|:-----------:|:------------------:|:------------------------:|
| BOOK1+BOOK2 |        gzip        |           37.5           |
| BOOK1+BOOK2 | Transformer enwik8 |         **35.8**         |
| BOOK1+BOOK2 | Transformer enwik7 |           45.8           |
| BOOK1+BOOK2 | Transformer enwik6 |           63.8           |

Risultati della compressione sul Calgary Corpus

</div>

### Compressione del Large Corpus

Il Large Corpus è una versione del Canterbury Corpus, una collezione di
$11$ file, che può essere considerato il successore del Calgary Corpus.
La versione da me utilizzata contiene invece $3$ file, tra questi vi si
trova BIBLE.txt, una versione della Bibbia di KingJames, il maggiore in
dimensione nella raccolta, con $4047392$ bytes. Ovviamente come per il
Corpus Calgary, ho dovuto aggiungere un nuovo iteratore in
**data_loader.py**, e le costanti dei chunk in cui è suddiviso il
dataset. I risultati ottenuti sono :

<div id="tab:compression_results_canterbury">

| **Dataset** |  **Compressore**   | **Compression Rate (%)** |
|:-----------:|:------------------:|:------------------------:|
|    BIBLE    |        gzip        |           29.1           |
|    BIBLE    | Transformer enwik8 |         **27.1**         |
|    BIBLE    | Transformer enwik7 |           36.2           |
|    BIBLE    | Transformer enwik6 |           54.8           |

Risultati della compressione sul Canterbury Corpus

</div>

## Analisi dei risultati

In entrambi i dataset, il migliore tasso di compressione, indicato in
grasseto nelle tabelle, è ottenuto dal Transformer allenato su enwik8.
Questo risultato è in linea con ciò che era stato ottenuto in precedenza
su enwik9. Gzip, invece, si attesta migliore degli altri due
Transformer.

## Conclusione e sviluppi futuri

Le principale sfida di questo esperimento è stata quella di riuscire
trovare un giusto equilibrio tra dimensione del dataset e configurazione
del Transformer e dei giusti parametri. Ad esempio abbiamo notato che
utilizzare "SINUSOIDAL" piuttosto che "ROTARY" per il
positional_encoding su modelli più piccoli porta ad un vantaggio
sensibile. Sarebbe interessante esaminare come diverse tecniche di
fine-tuning cambino le capacità di compressione dei modelli, esplorando
anche nuovi dataset di addestramento, e vedere quanto impattano non solo
sul tasso di compressione ma anche sul tempo di compressione e su quanto
il modello riesca a generalizzarsi anche su tipi di dati differenti da
quelli di addestramento. La GPU utilizzata è una NVIDIA A100 80GB, che
nella scheda tecnica riporta un consumo massimo di $300$W ora, nello
specifico nel nodo del cluster dove si è effettuato l’ allenamento viene
fornita circa la metà della sua memoria e il numero degli streaming
multiprocesssor è pari a $14$ quindi circa $\frac{3}{7}$ del totale. La
nostra stima dei consumi energetici per l’addestramento del Transformer
200k su enwik8 per $122.5$ ore è di circa "$12$ KWh". Per il nostro
modello i costi sono, quindi, molto contenuti nonostante anche
l’inefficienza riportata in precedenza che ha allungato il tempo di
addestramento. Discorso differente è per i modelli di grandi dimensioni,
secondo l’ Artificial Intelligence Index Report 2024 , i costi stimati
per allenare GPT4 si attestano intorno agli $80$ milioni di dollari e
per Gemini Ultra si superano i $190$. Per i modelli più recenti invece
ancora non sono reperibili dati al riguardo. Inoltre le grandi aziende
tecnologiche che producono LLM, hanno quasi sempre hardware dedicati e
chip di loro produzione specifici per questo tipo di applicazione,
ottimizzati per l’alto carico computazionale necessario ad allenare
questi modelli. I nostri risultati seppur buoni, hanno certamente un
margine di miglioramento, che in futuro cercheremo di ottenere, seguendo
il più accuratamente possibile le principali innovazioni che la ricerca
mette a disposizione in questa affascinante branca dell’informatica.
