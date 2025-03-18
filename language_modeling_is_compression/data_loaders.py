# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements data loaders."""

import audioop
from collections.abc import Iterator
import itertools
import os.path
import random
import urllib.request
import zipfile

import numpy as np
import tensorflow_datasets as tfds

from language_modeling_is_compression import constants


def _get_librispeech_dataset():
  return tfds.load('librispeech', split='train_clean100')


def _get_imagenet_dataset():
  return tfds.load('imagenet2012')['full']




def _extract_audio_patches(sample: bytes) -> Iterator[bytes]:
  patches = np.array_split(
      np.frombuffer(sample, dtype=np.uint8),
      range(
          constants.CHUNK_SIZE_BYTES,
          len(sample),
          constants.CHUNK_SIZE_BYTES,
      ),
  )
  if len(patches[-1]) != constants.CHUNK_SIZE_BYTES:
    patches.pop()
  return map(lambda x: x.tobytes(), patches)


def get_librispeech_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for librispeech data."""
  # Convert samples from 16 bit to 8 bit (i.e., changing from two channels to
  # one channel with `lin2lin`), adding 128 since 16 bit is signed (i.e., adding
  # 128 using `bias`).
  librispeech_dataset = map(
      lambda x: audioop.bias(audioop.lin2lin(x['speech'], 2, 1), 1, 128),
      _get_librispeech_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in librispeech_dataset:
    for patch in _extract_audio_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1


def get_random_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for random data."""
  for _ in range(num_chunks):
    yield random.randbytes(constants.CHUNK_SIZE_BYTES)


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
  return np.mean(image, axis=-1).astype(np.uint8)


def _extract_image_patches(image: np.ndarray) -> Iterator[bytes]:
  h, w = constants.CHUNK_SHAPE_2D
  height, width = image.shape

  for row, col in itertools.product(range(height // h), range(width // w)):
    yield image[row * h : (row + 1) * h, col * w : (col + 1) * w].tobytes()


def get_imagenet_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns a iterator for imagenet data."""
  imagenet_dataset = map(
      lambda x: _rgb_to_grayscale(x['image']),
      _get_imagenet_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in imagenet_dataset:
    for patch in _extract_image_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1


def get_enwik9_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
  """Returns an iterator for enwik9 data."""
  if not os.path.exists('enwik9'):
    # Downloading and extracting the dataset.
    urllib.request.urlretrieve(
        'https://mattmahoney.net/dc/enwik9.zip',
        'enwik9.zip',
    )
    with zipfile.ZipFile('enwik9.zip', 'r') as zip_ref:
      zip_ref.extract('enwik9')

  all_chunks = []
  with open('enwik9', 'rb') as file:
    for _ in range(num_chunks):
      all_chunks.append(file.read(sequence_length))
  return iter(all_chunks)

def get_calgary_corpus_iterator(
    num_chunks: int = constants.NUM_CHUNKS_CALGARY,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
    """Restituisce un iteratore per il Calgary Corpus, concatenando tutti i file e suddividendo in chunk da sequence_length byte."""
    
    
    calgary_corpus_url = "http://www.data-compression.info/files/corpora/CalgaryCorpus.zip"
    corpus_dir = 'calgary_corpus'

   
    if not os.path.exists(corpus_dir):
        urllib.request.urlretrieve(calgary_corpus_url, 'CalgaryCorpus.zip')
        with zipfile.ZipFile('CalgaryCorpus.zip', 'r') as zip_ref:
            zip_ref.extractall(corpus_dir)

    
    calgary_corpus_files = [
        'book1', 'book2',
    ]

    # Concatenare tutti i file in un unico flusso di byte
    full_data = bytearray()  # Utilizziamo un bytearray per concatenare i file
    for file_name in calgary_corpus_files:
        file_path = os.path.join(corpus_dir, file_name)
        with open(file_path, 'rb') as file:
            full_data.extend(file.read())  # Aggiungi i byte di ogni file al bytearray

    
    all_chunks = []
    total_size = len(full_data)
    for i in range(0, total_size, sequence_length):
        all_chunks.append(full_data[i:i+sequence_length])
        if len(all_chunks) >= num_chunks:  
            break

    return iter(all_chunks)


def get_bible_iterator(
    num_chunks: int = constants.NUM_CHUNKS_BIBLE,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
    """Restituisce un iteratore per il file Bible del Canterbury Corpus."""
    canterbury_corpus_url = "https://corpus.canterbury.ac.nz/resources/large.zip"
    corpus_dir = 'canterbury_corpus'
    bible_file = 'bible.txt'

    # Scarica e estrai il corpus se non è già presente
    if not os.path.exists(corpus_dir):
        urllib.request.urlretrieve(canterbury_corpus_url, 'large.zip')
        with zipfile.ZipFile('large.zip', 'r') as zip_ref:
            zip_ref.extractall(corpus_dir)

    # Verifica che il file Bible sia presente
    bible_path = os.path.join(corpus_dir, bible_file)
    if not os.path.exists(bible_path):
        raise FileNotFoundError(f"Il file {bible_file} non è stato trovato nella directory {corpus_dir}")

    # Lettura e divisione del file in chunk
    all_chunks = []
    with open(bible_path, 'rb') as file:
        while len(all_chunks) < num_chunks:
            chunk = file.read(sequence_length)
            if not chunk:  # Se il file finisce prima di riempire tutti i chunk richiesti
                break
            all_chunks.append(chunk)

    return iter(all_chunks)








GET_DATA_GENERATOR_FN_DICT = {
    'enwik9': get_enwik9_iterator,
    'imagenet': get_imagenet_iterator,
    'librispeech': get_librispeech_iterator,
    'random': get_random_iterator,
    'calgary_corpus': get_calgary_corpus_iterator, 
    'bible' : get_bible_iterator 

}

