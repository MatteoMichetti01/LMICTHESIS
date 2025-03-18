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

"""Trains a language model on the Enwik8 dataset."""

import functools
import random
import os  # Import per creare cartella dei checkpoint
from typing import Any

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

from language_modeling_is_compression import constants
from language_modeling_is_compression import data_loaders
from language_modeling_is_compression import transformer


def _to_marginals(
    predictions: jax.Array,
    sequences: jax.Array,
) -> jax.Array:
  """Converts a conditional array to a marginals array."""
  true_predictions = jnp.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).
  return jnp.sum(true_predictions, axis=1)  # Shape (B,).


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the loss function for update_parameters."""

  def loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The input of sequences to evaluate. See neural_predictors.py.
    """
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    marginals = _to_marginals(conditionals, sequences)
    return -jnp.mean(marginals)

  return loss_fn


@functools.partial(
    jax.jit, static_argnames=('optimizer', 'grad_fn', 'normalize_gradients')
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    opt_state: The optimizer state.
    sequences: The input of sequences to evaluate. See base_predictor.py.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optax optimizer.
    normalize_gradients: Whether to divide the gradients by the length of the
      sequences, or keep them as is. Using this option guarantees to have the
      same scale across various sequence lengths, and therefore tasks.
  """
  loss, grad = grad_fn(params, sequences)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return new_params, new_opt_state, log_dict


def train_transformer_decoder(
    training_steps: int,
    log_every: int,
    checkpoint_every: int = 100000,
    checkpoint_path: str = None,
    batch_size: int = 32,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
) -> tuple[hk.Params, float]:
    """Trains a language model on Enwik8 data, with checkpointing functionality.

    Args:
        training_steps: Number of batches to train on.
        log_every: How often to log the loss. If negative or 0, no log at all.
        checkpoint_every: Save the parameters every this many steps.
        checkpoint_path: Path to a checkpoint file to load the initial parameters.
        batch_size: The number of sequences in a batch.
        sequence_length: The length of the sequences to train on, in number of ASCII
          characters.
        use_tqdm: Whether to use a progress bar or not.

    Returns:
        The final loss, and final parameters.
    """
    config = transformer.TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )

    data_generator = data_loaders.get_enwik9_iterator(
        num_chunks=constants.NUM_CHUNKS // 10,
        sequence_length=sequence_length,
    )
    dataset = list(data_generator)

    def fetch_random_batch() -> np.ndarray:
        batch_list = random.choices(dataset, k=batch_size)
        batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
        return np.array(batch_list, dtype=np.uint8)

    # Initialize parameters
    dummy_batch = fetch_random_batch()
    rng = jax.random.PRNGKey(0)
    if checkpoint_path:
        logging.info('Loading parameters from checkpoint: %s', checkpoint_path)
        loaded_params = np.load(checkpoint_path, allow_pickle=True)
        params = {k: loaded_params[k].item() for k in loaded_params.files}
    else:
        params = model.init(rng, dummy_batch)

    # Make gradient function
    loss_fn = _make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

    # Make optimizer, to apply the gradients
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    logging.info('Initialization done, starting training...')
    last_loss = 0.0

   
    checkpoint_dir = 'checkpointenwik7_TRUE'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for step in tqdm.trange(training_steps, disable=not use_tqdm):
        batch = fetch_random_batch()

        params, opt_state, logs = _update_parameters(
            params=params,
            opt_state=opt_state,
            sequences=batch,
            grad_fn=grad_fn,
            optimizer=optimizer,
        )

        if log_every > 0 and step % log_every == 0:
            logging.info(
                'Step %d, Loss %f, Grad norm %f',
                step,
                logs['loss'],
                logs['grad_norm_unclipped'],
            )
        last_loss = logs['loss']

        # Save checkpoint every `checkpoint_every` steps
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            checkpoint_filename = f'{checkpoint_dir}/params_step_{step}.npz'  # Salvataggio in "checkpointenwik6"
            np.savez(checkpoint_filename, **params)
            logging.info('Checkpoint saved to %s', checkpoint_filename)

    return params, last_loss


def main(argv) -> None:
    """Trains a language model and saves the parameters to a JSON file."""
    checkpoint_path = None
    if len(argv) > 1:
        checkpoint_path = argv[1]  # Use the first argument as the checkpoint path

    params, loss = train_transformer_decoder(
        training_steps=100000,
        log_every=1000,
        checkpoint_every=1000,  # Save every 100000 steps

        checkpoint_path=checkpoint_path,  # Load checkpoint if provided
        sequence_length=constants.CHUNK_SIZE_BYTES,
    )
    logging.info('Final loss: %f', loss)

    np.savez('params.npz', **params)
    logging.info('Final parameters saved in file params.npz')



if __name__ == '__main__':
    app.run(main)
