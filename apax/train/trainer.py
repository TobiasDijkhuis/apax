import logging
import time
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from clu.metrics import Collection as MetricsCollection
from jax import tree_util
from jax.experimental import mesh_utils
from flax import nnx
from jax.sharding import PositionalSharding
from tqdm import trange

from apax.data.input_pipeline import InMemoryDataset
from apax.train.checkpoints import load_state, TrainState
from apax.train.parameters import EMAParameters
from apax.train.callbacks import CallbackCollection

log = logging.getLogger(__name__)


class EarlyStop(Exception):
    pass


def fit(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_ds: InMemoryDataset,
    loss_fn,
    metrics: MetricsCollection,
    callbacks: CallbackCollection,
    n_epochs: int,
    ckpt_dir,
    ckpt_interval: int = 1,
    val_ds: Optional[InMemoryDataset] = None,
    patience: Optional[int] = None,
    patience_min_delta: float = 0.0,
    disable_pbar: bool = False,
    disable_batch_pbar: bool = True,
    is_ensemble=False,
    data_parallel=True,
    ema_handler: Optional[EMAParameters] = None,
    rngs: Optional[nnx.Rngs] = None,  # TODO: Do I ever need this?
):
    """
    Trains the model using the provided training dataset.

    Parameters
    ----------
    state : TrainState
        The initial state of the model.
    train_ds : InMemoryDataset
        The training dataset.
    loss_fn :
        The loss function to be minimized.
    metrics : MetricsCollection
        Collection of metrics to evaluate during training.
    callbacks : CallbackCollection
        List of callback functions to be executed during training.
    n_epochs : int
        Number of epochs for training.
    ckpt_dir:
        Directory to save checkpoints.
    ckpt_interval : int, default = 1
        Interval for saving checkpoints.
    val_ds : InMemoryDataset, default = None
        Validation dataset.
    patience : int, default = None
        Patience for early stopping.
    disable_pbar : bool, default = False
        Whether to disable progress bar for epochs..
    disable_batch_pbar : bool, default = True
        Whether to disable progress bar for batches.
    is_ensemble : bool, default = False
        Whether the model is an ensemble.
    data_parallel : bool, default = True
        Whether to use data parallelism.
    rngs : nnx.Rngs, default = None
    """

    log.info("Beginning Training")
    callbacks.on_train_begin()

    latest_dir = ckpt_dir / "latest"
    best_dir = ckpt_dir / "best"

    options = ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1)
    train_step, val_step = make_step_fns(
        loss_fn, metrics, model=model, rngs=rngs, is_ensemble=is_ensemble
    )

    state, start_epoch = load_state(state, latest_dir)
    if start_epoch >= n_epochs:
        print(
            f"Training has already completed ({start_epoch} >= {n_epochs}). Nothing to be done"
        )
        return

    devices = len(jax.devices())
    if devices > 1 and data_parallel:
        sharding = PositionalSharding(mesh_utils.create_device_mesh((devices,)))
        state = jax.device_put(state, sharding.replicate())
    else:
        sharding = None

    train_steps_per_epoch = train_ds.steps_per_epoch()
    batch_train_ds = train_ds.shuffle_and_batch(sharding)

    if val_ds is not None:
        val_steps_per_epoch = val_ds.steps_per_epoch()
        batch_val_ds = val_ds.batch(sharding)

    best_loss = np.inf
    early_stopping_counter = 0
    epoch_loss = {}
    epoch_pbar = trange(
        start_epoch, n_epochs, desc="Epochs", ncols=100, disable=disable_pbar, leave=True
    )
    try:
        with (
            ocp.CheckpointManager(
                latest_dir.resolve(),
                options=options,
            ) as latest_ckpt_manager,
            ocp.CheckpointManager(
                best_dir.resolve(),
                options=options,
            ) as best_ckpt_manager,
        ):
            for epoch in range(start_epoch, n_epochs):
                epoch_start_time = time.time()
                callbacks.on_epoch_begin(epoch=epoch + 1)

                if ema_handler:
                    ema_handler.update(model.params, epoch)

                epoch_loss.update({"train_loss": 0.0})
                train_batch_metrics = metrics.empty()

                batch_pbar = trange(
                    0,
                    train_steps_per_epoch,
                    desc="Batches",
                    ncols=100,
                    mininterval=1.0,
                    disable=disable_batch_pbar,
                    leave=False,
                )

                model.train()  # Set model to train (not deterministic)
                for batch_idx in range(train_steps_per_epoch):
                    callbacks.on_train_batch_begin(batch=batch_idx)

                    batch = next(batch_train_ds)
                    train_batch_metrics, batch_loss = train_step(
                        model,
                        optimizer,
                        train_batch_metrics,
                        batch,
                    )

                    epoch_loss["train_loss"] += jnp.mean(batch_loss)
                    callbacks.on_train_batch_end(batch=batch_idx)
                    batch_pbar.update()

                epoch_loss["train_loss"] /= train_steps_per_epoch
                epoch_loss["train_loss"] = float(epoch_loss["train_loss"])

                epoch_metrics = {
                    f"train_{key}": float(val)
                    for key, val in train_batch_metrics.compute().items()
                }

                if ema_handler:
                    ema_handler.update(model.params, epoch)
                    val_params = ema_handler.ema_params
                else:
                    val_params = model.params

                if val_ds is not None:
                    model.eval()  # Set model to eval (deterministic)
                    epoch_loss.update({"val_loss": 0.0})
                    val_batch_metrics = metrics.empty()

                    batch_pbar = trange(
                        0,
                        val_steps_per_epoch,
                        desc="Batches",
                        ncols=100,
                        mininterval=1.0,
                        disable=disable_batch_pbar,
                        leave=False,
                    )
                    for batch_idx in range(val_steps_per_epoch):
                        batch = next(batch_val_ds)

                        batch_loss, val_batch_metrics = val_step(
                            model, batch, val_batch_metrics
                        )
                        epoch_loss["val_loss"] += batch_loss
                        batch_pbar.update()

                    epoch_loss["val_loss"] /= val_steps_per_epoch
                    epoch_loss["val_loss"] = float(epoch_loss["val_loss"])

                    epoch_metrics.update(
                        {
                            f"val_{key}": float(val)
                            for key, val in val_batch_metrics.compute().items()
                        }
                    )

                epoch_metrics.update({**epoch_loss})
                epoch_end_time = time.time()
                epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})

                ckpt = {"model": model, "epoch": epoch}
                if epoch % ckpt_interval == 0:
                    latest_ckpt_manager.save(epoch, args=ocp.args.StandardSave(ckpt))

                if epoch_metrics["val_loss"] < best_loss:
                    best_ckpt_manager.save(epoch, args=ocp.args.StandardSave(ckpt))
                    if abs(epoch_metrics["val_loss"] - best_loss) < patience_min_delta:
                        early_stopping_counter += 1
                    else:
                        early_stopping_counter = 0

                    best_loss = epoch_metrics["val_loss"]
                else:
                    early_stopping_counter += 1

                callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)

                epoch_pbar.set_postfix(val_loss=epoch_metrics["val_loss"])
                epoch_pbar.update()

                if patience is not None and early_stopping_counter >= patience:
                    raise EarlyStop()
    except EarlyStop:
        log.info(
            f"Early stopping patience exceeded. Stopping training after {epoch} epochs."
        )

    epoch_pbar.close()
    callbacks.on_train_end()

    train_ds.cleanup()
    if val_ds:
        val_ds.cleanup()


def calc_loss(
    model: nnx.Module, rngs: nnx.Rngs, inputs, labels, loss_fn: Callable
) -> tuple[float, jax.Array]:
    R, Z, idx, box, offsets = (
        inputs["positions"],
        inputs["numbers"],
        inputs["idx"],
        inputs["box"],
        inputs["offsets"],
    )
    predictions = model(R, Z, idx, box, offsets, rngs)
    loss = loss_fn(inputs, predictions, labels)
    return loss, predictions


def make_ensemble_update(update_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0, 0))

    def ensemble_update_fn(model: nnx.Module, optimizer: nnx.Optimizer, inputs, labels):
        loss, predictions = v_update_fn(model, inputs, labels)

        mean_predictions = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        # Should we add std to predictions?
        return mean_loss, mean_predictions

    return ensemble_update_fn


def make_ensemble_eval(eval_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0))

    def ensemble_eval_fn(model: nnx.Module, inputs, labels):
        loss, predictions = eval_fn(model, inputs, labels)

        mean_predictions = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        return mean_loss, mean_predictions

    return ensemble_eval_fn


def make_step_fns(
    loss_fn: Callable,
    metrics: MetricsCollection,
    model: nnx.Module,
    rngs: nnx.Rngs,
    is_ensemble: bool,
):
    loss_calculator = partial(calc_loss, rngs=rngs, loss_fn=loss_fn)
    grad_fn = nnx.value_and_grad(loss_calculator, has_aux=True)

    def update_step(model: nnx.Module, optimizer: nnx.Optimizer, inputs, labels):
        (loss, predictions), grads = grad_fn(model, inputs, labels)
        optimizer.update(model, grads)
        return loss, predictions

    if is_ensemble:
        update_fn = make_ensemble_update(update_step)
        eval_fn = make_ensemble_eval(loss_calculator)
    else:
        update_fn = update_step
        eval_fn = loss_calculator

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch_metrics: MetricsCollection,
        batch,
    ) -> tuple[MetricsCollection, float]:
        inputs, labels = batch
        loss, predictions = update_fn(model, optimizer, inputs, labels)

        new_batch_metrics = metrics.single_from_model_output(
            inputs=inputs, label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)

        return loss, batch_metrics

    @nnx.jit
    def val_step(
        model: nnx.Module, batch_metrics: MetricsCollection, batch
    ) -> tuple[float, MetricsCollection]:
        inputs, labels = batch
        loss, predictions = eval_fn(model, inputs, labels)

        new_batch_metrics = metrics.single_from_model_output(
            inputs=inputs, label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return loss, batch_metrics

    return train_step, val_step
