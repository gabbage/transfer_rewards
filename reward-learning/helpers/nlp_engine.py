import logging

import torch
from ignite.engine import Engine
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from helpers.loss import SimpleMarginRankingLoss
from models.model_api import ModelAPI


def pick(batch, fields, as_list=False):
    result = {}

    for field_name in list(fields.keys()):
        result.update({field_name: batch[field_name]})

    if as_list:
        result = list(result.values())

    return result


def is_nn(phi):
    return phi and isinstance(phi, Module)


def is_trainable_nn(phi):
    return is_nn(phi) and phi.trainable


def prepare_batch(batch, prep_targets_fn=None):
    # Pick input fields and target field from patch
    input_fields = pick(batch, batch['input_fields'])
    target_fields = pick(batch, batch['target_fields'], as_list=True)

    # Right now only one target field is allowed
    assert len(target_fields) == 1, "Only one target field is allowed!"

    # Remove all fields that are not actual input fields and pick first target_field
    for field_name, field in batch['input_fields'].items():
        if not field.actual_input:
            del input_fields[field_name]

    x = input_fields
    y = target_fields[0] if prep_targets_fn is None else prep_targets_fn(target_fields[0])

    return x, y


def create_trainer(model, optimizer, criterion, device=torch.device("cpu"), metrics=None, phi=None, train_phi=False,
                   clip_grad=2, max_grad=1e2, fp16=False):
    prep_targets_fn = None

    if isinstance(model, ModelAPI):
        prep_targets_fn = model.prep_targets_fn

    if device:
        model.to(device)

        if fp16:
            model.half()

        if is_nn(phi):
            phi.to(device)

            if fp16:
                phi.half()

    def train_update(engine, batch):
        # Bring model in training mode
        model.train()

        if is_trainable_nn(phi) and train_phi:
            phi.train()

        # Reset the current gradient
        optimizer.zero_grad()

        # Pick input fields and target field from patch
        x, y = prepare_batch(batch, prep_targets_fn)

        # Forward pass: feed the input through the model
        if is_trainable_nn(phi) and not train_phi:
            with torch.no_grad():
                x = phi(x)

            with torch.enable_grad():
                y_pred = model(x)
        else:
            y_pred = model(x) if phi is None else model(phi(x))

        # Calculate the loss and do the backward pass
        if isinstance(criterion, SimpleMarginRankingLoss) and criterion.margin == 'max':
            if 'swap_count' not in batch and 'overall' not in batch:
                raise ValueError("In batch there must be 'swap_count' or 'overall' values for maximum margin loss!")

            relevance_key = 'swap_count' if 'swap_count' in batch else 'overall'
            relevance_scores = batch[relevance_key]

            if len(relevance_scores) != 2:
                raise ValueError("There must be exactly two relevance scores!")

            loss = criterion(y_pred, y, margin=torch.abs(relevance_scores[0] - relevance_scores[1]))
        else:
            loss = criterion(y_pred, y)

        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # Update the weights
        optimizer.step()

        return loss.item()

    engine = Engine(train_update)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    return engine


def create_evaluator(model, metrics=None, phi=None):
    prep_targets_fn = None

    if isinstance(model, ModelAPI):
        prep_targets_fn = model.prep_targets_fn

    def evaluate_update(engine, batch):
        model.eval()

        if is_nn(phi):
            phi.eval()

        with torch.no_grad():
            x, y = prepare_batch(batch, prep_targets_fn)

            # Forward pass: feed the input through the model
            y_pred = model(x) if phi is None else model(phi(x))

            return y_pred, y

    engine = Engine(evaluate_update)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    return engine
