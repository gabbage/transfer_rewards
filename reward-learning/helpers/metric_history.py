import logging

import numpy as np
from ignite.engine import Events


class MetricHistory(object):
    def __init__(self):
        self.values = {}

    def _check(self, engine, metric_name):
        engine_id = id(engine)

        if engine_id not in self.values:
            self.values.update({engine_id: {}})

        if metric_name not in self.values[engine_id]:
            self.values[engine_id].update({metric_name: []})

    def _reduce(self, engine, metric_name, reduce_fn):
        self._check(engine, metric_name)
        tmp_metric_name = metric_name + "_tmp"
        engine_id = id(engine)

        if tmp_metric_name in self.values[engine_id]:
            self.values[engine_id][metric_name].append(reduce_fn(self.values[engine_id][tmp_metric_name]))
            self.values[engine_id][tmp_metric_name] = []
        else:
            logging.warning("Metric '{}' was not found in self.values!".format(tmp_metric_name))

    def _store(self, engine, metric_name, output_transform_fn=None):
        self._check(engine, metric_name)
        engine_id = id(engine)

        if output_transform_fn:
            self.values[engine_id][metric_name].append(output_transform_fn(engine.state.output))
        else:
            if metric_name in engine.state.metrics:
                self.values[engine_id][metric_name].append(engine.state.metrics[metric_name])
            else:
                logging.warning("Metric '{}' was not found in engine.state.metrics!".format(metric_name))

    @staticmethod
    def output(name):
        return tuple((name, lambda x: x))

    def attach(self, engine, metrics, store_event=Events.EPOCH_COMPLETED, reduce_event=None, reduce_fn=np.mean):
        if metrics and not isinstance(metrics, list):
            raise TypeError("metrics should be a list, got {} instead".format(type(metrics)))

        for metric in metrics:
            if isinstance(metric, tuple) and len(metric) == 2:
                metric_name = metric[0]
                output_transform_fn = metric[1]
            elif isinstance(metric, str):
                metric_name = metric
                output_transform_fn = None
            else:
                continue

            if reduce_event:
                engine.add_event_handler(store_event, self._store, metric_name + "_tmp", output_transform_fn)
                engine.add_event_handler(reduce_event, self._reduce, metric_name, reduce_fn)
            else:
                engine.add_event_handler(store_event, self._store, metric_name, output_transform_fn)
