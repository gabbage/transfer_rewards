import logging

import numpy as np
import visdom


class VisdomPlot(object):
    def __init__(self, env, win, k_fold=None):
        super(VisdomPlot, self).__init__()
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False)
        self.env = env
        self.enabled = self.vis.check_connection()
        self.k_fold = '' if k_fold is None else '_{}/{}'.format(k_fold[0], k_fold[1])
        self.win = win + self.k_fold
        self.step = 0
        self.metric_history = None

        if not self.enabled:
            logging.warning('Visdom server not running. Please run python -m visdom.server')

    def plot(self, engine=None):
        values = []
        labels = []

        if self.metric_history is None:
            return

        for metrics_dict in self.metric_history.values.values():
            for metric_name, metric_values in metrics_dict.items():
                logging.debug("Metric '{}' has values: {}".format(metric_name, metric_values))

                if not metric_name.endswith('_tmp') and len(metric_values) > 0:
                    values.append(metric_values[-1])
                    labels.append(metric_name)

        # Sort according to labels to avoid problems with mixed values
        values = [x for _, x in sorted(zip(labels, values))]
        labels = sorted(labels)

        self.step += 1

        y = np.array([values])
        x = np.ones_like(y) * self.step

        self.vis.line(X=x, Y=y, win=self.win, update="append", opts={"title": self.win, "legend": labels,
                                                                     "xlabel": "epochs", "ylabel": "value"})

        # Save the visdom plot environment after drawing
        self.vis.save([self.env])
