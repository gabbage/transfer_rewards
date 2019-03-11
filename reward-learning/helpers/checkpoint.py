import glob
import logging
import os

import numpy as np
import torch


def save_weights(component, session_dir, epoch, component_key='model'):
    if not os.path.isdir(session_dir):
        os.makedirs(session_dir)

    weight_file = os.path.join(session_dir, '{}_{:02d}.pt'.format(component_key, epoch))
    torch.save(component.state_dict(), weight_file)
    logging.info('Saved {} weights to file {}.'.format(component_key, weight_file))


def restore_weights(component, session_dir, epoch, component_key='model', device=torch.device('cpu'),
                    drop_embeddings=True):
    if not os.path.isdir(session_dir):
        os.makedirs(session_dir)

    if epoch is not None and epoch < 0:
        assert os.path.isfile(os.path.join(session_dir, 'restore.txt')), \
            'The "restore.txt" file does not exist!'

        for line in open(os.path.join(session_dir, 'restore.txt'), 'r'):
            assert line.strip() is not '', 'The file "restore.txt" has to contain the epoch as integer!'
            epoch = int(line.strip())
            break

    if epoch is not None and epoch > 0:
        recent_weight_file = os.path.join(session_dir, '{}_{:02d}.pt'.format(component_key, epoch))
        last_epoch = epoch
    else:
        weight_files = glob.glob(os.path.join(session_dir, '{}_*.pt'.format(component_key)))
        epochs = [int(os.path.splitext(os.path.basename(f))[0].split('_')[1]) for f in weight_files]
        last_epoch = 0 if len(epochs) == 0 else np.max(epochs)
        recent_weight_file = os.path.join(session_dir, '{}_{:02d}.pt'.format(component_key, last_epoch))

    if os.path.isfile(recent_weight_file):
        logging.info('Restoring {} weights from {}'.format(component_key, recent_weight_file))
        checkpoint = torch.load(recent_weight_file, map_location=str(device))

        if drop_embeddings:
            weight_key = None

            for k in list(checkpoint.keys()):
                if 'prep_inputs_fn' in k:
                    weight_key = k

            if weight_key is not None:
                del checkpoint[weight_key]

        if component_key == 'optimizer':
            component.load_state_dict(checkpoint)

            for state in component.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            component.load_state_dict(checkpoint, strict=False)

    return last_epoch
