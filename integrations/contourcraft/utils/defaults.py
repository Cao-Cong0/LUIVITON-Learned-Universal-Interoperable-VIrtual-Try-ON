import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'
DEFAULTS['data_root'] = os.environ['HOOD_DATA']
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = os.environ['HOOD_PROJECT']
DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = '/path/to/AMASS/smpl/CMU'

DEFAULTS = munchify(DEFAULTS)
