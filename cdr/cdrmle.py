import sys
import os
import numpy as np
import pandas as pd

from .cdrbase import CDR
from .kwargs import CDRMLE_INITIALIZATION_KWARGS
from .util import sn, stderr

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None



######################################################
#
#  MLE IMPLEMENTATION OF CDR
#
######################################################


class CDRMLE(CDR):
    _INITIALIZATION_KWARGS = CDRMLE_INITIALIZATION_KWARGS

    _doc_header = """
        A CDR implementation fitted using maximum likelihood estimation.
    """
    _doc_args = CDR._doc_args
    _doc_kwargs = CDR._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    ######################################################
    #
    #  Initialization methods
    #
    ######################################################

    def __init__(
            self,
            form_str,
            X,
            Y,
            **kwargs
    ):

        super(CDRMLE, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self._initialize_metadata()

        self.build()

    def _pack_metadata(self):
        md = super(CDRMLE, self)._pack_metadata()
        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        super(CDRMLE, self)._unpack_metadata(md)

        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            stderr('Saved model contained unrecognized attributes %s which are being ignored\n' % sorted(list(md.keys())))





    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def report_settings(self, indent=0):
        out = super(CDRMLE, self).report_settings(indent=indent)
        for kwarg in CDRMLE_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out
