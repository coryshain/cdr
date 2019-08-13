.. _models:

CDR model classes
==================

This package implements two inference algorithms for CDR: ``CDRMLE`` (:ref:`cdrmle`) and ``CDRBayes`` (:ref:`cdrbayes`).
The abstract base class ``CDR`` defines common network-building code and public-facing methods.
To support 3rd party development of new CDR classes, methods that must be implemented by subclasses of ``CDR`` are described in its documentation.

.. toctree::
   :caption: CDR Models:

   cdrbase
   cdrmle
   cdrbayes