.. _models:

DTSR Model classes
==================

This package implements two inference algorithms for DTSR: ``DTSRMLE`` (:ref:`dtsrmle`) and ``DTSRBayes`` (:ref:`dtsrbayes`).
The abstract base class ``DTSR`` defines common network-building code and public-facing methods.
To support 3rd party development of new DTSR classes, methods that must be implemented by subclasses of ``DTSR`` are described in its documentation.

.. toctree::
   :caption: DTSR Models:

   dtsrbase
   dtsrmle
   dtsrbayes