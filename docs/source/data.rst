.. _data.rst:

Data processing tools
=====================

There are three core high-level data processing methods provided by this package: ``read_data()``, ``preprocess_data()``, and ``build_DTSR_impulses()``.
The ``read_data()`` method loads data from source tables into memory and performs minimal preprocessing.
The ``preprocess_data()`` method performs any preprocessing steps required by the model structure and hyperparameters.
The ``build_DTSR_impulses()`` compiles the data into arrays that are ready to be passed to DTSR instances for training or evaluation.

Since ``build_DTSR_impulses()`` is run internally by all DTSR instance methods that deal with training or prediction, it should generally not need to be called by hand.
However, for expository purposes, the following is an example of a pipeline that converts text data from files into DTSR-ready model inputs:

.. code-block:: python

  X_train_path = ['X_train.csv']
  y_train_path = ['y_train.csv']
  series_ids = ['subject', 'documentID']
  categorical_columns = ['subject', 'documentID']
  dstr_formula_list = ['DTSR_modelA', 'DTSR_modelB']
  impulse_names = ['predictor_1', 'predictor_2', 'predictor_3']

  # Read data
  X, y = read_data(
      X_train_path,
      y_train_path,
      series_ids,
      categorical_columns=categorical_columns
  )

  # Preprocess data
  X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, \
  X_2d_predictor_names, X_2d_predictors = preprocess_data(
      X,
      y,
      dtsr_formula_list,
      series_ids,
  )

  # Build DTSR-ready data
  X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
      X,
      y.first_obs,
      y.last_obs,
      impulse_names,
      X_response_aligned_predictor_names=X_response_aligned_predictor_names,
      X_response_aligned_predictors=X_response_aligned_predictors,
      X_2d_predictor_names=X_2d_predictor_names,
      X_2d_predictors=X_2d_predictors
  )

The variables ``X_2d``, ``time_X_2d``, and ``time_X_mask`` can then be fed to the ``X``, ``time_X``, and ``time_X_mask`` Tensorflow placeholders of the DTSR instance.
Note that ``X`` and ``y`` are the streams of impulse and response data, respectively.
In addition, ``preprocess_data`` returns some additional data structures that may be ``None`` if not relevant to the model.
Response-aligned predictors are predictors contained in the response data ``y`` rather than the impulse data ``X`` (i.e. predictors that share a temporal tokenization with the response).
2D predictors are impulses whose value depends on that of the most recent impulse.
The ``preprocess_data`` method returns variable names and data associated with any such impulses in the model.
For the most part, these variables do not need to be inspected or manipulated, simply stored and passed to any other methods that call for them.

For detailed documentation of all data processing methods, see the ``dtsr.io``, ``dtsr.data``, and ``dtsr.formula`` sections of the Module Index.



