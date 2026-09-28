Installation
============


The PySAD framework can be installed via:


.. code-block:: bash

    pip install -U pysad


Alternatively, you can install the library directly using the source code in Github repository by:


.. code-block:: bash

    git clone https://github.com/selimfirat/pysad.git
    cd pysad
    pip install .


**Required Dependencies:**

* Python: 3.10+
* numpy: 2.2.6
* scikit-learn: 1.7.2
* scipy: 1.15.3
* statsmodels: 0.15.0 (for ``pysad.models.SeasonalESD``, ``pysad.models.SeasonalHybridESD`` and ``pysad.transform.preprocessing.ModifiedSTLResidualTransformer``)
* pyod: >=3.6.2
* combo: 0.1.3

**Optional Dependencies:**

* rrcf: 0.4.4 (``pip install pysad[rrcf]``, for ``pysad.models.robust_random_cut_forest.RobustRandomCutForest``)
* PyNomaly: 0.4.0 (``pip install pysad[slop]``, for ``pysad.models.LocalOutlierProbability``)
* mmh3: 5.3.0 (``pip install pysad[xStream]``, for ``pysad.models.xstream.xStream``)
* pandas: 2.3.3 (``pip install pysad[pandas]``, for ``pysad.utils.pandas_streamer.PandasStreamer``)
* jax and jaxlib: >=0.6.1 (``pip install pysad[inqmad]``, for ``pysad.models.inqmad.Inqmad``; required for NumPy 2.0+ compatibility of this module)

Install all optional dependencies at once with ``pip install "pysad[rrcf,slop,xStream,pandas,inqmad]"``.
