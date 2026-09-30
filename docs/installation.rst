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
* numpy: >=1.22.4
* scikit-learn: >=1.4.2
* scipy: >=1.13.0
* statsmodels: >=0.14.2 (for ``pysad.models.SeasonalESD``, ``pysad.models.SeasonalHybridESD`` and ``pysad.transform.preprocessing.ModifiedSTLResidualTransformer``)
* pyod: >=3.6.2

**Optional Dependencies:**

* rrcf: 0.4.4 (``pip install pysad[rrcf]``, for ``pysad.models.robust_random_cut_forest.RobustRandomCutForest``)
* PyNomaly: 0.4.0 (``pip install pysad[slop]``, for ``pysad.models.LocalOutlierProbability``)
* mmh3: 5.3.0 (``pip install pysad[xStream]``, for ``pysad.models.xstream.xStream``)
* pandas: >=2.2.2 (``pip install pysad[pandas]``, for ``pysad.utils.pandas_streamer.PandasStreamer``)

Install all optional dependencies at once with ``pip install "pysad[rrcf,slop,xStream,pandas]"``.
