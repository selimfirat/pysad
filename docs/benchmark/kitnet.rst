KitNet on lympho
----------------

Commit ``87f315b243`` removed temporary autoencoders from the feature mapping period.
This follows the training phases in :cite:`mirsky2018kitsune` and
`KitNET-py <https://github.com/ymirsky/KitNET-py/blob/master/KitNET.py>`_.
During feature mapping, KitNet updates only feature correlations and returns zero scores.
The earlier code trained temporary autoencoders during this period and rebuilt them when the number of points doubled.
Those temporary models produced scores before the final feature map existed.

With seed 0 and default parameters, the lympho AUROC fell from 0.8113 to 0.2783.
The dataset contains 148 points. The benchmark excludes the first 37 scores as warm-up.
KitNet uses 100 points for feature mapping and then starts training its autoencoders.
Prequential scoring runs before fitting, so the first 101 scores are zero.
All five evaluated anomalies occur at zero-based indices 70, 86, 90, 94 and 98.
Each receives a zero score, while later normal points receive positive scores.
This explains the AUROC below chance.

The lower AUROC is an intended consequence of removing the temporary models.
It does not indicate a reversed anomaly score.
The benchmark uses default parameters for every detector and includes scores from unfinished training phases.
If a stream is short, choose grace periods that leave enough points for evaluation after training.
This changes the evaluation setup, so compare such results separately from the default benchmark.
