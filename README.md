# In Pursuit of Open-Weights Models for Intrinsic Interpretability in Wind Turbine Blade Monitoring with Tower-Mounted Radar
Data and code from the paper on public classifiers for mast-bound radar-based remote sensing.

## Hypothesis Testing

A code snippet to illustrate the numerical hypothesis testing can be found [here](./Kolmogorov-Smirnov-Statistic-Autoencoder-Neck/).

## Datasets

Although its authors did not give it that particular name, the field-measurement dataset was later dubbed WiRoRa (**Wi**nd-turbine **Ro**tor-blade **Ra**dargrams) and can be obtained [here](https://zenodo.org/records/11483931).

The related **Si**mulated **WiRoRa** (SiWiRoRa) dataset is deposited [here](https://zenodo.org/records/13318595).

## Annotations

Multi-class anomaly labels for WiRoRa are [here](./Outlier-Annotation-WiRoRa/).

To retrieve on-the-fly orientation angles from radargrams, ellipse estimation has been tested [here](./Fitzgibbon-Ellipse-Fitting-Yaw/).

Further explanation for human-like pairwise image ranking are located [here](./Human-Bradley-Terry-Yaw-Ranking/).

## Surrogate Data

The image-data generator resides [here](./Billion-Surrogate-Samples-Generator).

## Model Checkpoints

The main neural models can be found [here](https://zenodo.org/records/18937969).
