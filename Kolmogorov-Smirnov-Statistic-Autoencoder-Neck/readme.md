# Numerical Hypothesis Testing

How well the *pristine* and *damage* subpopulations of samples can be discriminated, gets expressed in one number: The numerical computations are summarized in the following pseudo code based on the R language...

```
mapping    <- prcomp(data_train)                   # PCA-based autoencoder fitting either "data_focal" or "data_context"
projection <- predict(mapping, data_focal)[, 1]    # compressing 2D to 1D
ks_result  <- ks.test(projection[pristine_indices], projection[damaged_indices])
print(ks_result$statistic)                         # comparison results: 0 = indistinguishable, 1 = fully separated```
