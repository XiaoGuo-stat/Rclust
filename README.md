## Randomized Spectral Clustering for Large-Scale Undirected Networks
### Introduction

**Rclust** performs spectral clustering for large-scale undirected networks using
randomization techniques including the random projection and the random sampling. Specifically, the
random-projection-based eigendecomposition or the random-sampling-based eigendecomposition is first computed for the 
adjacency matrix of the undirected network. The k-means is 
then performed on the randomized eigen vectors. 

### Examples

We use the a real network `youtubeNetwork` to illustrate. The `youtubeNetwork` object is a sparse matrix 
representing the adjacency matrix of the undirected youtube social network. There is 1157828 nodes and 2987624 edges.

```r
library(Rclust)
data(youtubeNetwork)
A <- youtubeNetwork
```

The random-projection-based eigendecomposition of `A` can be computed via

```r
reig.pro(A, rank = 2, p = 10, q = 2, dist = "normal", nthread = 2)
```

The random-sampling-based eigendecomposition of `A` can be computed via

```r
reig.sam(A, P = 0.7, use_lower = TRUE, k = 2, tol = 1e-05)
```

The corresponding random-projection-based and random-sampling-based spectral clustering can be
performed respectively using 

```r
rclust(A, method = "rproject", k = 3, rank = 2, p = 10, q = 2, dist = "normal")
rclust(A, method = "rsample", k = 3, rank = 2, P = 0.7)
```

The package also provides a function for sampling a sparse symmetric matrix with given probability:

```r
rsample_sym(A, P = 0.5)
```














