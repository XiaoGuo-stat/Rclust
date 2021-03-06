% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rclust.R
\name{rclust}
\alias{rclust}
\title{Randomized spectral clustering using random sampling or random projection}
\usage{
rclust(
  A,
  method = c("rsample", "rproject"),
  k,
  rank,
  p = 10,
  q = 2,
  dist = "normal",
  abs = FALSE,
  P,
  iter.max = 50,
  nstart = 10,
  ...
)
}
\arguments{
\item{A}{The adjacency matrix of an undirected network (binary and symmetric) with type "dgCMatrix".}

\item{method}{The method for computing the randomized eigendecomposition. Random sampling-based eigendecomposition
is implemented if \code{method="rsample"}, and random projection-based
eigendecomposition is implemented if \code{method="rproject"}.}

\item{k}{The number of target clusters.}

\item{rank}{The number of target rank.}

\item{p}{The oversampling parameter in the random projection scheme. Requested only
if \code{method="rproject"}. Default is 10.}

\item{q}{The power parameter in the random projection scheme. Requested only if
\code{method="rproject"}. Default is 2.}

\item{dist}{The distribution of the entry of the random test matrix in the random
projection scheme. Requested only if \code{method="rproject"}. Default
is \code{"normal"}.}

\item{abs}{A logical variable indicating whether the eigen values should be largest in absolute value.
Default is \code{FALSE}, indicating that the eigen values are largest in value.}

\item{P}{The sampling probability in the random sampling scheme. Requested only
if \code{method="rsample"}.}

\item{iter.max}{Maximum number of iterations in the \code{\link[stats]{kmeans}}.
Default is 50.}

\item{nstart}{The number of random sets in \code{\link[stats]{kmeans}}. Default is 10.}

\item{...}{Additional arguments.}
}
\value{
\item{cluster}{The cluster vector (from \code{1:k}) with the numbers indicating which
             cluster each node is assigned.}
        \item{rvectors}{The randomized \code{rank} eigen vectors computed by
             \code{\link[Rclust]{reig.pro}} or \code{\link[Rclust]{reig.sam}}.}
}
\description{
Randomized spectral clustering for undirected networks. The clusters are computed using two random schemes, namely, the random sampling
and the random projection scheme. Can deal with very large networks.
}
\details{
This function computes the clusters of undirected networks using
randomized spectral clustering algorithms. The random projection-based eigendecomposition or the
random sampling-based eigendecomposition is first computed for the adjacency matrix of the undirected network.
The k-means is then performed on the randomized eigen vectors.
}
\examples{
n <- 100
k <- 2
clustertrue <- rep(1:k, each = n/k)
A <- matrix(0, n, n)
for(i in 1: (n-1)) {
   for(j in (i+1):n) {
       A[i, j] <- ifelse(clustertrue[i] == clustertrue[j], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
       A[j, i] <- A[i, j]
    }
}
A <- as(A, "dgCMatrix")
rclust(A, method = "rsample", k = k, rank = k, P = 0.7)


}
\seealso{
\code{\link[Rclust]{reig.pro}}, \code{\link[Rclust]{reig.sam}}.
}
