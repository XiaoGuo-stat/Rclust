#include <RcppEigen.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

#ifdef __AVX2__
#include <immintrin.h>
// Array x, indices ind = (i1, ..., in), array y
// (1) Compute r = x[i1] + ... + x[in]
// (2) Do y[i1] += c, ..., y[in] += c
// (3) On exit the ind pointer points to the end of ind
inline double gather_and_scatter(
    const double* x, const int*& ind, double c, double* y, int n
)
{
    // Process 4 values at one time
    constexpr int simd_size = 4;
    const int* ind_simd_end = ind + (n - n % simd_size);
    const int* ind_end = ind + n;

    __m256d rs = _mm256_set1_pd(0.0);
    for(; ind < ind_simd_end; ind += simd_size)
    {
        // 4 indices in a packet
        __m128i inds = _mm_loadu_si128((__m128i*) ind);
        // 4 x values in a packet
        __m256d xs = _mm256_i32gather_pd(x, inds, sizeof(double));
        // Add values
        rs += xs;
        // Scattering
        y[ind[0]] += c;
        y[ind[1]] += c;
        y[ind[2]] += c;
        y[ind[3]] += c;
    }

    double r = 0.0;
    for(; ind < ind_end; ind++)
    {
        r += x[*ind];
        y[*ind] += c;
    }
    return r + rs[0] + rs[1] + rs[2] + rs[3];
}
#else
inline double gather_and_scatter(
    const double* x, const int*& ind, double c, double* y, int n
)
{
    const int* ind_end = ind + n;
    double r = 0.0;
    for(; ind < ind_end; ind++)
    {
        r += x[*ind];
        y[*ind] += c;
    }
    return r;
}
#endif

using SpMat = Eigen::SparseMatrix<double>;
using MapSpMat = Eigen::Map<SpMat>;

// mat is a binary symmetric sparse matrix of class dgCMatrix,
// with only lower-triangular part. The diagonal elements are all zero
//
// res = mat * v
inline void symspbin_prod(const SpMat& mat, const double* v, double* res)
{
    const int n = mat.rows();

    // Zero out result vector
    std::fill(res, res + n, 0.0);

    const int* inner = mat.innerIndexPtr();
    const int* outer = mat.outerIndexPtr();
    for(int j = 0; j < n; j++)
    {
        const int* Ai_start = inner + outer[j];
        const int* Ai_end = inner + outer[j + 1];
        double tprod = gather_and_scatter(v, Ai_start, v[j], res, Ai_end - Ai_start);
        res[j] += tprod;
    }
}

// res = (AA')^q AP = A^(2q+1)P
// [[Rcpp::export]]
NumericMatrix symspbin_power_prod(Rcpp::S4 A, NumericMatrix P, int q = 0, int nthread = 1)
{
    SpMat mat = Rcpp::as<MapSpMat>(A).triangularView<Eigen::StrictlyLower>();
    const int n = P.nrow();
    const int k = P.ncol();
    NumericMatrix res(Rcpp::no_init_matrix(n, k));
    // Allocate additional working space when q>0
    const int wrows = (q == 0) ? 1 : n;
    const int wcols = (q == 0) ? 1 : k;
    double* work = new double[wrows * wcols];

    const double* P_ptr = P.begin();
    double* res_ptr = res.begin();

    #pragma omp parallel for shared(P_ptr, res_ptr, work, mat) num_threads(nthread)
    for(int j = 0; j < k; j++)
    {
        const int offset = j * n;
        const double* v = P_ptr + offset;
        double* r = res_ptr + offset;
        double* w = work + offset;
        // res = AP
        symspbin_prod(mat, v, r);

        // Power iterations
        for(int i = 0; i < q; i++)
        {
            symspbin_prod(mat, r, w);
            symspbin_prod(mat, w, r);
        }
    }

    delete [] work;

    return res;
}
