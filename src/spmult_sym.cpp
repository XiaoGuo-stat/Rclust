#include <RcppEigen.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

// res = x[ind[0]] + x[ind[1]] + ... + x[ind[n-1]]
#ifdef __AVX2__
#include <immintrin.h>
inline double gather_sum(const double* x, const int* ind, int n)
{
    constexpr int simd_size = 8;
    const int* idx_end = ind + n;
    // const int* idx_simd_end = ind + (n - n % simd_size);
    // n % 8 == n & 7, see https://stackoverflow.com/q/3072665
    const int* idx_simd_end = idx_end - (n & (simd_size - 1));

    __m256d resv = _mm256_set1_pd(0.0);
    for(; ind < idx_simd_end; ind += simd_size)
    {
        __m128i idx1 = _mm_loadu_si128((__m128i*) ind);
        __m256d v1 = _mm256_i32gather_pd(x, idx1, sizeof(double));
        __m128i idx2 = _mm_loadu_si128((__m128i*) (ind + 4));
        __m256d v2 = _mm256_i32gather_pd(x, idx2, sizeof(double));
        resv += (v1 + v2);
    }

    double res = 0.0;
    for(; ind < idx_end; ind++)
        res += x[*ind];

    return res + resv[0] + resv[1] + resv[2] + resv[3];
}
#else
inline double gather_sum(const double* x, const int* ind, int n)
{
    double res = 0.0;
    const int* idx_end = ind + n;
    for(; ind < idx_end; ind++)
        res += x[*ind];
    return res;
}
#endif

// y[ind[0]] += c, y[ind[1]] += c, ..., y[ind[n-1]] += c
inline void scatter(double* y, const int* ind, int n, double c)
{
    constexpr int simd_size = 8;
    const int* idx_end = ind + n;
    // n % 8 == n & 7, see https://stackoverflow.com/q/3072665
    const int* idx_simd_end = idx_end - (n & (simd_size - 1));

    for(; ind < idx_simd_end; ind += simd_size)
    {
        y[ind[0]] += c;
        y[ind[1]] += c;
        y[ind[2]] += c;
        y[ind[3]] += c;
        y[ind[4]] += c;
        y[ind[5]] += c;
        y[ind[6]] += c;
        y[ind[7]] += c;
    }

    for(; ind < idx_end; ind++)
        y[*ind] += c;
}

using SpMat = Eigen::SparseMatrix<double>;
using MapSpMat = Eigen::Map<SpMat>;

// mat is a binary symmetric sparse matrix of class dgCMatrix,
// with zeros on the diagonal.
//
// res = mat * v = mat' * v
inline void symspbin_prod(const MapSpMat& mat, const double* v, double* res)
{
    const int n = mat.rows();

    const int* inner = mat.innerIndexPtr();
    const int* outer = mat.outerIndexPtr();
    for(int j = 0; j < n; j++)
    {
        const int Ai_len = outer[j + 1] - outer[j];
        const int* Ai_start = inner + outer[j];
        const int* Ai_end = Ai_start + Ai_len;
        res[j] = gather_sum(v, Ai_start, Ai_len);
    }
}

// mat is a binary symmetric sparse matrix of class dgCMatrix,
// with zeros on the diagonal.
//
// res = mat^2 * v
inline void symspbin_doubleprod(const MapSpMat& mat, const double* v, double* res)
{
    const int n = mat.rows();

    // Zero out result vector
    std::fill(res, res + n, 0.0);

    const int* inner = mat.innerIndexPtr();
    const int* outer = mat.outerIndexPtr();
    for(int j = 0; j < n; j++)
    {
        const int Ai_len = outer[j + 1] - outer[j];
        const int* Ai_start = inner + outer[j];
        const double c = gather_sum(v, Ai_start, Ai_len);
        scatter(res, Ai_start, Ai_len, c);
    }
}

// res = (AA')^q AP = A^(2q+1)P
// [[Rcpp::export]]
NumericMatrix symspbin_power_prod(Rcpp::S4 A, NumericMatrix P, int q = 0, int nthread = 1)
{
    MapSpMat mat = Rcpp::as<MapSpMat>(A);
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
            symspbin_doubleprod(mat, r, w);
            std::copy(w, w + n, r);
        }
    }

    delete [] work;

    return res;
}
