#include <Eigen/SparseCore>
#include "csr_matrix.hpp"


CSRMatrix make_csr_from_eigen(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& m);
