// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

namespace mlx::core {

void inverse_impl(const array& a, array& inv) {
  // Lapack uses the column-major convention. We take advantage of the following
  // identity to avoid transposing (see
  // https://math.stackexchange.com/a/340234):
  //   (A⁻¹)ᵀ = (Aᵀ)⁻¹

  // The inverse is computed in place, so just copy the input to the output.
  inv.set_data(allocator::malloc_or_wait(inv.nbytes()));
  copy_inplace(a, inv, CopyType::GeneralGeneral);

  const int N = a.shape(-1);
  const size_t num_matrices = a.size() / (N * N);

  int info;
  std::vector<int> ipiv;
  ipiv.resize(N);

  for (int i = 0; i < num_matrices; i++) {
    // Compute LU factorization.
    sgetrf_(
        /* m = */ &N,
        /* n = */ &N,
        /* a = */ inv.data<float>() + N * N * i,
        /* lda = */ &N,
        /* ipiv = */ ipiv.data(),
        /* info = */ &info);

    if (info != 0) {
      throw std::runtime_error("inverse_impl: LU factorization failed.");
    }

    static const int lwork_query = -1;
    float workspace_size = 0;

    // Compute workspace size.
    sgetri_(
        /* m = */ &N,
        /* a = */ nullptr,
        /* lda = */ &N,
        /* ipiv = */ nullptr,
        /* work = */ &workspace_size,
        /* lwork = */ &lwork_query,
        /* info = */ &info);

    if (info != 0) {
      throw std::runtime_error("inverse_impl: workspace calculation failed.");
    }

    const int lwork = workspace_size;
    auto scratch = allocator::malloc_or_wait(sizeof(float) * lwork);

    // Compute inverse.
    sgetri_(
        /* m = */ &N,
        /* a = */ inv.data<float>() + N * N * i,
        /* lda = */ &N,
        /* ipiv = */ ipiv.data(),
        /* work = */ static_cast<float*>(scratch.raw_ptr()),
        /* lwork = */ &lwork,
        /* info = */ &info);

    if (info != 0) {
      throw std::runtime_error("inverse_impl: inversion failed.");
    }
  }
}

void Inverse::eval(const std::vector<array>& inputs, array& output) {
  if (!(inputs[0].dtype() == float32)) {
    throw std::runtime_error("[Inverse::eval] only supports float32.");
  }
  inverse_impl(inputs[0], output);
}

} // namespace mlx::core
