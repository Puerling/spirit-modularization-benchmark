#define ANKERL_NANOBENCH_IMPLEMENT

#include "polymorphic.hpp"
#include "benchmark.hpp"
#include <memory>

#ifdef SPIRIT_USE_CUDA
#include <Eigen/Dense>
#endif

namespace Polymorphic {

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Site(const Vector3 *spins, const size_t atoms_per_cell,
                          const size_t n_anisotropies, const int *indices_,
                          const scalar *magnitude_, const Vector3 *direction_,
                          scalar *energy, const size_t n_cells_total) {
  for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
       icell < n_cells_total; icell += blockDim.x * gridDim.x) {
    for (size_t iatom = 0; iatom < n_anisotropies; ++iatom) {
      const int ispin = iatom + icell * atoms_per_cell;
      const scalar d = spins[ispin].dot(direction_[iatom]);
      energy[ispin] += magnitude_[iatom] * d * d;
    }
  }
};
#endif

void SiteInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

#ifdef SPIRIT_USE_CUDA
  static constexpr int blockSize = 1024;
  CU_E_Site<<<(spins.size() + blockSize - 1) / blockSize, blockSize>>>(
      spins.data(), geometry->atoms_per_cell, indices_.size(), indices_.data(),
      magnitude_.data(), direction_.data(), energy.data(),
      geometry->n_cells_total);
  CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t iatom = 0; iatom < indices_.size(); ++iatom) {
      const int ispin = iatom + icell * geometry->atoms_per_cell;
      const scalar d = dot(spins[ispin], direction_[iatom]);
      energy[ispin] += magnitude_[iatom] * d * d;
    }
  }
#endif
};

void SiteInteraction::setParameters(const input_data &parameters) {
  indices_ = parameters.indices_;
  magnitude_ = parameters.magnitude_;
  direction_ = parameters.direction_;
}
void SiteInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
  direction_ = parameters.direction_;
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Pair(const Vector3 *spins, const int atoms_per_cell,
                          const size_t n_pairs, const Pair *pairs_,
                          const scalar *magnitude_, scalar *energy,
                          const size_t n_cells[3], const size_t n_cells_total) {
  for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
       icell < n_cells_total; icell += blockDim.x * gridDim.x) {
    for (size_t ipair = 0; ipair < n_pairs; ++ipair) {
      const auto &pair = pairs_[ipair];
      int ispin = pair.i + icell * atoms_per_cell;
      int jspin = idx_from_pair(ispin, n_cells, atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      const scalar E = 0.5 * magnitude_[ipair] * spins[ispin].dot(spins[jspin]);
      energy[ispin] += E;
      energy[jspin] += E;
    }
  }
};
#endif

void PairInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

#ifdef SPIRIT_USE_CUDA
  static constexpr int blockSize = 1024;
  CU_E_Pair<<<(spins.size() + blockSize - 1) / blockSize, blockSize>>>(
      spins.data(), geometry->atoms_per_cell, pairs_.size(), pairs_.data(),
      magnitude_.data(), energy.data(), geometry->n_cells.data(),
      geometry->n_cells_total);
  CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t ipair = 0; ipair < pairs_.size(); ++ipair) {
      const auto &pair = pairs_[ipair];
      int ispin = pair.i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      const scalar E = 0.5 * magnitude_[ipair] * dot(spins[ispin], spins[jspin]);
      energy[ispin] += E;
      energy[jspin] += E;
    }
  }
#endif
};

void PairInteraction::setParameters(const input_data &parameters) {
  pairs_ = parameters.pairs_;
  magnitude_ = parameters.magnitude_;
}
void PairInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Triplet(const Vector3 *spins, const size_t atoms_per_cell,
                             const size_t n_triplet, const Triplet *triplets_,
                             const scalar *magnitude_, scalar *energy,
                             const size_t n_cells[3],
                             const size_t n_cells_total) {
  for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
       icell < n_cells_total; icell += blockDim.x * gridDim.x) {
    for (std::size_t itriplet = 0; itriplet < n_triplet; ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplets_[itriplet];
      const int ispin = i + icell * atoms_per_cell;
      const int jspin = idx_from_pair(ispin, n_cells, atoms_per_cell,
                                      {i, j, {d_j[0], d_j[1], d_j[2]}});
      const int kspin = idx_from_pair(ispin, n_cells, atoms_per_cell,
                                      {i, k, {d_k[0], d_k[1], d_k[2]}});

      if ((jspin < 0) || (kspin < 0))
        continue;

      const scalar E = (1.0 / 3.0) * magnitude_[itriplet] *
                       spins[ispin].dot(spins[jspin].cross(spins[kspin]));
      energy[ispin] += E;
      energy[jspin] += E;
      energy[kspin] += E;
    }
  }
};
#endif

void TripletInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

#ifdef SPIRIT_USE_CUDA
  static constexpr int blockSize = 1024;
  CU_E_Triplet<<<(spins.size() + blockSize - 1) / blockSize, blockSize>>>(
      spins.data(), geometry->atoms_per_cell, triplets_.size(),
      triplets_.data(), magnitude_.data(), energy.data(),
      geometry->n_cells.data(), geometry->n_cells_total);
  CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t itriplet = 0; itriplet < triplets_.size(); ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplets_[itriplet];
      const int ispin = i + icell * geometry->atoms_per_cell;
      const int jspin = idx_from_pair(ispin, geometry->n_cells,
                                      geometry->atoms_per_cell, {i, j, d_j});
      const int kspin = idx_from_pair(ispin, geometry->n_cells,
                                      geometry->atoms_per_cell, {i, k, d_k});

      if ((jspin < 0) || (kspin < 0))
        continue;

      const scalar E = (1.0 / 3.0) * magnitude_[itriplet] *
                       dot(spins[ispin], cross(spins[jspin], spins[kspin]));
      energy[ispin] += E;
      energy[jspin] += E;
      energy[kspin] += E;
    }
  }
#endif
};

void TripletInteraction::setParameters(const input_data &parameters) {
  triplets_ = parameters.triplets_;
  magnitude_ = parameters.magnitude_;
}
void TripletInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
}

#ifdef SPIRIT_USE_CUDA
__global__ void
CU_E_Quadruplet(const Vector3 *spins, const size_t atoms_per_cell,
                const size_t n_quad, const Quadruplet *quadruplets_,
                const scalar *magnitude_, scalar *energy,
                const size_t n_cells[3], const size_t n_cells_total) {
  for (std::size_t iquad = 0; iquad < n_quad; ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplets_[iquad];

    for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
         icell < n_cells_total; icell += blockDim.x * gridDim.x) {
      for (size_t iani = 0; iani < n_quad; ++iani) {
        const int ispin = i + icell * atoms_per_cell;
        const int jspin = idx_from_pair(ispin, n_cells, atoms_per_cell,
                                        {i, j, {d_j[0], d_j[1], d_j[2]}});
        const int kspin = idx_from_pair(ispin, n_cells, atoms_per_cell,
                                        {i, k, {d_k[0], d_k[1], d_k[2]}});
        const int lspin = idx_from_pair(ispin, n_cells, atoms_per_cell,
                                        {i, l, {d_l[0], d_l[1], d_l[2]}});

        if ((jspin < 0) || (kspin < 0) || (lspin < 0))
          continue;

        const scalar E = 0.25 * magnitude_[iquad] *
                         spins[ispin].dot(spins[jspin]) *
                         spins[kspin].dot(spins[lspin]);
        energy[ispin] += E;
        energy[jspin] += E;
        energy[kspin] += E;
        energy[lspin] += E;
      }
    }
  }
};
#endif

void QuadrupletInteraction::Energy(const vectorfield &spins,
                                   scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

#ifdef SPIRIT_USE_CUDA
  static constexpr int blockSize = 1024;
  CU_E_Quadruplet<<<(spins.size() + blockSize - 1) / blockSize, blockSize>>>(
      spins.data(), geometry->atoms_per_cell, quadruplets_.size(),
      quadruplets_.data(), magnitude_.data(), energy.data(),
      geometry->n_cells.data(), geometry->n_cells_total);
  CU_CHECK_AND_SYNC();
#else
  for (std::size_t iquad = 0; iquad < quadruplets_.size(); ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplets_[iquad];

#pragma omp parallel for
    for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
      int ispin = i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, j, d_j});
      int kspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, k, d_k});
      int lspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, l, d_l});

      if ((jspin < 0) || (kspin < 0) || (lspin < 0))
        continue;

      const scalar E = 0.25 * magnitude_[iquad] *
                       dot(spins[ispin], spins[jspin]) *
                       dot(spins[kspin], spins[lspin]);
      energy[ispin] += E;
      energy[jspin] += E;
      energy[kspin] += E;
      energy[lspin] += E;
    }
  }
#endif
}

void QuadrupletInteraction::setParameters(const input_data &parameters) {
  quadruplets_ = parameters.quadruplets_;
  magnitude_ = parameters.magnitude_;
}
void QuadrupletInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
}

} // namespace Polymorphic

int main(const int argc, const char *argv[]) {
  using PolymorphicAggregator = Polymorphic::Aggregator<
      Polymorphic::SiteInteraction, Polymorphic::PairInteraction,
      Polymorphic::TripletInteraction, Polymorphic::QuadrupletInteraction>;
  return run_benchmark<PolymorphicAggregator>(argc, argv);
}
