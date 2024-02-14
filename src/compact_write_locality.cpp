#define ANKERL_NANOBENCH_IMPLEMENT

#include "compact_write_locality.hpp"
#include "benchmark.hpp"

namespace CompactWriteLocality {

template <typename IndexVector>
void SiteInteraction::applyGeometry(const Geometry &geometry,
                                    IndexVector &data) const {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t iatom = 0; iatom < indices_.size(); ++iatom) {
      const int ispin = iatom + icell * geometry.atoms_per_cell;

      std::get<index_t>(data[ispin]).emplace(Index{iatom, ispin});
    }
}

template <typename IntexTuple>
void SiteInteraction::clearData(IntexTuple &element) const {
  std::get<index_t>(element).reset();
};

scalar SiteInteraction::Energy(const index_t &indices,
                               const vectorfield &spins) const {
  if (!indices.has_value())
    return 0;

  const auto &[iatom, ispin] = *indices;
  const scalar d = dot(spins[ispin], direction_[iatom]);
  return magnitude_[iatom] * d * d;
};

void SiteInteraction::setParameters(const input_data &parameters) {
  indices_ = parameters.indices_;
  magnitude_ = parameters.magnitude_;
  direction_ = parameters.direction_;
};

void SiteInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
  direction_ = parameters.direction_;
};

template <typename IndexVector>
void PairInteraction::applyGeometry(const Geometry &geometry,
                                    IndexVector &indices) const {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t ipair = 0; ipair < pairs_.size(); ++ipair) {
      const auto &pair = pairs_[ipair];
      const int ispin = pair.i + icell * geometry.atoms_per_cell;
      const int jspin =
          idx_from_pair(ispin, geometry.n_cells, geometry.atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      // In copy elision we trust
      std::get<index_t>(indices[ispin])
          .emplace_back(Index{ipair, ispin, jspin});
      std::get<index_t>(indices[jspin])
          .emplace_back(Index{ipair, jspin, ispin});
    }
};

template <typename IndexTuple>
void PairInteraction::clearData(IndexTuple &element) const {
  std::get<index_t>(element).clear();
};

scalar PairInteraction::Energy(const index_t &indices,
                               const vectorfield &spins) const {
  scalar E = 0;
  for (const auto &[ipair, i, j] : indices) {
    E += magnitude_[ipair] * dot(spins[i], spins[j]);
  }

  return 0.5 * E;
};

void PairInteraction::setParameters(const input_data &parameters) {
  pairs_ = parameters.pairs_;
  magnitude_ = parameters.magnitude_;
};

void PairInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
};

template <typename IndexVector>
void TripletInteraction::applyGeometry(const Geometry &geometry,
                                       IndexVector &indices) const {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t itriplet = 0; itriplet < triplets_.size(); ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplets_[itriplet];
      const int ispin = i + icell * geometry.atoms_per_cell;
      const int jspin = idx_from_pair(
          ispin, geometry.n_cells, geometry.atoms_per_cell, {i, j, d_j});
      const int kspin = idx_from_pair(
          ispin, geometry.n_cells, geometry.atoms_per_cell, {i, k, d_k});
      if ((jspin < 0) || (kspin < 0))
        continue;

      std::get<index_t>(indices[ispin])
          .emplace_back(Index{itriplet, ispin, jspin, kspin});
      std::get<index_t>(indices[jspin])
          .emplace_back(Index{itriplet, jspin, kspin, ispin});
      std::get<index_t>(indices[kspin])
          .emplace_back(Index{itriplet, kspin, ispin, jspin});
    }
}

template <typename IndexTuple>
void TripletInteraction::clearData(IndexTuple &element) const {
  std::get<index_t>(element).clear();
}

scalar TripletInteraction::Energy(const index_t &indices,
                                  const vectorfield &spins) const {
  scalar E = 0;
  for (const auto &[itriplet, i, j, k] : indices) {
    E += magnitude_[itriplet] * dot(spins[i], cross(spins[j], spins[k]));
  }

  return (1.0 / 3.0) * E;
};

void TripletInteraction::setParameters(const input_data &parameters) {
  triplets_ = parameters.triplets_;
  magnitude_ = parameters.magnitude_;
};

void TripletInteraction::updateParameters(const update_data &parameters) {

  magnitude_ = parameters.magnitude_;
};

template <typename IndexVector>
void QuadrupletInteraction::applyGeometry(const Geometry &geometry,
                                          IndexVector &indices) const {
  for (std::size_t iquad = 0; iquad < quadruplets_.size(); ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplets_[iquad];
    for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell) {
      const int ispin = i + icell * geometry.atoms_per_cell;
      const int jspin = idx_from_pair(
          ispin, geometry.n_cells, geometry.atoms_per_cell, {i, j, d_j});
      const int kspin = idx_from_pair(
          ispin, geometry.n_cells, geometry.atoms_per_cell, {i, k, d_k});
      const int lspin = idx_from_pair(
          ispin, geometry.n_cells, geometry.atoms_per_cell, {i, l, d_l});
      if ((jspin < 0) || (kspin < 0) || (lspin < 0))
        continue;

      std::get<index_t>(indices[ispin])
          .emplace_back(Index{iquad, ispin, jspin, kspin, lspin});
      std::get<index_t>(indices[jspin])
          .emplace_back(Index{iquad, jspin, ispin, kspin, lspin});
      std::get<index_t>(indices[kspin])
          .emplace_back(Index{iquad, kspin, lspin, ispin, jspin});
      std::get<index_t>(indices[lspin])
          .emplace_back(Index{iquad, lspin, kspin, ispin, jspin});
    }
  }
}
template <typename IndexTuple>
void QuadrupletInteraction::clearData(IndexTuple &element) const {
  std::get<index_t>(element).clear();
}

scalar QuadrupletInteraction::Energy(const index_t &indices,
                                     const vectorfield &spins) const {
  scalar E = 0;
  for (const auto &[iquad, i, j, k, l] : indices) {
    E += magnitude_[iquad] * dot(spins[i], spins[j]) * dot(spins[k], spins[l]);
  };

  return 0.25 * E;
};

void QuadrupletInteraction::setParameters(const input_data &parameters) {
  quadruplets_ = parameters.quadruplets_;
  magnitude_ = parameters.magnitude_;
};

void QuadrupletInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
}

} // namespace CompactWriteLocality

int main(const int argc, const char *argv[]) {
  using namespace CompactWriteLocality;
  using CompactWriteAggregator =
      Aggregator<SiteInteraction, PairInteraction, TripletInteraction,
                 QuadrupletInteraction>;
  return run_benchmark<CompactWriteAggregator>(argc, argv);
}
