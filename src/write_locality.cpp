#define ANKERL_NANOBENCH_IMPLEMENT

#include "write_locality.hpp"
#include "benchmark.hpp"
#include <execution>

namespace WriteLocality {

template <typename DataVector>
void SiteInteraction::applyParameters(const Geometry &geometry,
                                      DataVector &data) {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t iatom = 0; iatom < indices_.size(); ++iatom) {
      const int ispin = iatom + icell * geometry.atoms_per_cell;

      std::get<data_t>(data[ispin])
          .emplace(Data{ispin, magnitude_[iatom], direction_[iatom]});
    }
}

template <typename DataTuple>
void SiteInteraction::clearData(DataTuple &element) const {
  std::get<data_t>(element).reset();
};

scalar SiteInteraction::Energy(const data_t &data_, const vectorfield &spins) {
  if (!data_.has_value())
    return 0;

  const scalar d = dot(spins[data_->i], data_->direction);
  return data_->magnitude * d * d;
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

template <typename DataVector>
void PairInteraction::applyParameters(const Geometry &geometry,
                                      DataVector &data) {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t ipair = 0; ipair < pairs_.size(); ++ipair) {
      const auto &pair = pairs_[ipair];
      int ispin = pair.i + icell * geometry.atoms_per_cell;
      int jspin =
          idx_from_pair(ispin, geometry.n_cells, geometry.atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      // In copy elision we trust
      std::get<data_t>(data[ispin])
          .emplace_back(Data{ispin, jspin, magnitude_[ipair]});
      std::get<data_t>(data[jspin])
          .emplace_back(Data{jspin, ispin, magnitude_[ipair]});
    }
};

template <typename DataTuple>
void PairInteraction::clearData(DataTuple &element) const {
  std::get<data_t>(element).clear();
};

scalar PairInteraction::Energy(const data_t &data_, const vectorfield &spins) {
  scalar E = 0;
  for (const auto &[i, j, c] : data_) {
    E += c * dot(spins[i], spins[j]);
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

template <typename DataVector>
void TripletInteraction::applyParameters(const Geometry &geometry,
                                         DataVector &data) {
  for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell)
    for (std::size_t itriplet = 0; itriplet < triplets_.size(); ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplets_[itriplet];
      int ispin = i + icell * geometry.atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry.n_cells,
                                geometry.atoms_per_cell, {i, j, d_j});
      int kspin = idx_from_pair(ispin, geometry.n_cells,
                                geometry.atoms_per_cell, {i, k, d_k});
      if ((jspin < 0) || (kspin < 0))
        continue;

      std::get<data_t>(data[ispin])
          .emplace_back(Data{ispin, jspin, kspin, magnitude_[itriplet]});
      std::get<data_t>(data[jspin])
          .emplace_back(Data{jspin, kspin, ispin, magnitude_[itriplet]});
      std::get<data_t>(data[kspin])
          .emplace_back(Data{kspin, ispin, jspin, magnitude_[itriplet]});
    }
}

template <typename DataTuple>
void TripletInteraction::clearData(DataTuple &element) const {
  std::get<data_t>(element).clear();
}

scalar TripletInteraction::Energy(const data_t &data_,
                                  const vectorfield &spins) {
  scalar E = 0;
  for (const auto &[i, j, k, c] : data_) {
    E += c * dot(spins[i], cross(spins[j], spins[k]));
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

template <typename DataVector>
void QuadrupletInteraction::applyParameters(const Geometry &geometry,
                                            DataVector &data) {
  for (std::size_t iquad = 0; iquad < quadruplets_.size(); ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplets_[iquad];
    for (std::size_t icell = 0; icell < geometry.n_cells_total; ++icell) {
      int ispin = i + icell * geometry.atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry.n_cells,
                                geometry.atoms_per_cell, {i, j, d_j});
      int kspin = idx_from_pair(ispin, geometry.n_cells,
                                geometry.atoms_per_cell, {i, k, d_k});
      int lspin = idx_from_pair(ispin, geometry.n_cells,
                                geometry.atoms_per_cell, {i, l, d_l});
      if ((jspin < 0) || (kspin < 0) || (lspin < 0))
        continue;

      std::get<data_t>(data[ispin])
          .emplace_back(Data{ispin, jspin, kspin, lspin, magnitude_[iquad]});
      std::get<data_t>(data[jspin])
          .emplace_back(Data{jspin, ispin, kspin, lspin, magnitude_[iquad]});
      std::get<data_t>(data[kspin])
          .emplace_back(Data{kspin, lspin, ispin, jspin, magnitude_[iquad]});
      std::get<data_t>(data[lspin])
          .emplace_back(Data{lspin, kspin, ispin, jspin, magnitude_[iquad]});
    }
  }
}
template <typename DataTuple>
void QuadrupletInteraction::clearData(DataTuple &element) const {
  std::get<data_t>(element).clear();
}

scalar QuadrupletInteraction::Energy(const data_t &data_,
                                     const vectorfield &spins) {
  scalar E = 0;
  for (const auto &[i, j, k, l, c] : data_) {
    E += c * dot(spins[i], spins[j]) * dot(spins[k], spins[l]);
  };

  return 0.25 * E;
};

void QuadrupletInteraction::setParameters(const input_data &parameters ) {
  quadruplets_ = parameters.quadruplets_;
  magnitude_ = parameters.magnitude_;
};

void QuadrupletInteraction::updateParameters(const update_data &parameters) {
  magnitude_ = parameters.magnitude_;
}

} // namespace WriteLocality

int main(const int argc, const char *argv[]) {
  using WriteAggregator = WriteLocality::Aggregator<
      WriteLocality::SiteInteraction, WriteLocality::PairInteraction,
      WriteLocality::TripletInteraction, WriteLocality::QuadrupletInteraction>;
  return run_benchmark<WriteAggregator>(argc, argv);
}
