#define ANKERL_NANOBENCH_IMPLEMENT

#include "read_locality.hpp"
#include "benchmark.hpp"

namespace ReadLocality {

void SiteInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t iatom = 0; iatom < indices_.size(); ++iatom) {
      const int ispin = iatom + icell * geometry->atoms_per_cell;
      const scalar d = dot(spins[ispin], direction_[iatom]);
      energy[ispin] += magnitude_[iatom] * d * d;
    }
  }
};

void SiteInteraction::setParameters(const input_data& parameters) {
    indices_ = parameters.indices_;
    magnitude_ = parameters.magnitude_;
    direction_ = parameters.direction_;
}
void SiteInteraction::updateParameters(const update_data& parameters) {
    magnitude_ = parameters.magnitude_;
    direction_ = parameters.direction_;
}

void PairInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t ipair = 0; ipair < pairs_.size(); ++ipair) {
      const auto &pair = pairs_[ipair];
      int ispin = pair.i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      const scalar E =
          0.5 * magnitude_[ipair] * dot(spins[ispin], spins[jspin]);
      energy[ispin] += E;
      energy[jspin] += E;
    }
  }
};

void PairInteraction::setParameters(const input_data& parameters) {
    pairs_ = parameters.pairs_;
    magnitude_ = parameters.magnitude_;
}
void PairInteraction::updateParameters(const update_data& parameters) {
    magnitude_ = parameters.magnitude_;
}

void TripletInteraction::Energy(const vectorfield &spins, scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t itriplet = 0; itriplet < triplets_.size(); ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplets_[itriplet];
      int ispin = i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, j, d_j});
      int kspin = idx_from_pair(ispin, geometry->n_cells,
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
};

void TripletInteraction::setParameters(const input_data& parameters) {
    triplets_ = parameters.triplets_;
    magnitude_ = parameters.magnitude_;
}
void TripletInteraction::updateParameters(const update_data& parameters) {
    magnitude_ = parameters.magnitude_;
}

void QuadrupletInteraction::Energy(const vectorfield &spins,
                                   scalarfield &energy) {
  if (!owner)
    return;

  const auto *geometry = owner->geometry.get();
  if (!geometry)
    return;

  for (std::size_t iquad = 0; iquad < quadruplets_.size(); ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplets_[iquad];

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
}

void QuadrupletInteraction::setParameters(const input_data& parameters) {
    quadruplets_ = parameters.quadruplets_;
    magnitude_ = parameters.magnitude_;
}
void QuadrupletInteraction::updateParameters(const update_data& parameters) {
    magnitude_ = parameters.magnitude_;
}

} // namespace ReadLocality

int main(const int argc, const char *argv[]) {
  using ReadAggregator = ReadLocality::Aggregator<
      ReadLocality::SiteInteraction, ReadLocality::PairInteraction,
      ReadLocality::TripletInteraction, ReadLocality::QuadrupletInteraction>;
  return run_benchmark<ReadAggregator>(argc, argv);
}
