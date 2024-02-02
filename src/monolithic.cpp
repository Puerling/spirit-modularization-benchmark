#define ANKERL_NANOBENCH_IMPLEMENT

#include "monolithic.hpp"
#include "benchmark.hpp"

namespace Monolithic {

void Aggregator<SiteInteraction, PairInteraction, TripletInteraction,
                QuadrupletInteraction>::Energy(const vectorfield &spins,
                                               scalarfield &energy) {
  if (!geometry)
    return;

  // sites
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t iatom = 0; iatom < site_indices.size(); ++iatom) {
      const int ispin = iatom + icell * geometry->atoms_per_cell;
      const scalar d = dot(spins[ispin], site_directions[iatom]);
      energy[ispin] += site_magnitudes[iatom] * d * d;
    }
  }

  // pairs
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t ipair = 0; ipair < pair_pairs.size(); ++ipair) {
      const auto &pair = pair_pairs[ipair];
      int ispin = pair.i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, pair);

      if (jspin < 0)
        continue;

      const scalar E =
          0.5 * pair_magnitudes[ipair] * dot(spins[ispin], spins[jspin]);
      energy[ispin] += E;
      energy[jspin] += E;
    }
  }

  // triplets
  for (std::size_t icell = 0; icell < geometry->n_cells_total; ++icell) {
    for (std::size_t itriplet = 0; itriplet < triplet_triplets.size();
         ++itriplet) {
      const auto &[i, j, k, d_j, d_k] = triplet_triplets[itriplet];
      int ispin = i + icell * geometry->atoms_per_cell;
      int jspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, j, d_j});
      int kspin = idx_from_pair(ispin, geometry->n_cells,
                                geometry->atoms_per_cell, {i, k, d_k});

      if ((jspin < 0) || (kspin < 0))
        continue;

      const scalar E = (1.0 / 3.0) * triplet_magnitudes[itriplet] *
                       dot(spins[ispin], cross(spins[jspin], spins[kspin]));
      energy[ispin] += E;
      energy[jspin] += E;
      energy[kspin] += E;
    }
  }

  // quadruplets
  for (std::size_t iquad = 0; iquad < quadruplet_quadruplets.size(); ++iquad) {
    const auto &[i, j, k, l, d_j, d_k, d_l] = quadruplet_quadruplets[iquad];

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

      const scalar E = 0.25 * quadruplet_magnitudes[iquad] *
                       dot(spins[ispin], spins[jspin]) *
                       dot(spins[kspin], spins[lspin]);
      energy[ispin] += E;
      energy[jspin] += E;
      energy[kspin] += E;
      energy[lspin] += E;
    }
  }
}

template <typename Interaction>
void Aggregator<SiteInteraction, PairInteraction, TripletInteraction,
                QuadrupletInteraction>::
    setParameters(const typename Interaction::input_data &parameters) {
  if constexpr (std::is_same_v<Interaction, SiteInteraction>) {
    site_indices = parameters.indices_;
    site_magnitudes = parameters.magnitude_;
    site_directions = parameters.direction_;
  } else if constexpr (std::is_same_v<Interaction, PairInteraction>) {
    pair_pairs = parameters.pairs_;
    pair_magnitudes = parameters.magnitude_;
  } else if constexpr (std::is_same_v<Interaction, TripletInteraction>) {
    triplet_triplets = parameters.triplets_;
    triplet_magnitudes = parameters.magnitude_;
  } else if constexpr (std::is_same_v<Interaction, QuadrupletInteraction>) {
    quadruplet_quadruplets = parameters.quadruplets_;
    quadruplet_magnitudes = parameters.magnitude_;
  } else {
    static_assert(false);
  }
}

template <typename Interaction>
void Aggregator<SiteInteraction, PairInteraction, TripletInteraction,
                QuadrupletInteraction>::
    updateParameters(const typename Interaction::update_data &parameters) {
  if constexpr (std::is_same_v<Interaction, SiteInteraction>) {
    site_magnitudes = parameters.magnitude_;
    site_directions = parameters.direction_;
  } else if constexpr (std::is_same_v<Interaction, PairInteraction>) {
    pair_magnitudes = parameters.magnitude_;
  } else if constexpr (std::is_same_v<Interaction, TripletInteraction>) {
    triplet_magnitudes = parameters.magnitude_;
  } else if constexpr (std::is_same_v<Interaction, QuadrupletInteraction>) {
    quadruplet_magnitudes = parameters.magnitude_;
  } else {
    static_assert(false);
  }
}

}; // namespace Monolithic

int main(const int argc, const char *argv[]) {
  using MonolithicAggregator = Monolithic::Aggregator<
      Monolithic::SiteInteraction, Monolithic::PairInteraction,
      Monolithic::TripletInteraction, Monolithic::QuadrupletInteraction>;
  return run_benchmark<MonolithicAggregator>(argc, argv);
}
