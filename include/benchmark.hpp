#pragma once
#include "natives.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <nanobench/nanobench.h>
#include <numeric>
#include <random>
#include <utility>

namespace {

std::shared_ptr<Geometry> make_geometry(tri_int n_cells = {5, 5, 5},
                                        int atoms_per_cell = 4) {
  return std::make_shared<Geometry>(n_cells, atoms_per_cell);
};

template <typename RandomFunc>
Vector3 make_random_normal_vector(RandomFunc &rng) {
  static std::uniform_real_distribution<> dist(-1.0, 1.0);
  return normalized(std::array{dist(rng), dist(rng), dist(rng)});
};

template <typename RandomFunc>
void generate_random_normal_vectorfield(vectorfield::iterator begin,
                                        vectorfield::iterator end,
                                        RandomFunc &rng) {
  std::generate(begin, end,
                [&rng]() { return make_random_normal_vector(rng); });
};

template <typename RandomFunc>
vectorfield make_random_normal_vectorfield(const std::size_t n,
                                           RandomFunc &rng) {
  vectorfield output(n);
  generate_random_normal_vectorfield(begin(output), end(output), rng);
  return output;
};

template <typename RandomFunc>
scalarfield make_random_scalarfield(const std::size_t n, RandomFunc &rng) {
  static std::uniform_real_distribution<> dist(-1.0, 1.0);
  scalarfield output(n);
  std::generate_n(begin(output), n, [&rng]() { return dist(rng); });
  return output;
};

template <typename RandomFunc>
std::array<int, 3> make_random_translation(RandomFunc &rng) {
  static std::uniform_int_distribution<> dist(0, 2);
  return {dist(rng), dist(rng), dist(rng)};
};

template <typename Interaction, typename... Args>
Interaction make_interaction_impl(std::shared_ptr<Geometry> geometry,
                                  Args &&...args) {
  if constexpr (std::is_constructible_v<Interaction, std::shared_ptr<Geometry>,
                                        Args...>)
    return Interaction(geometry, std::forward<Args>(args)...);
  else if constexpr (std::is_constructible_v<Interaction, Args...>)
    return Interaction(std::forward<Args>(args)...);
  else
    static_assert(false, "No matching constructor for interaction type");
};

template <typename RandomFunc>
SiteUpdateData make_site_update_params(const std::size_t n, RandomFunc &rng) {
  return SiteUpdateData{make_random_scalarfield(n, rng),
                        make_random_normal_vectorfield(n, rng)};
}

template <typename RandomFunc>
SiteInputData make_site_input_params(const std::size_t atoms_per_cell,
                                     const std::size_t n, RandomFunc &rng) {
  const auto n_max = std::min(n, atoms_per_cell);
  SiteInputData output{make_site_update_params(n_max, rng), intfield(n_max)};
  intfield input{};
  std::generate_n(std::back_inserter(input), atoms_per_cell,
                  [i = 0]() mutable -> int { return i++; });
  std::sample(begin(input), end(input), begin(output.indices_), n_max, rng);
  return output;
}

template <typename T, typename RandomFunc>
T make_site_interaction(std::shared_ptr<Geometry> geometry, const std::size_t n,
                        RandomFunc &rng) {
  return make_interaction_impl<T>(
      geometry, make_site_input_params(geometry->atoms_per_cell, n, rng));
}

template <typename RandomFunc>
PairUpdateData make_pair_update_params(const std::size_t n, RandomFunc &rng) {
  return PairUpdateData{make_random_scalarfield(n, rng)};
}

template <typename RandomFunc>
PairInputData make_pair_input_params(const std::size_t atoms_per_cell,
                                     const std::size_t n, RandomFunc &rng) {
  std::uniform_int_distribution<> dist(0, atoms_per_cell - 1);
  PairInputData output{make_pair_update_params(n, rng), pairfield(n)};
  std::generate(begin(output.pairs_), end(output.pairs_), [&dist, &rng]() {
    return Pair{dist(rng), dist(rng), make_random_translation(rng)};
  });
  return output;
}

template <typename T, typename RandomFunc>
T make_pair_interaction(std::shared_ptr<Geometry> geometry, const std::size_t n,
                        RandomFunc &rng) {
  return make_interaction_impl<T>(
      geometry, make_pair_input_params(geometry->atoms_per_cell, n, rng));
}

template <typename RandomFunc>
TripletUpdateData make_triplet_update_params(const std::size_t n,
                                             RandomFunc &rng) {
  return TripletUpdateData{make_random_scalarfield(n, rng)};
}

template <typename RandomFunc>
TripletInputData make_triplet_input_params(const std::size_t atoms_per_cell,
                                           const std::size_t n,
                                           RandomFunc &rng) {
  std::uniform_int_distribution<> dist(0, atoms_per_cell - 1);
  TripletInputData output{make_triplet_update_params(n, rng), tripletfield(n)};
  std::generate(begin(output.triplets_), end(output.triplets_),
                [&dist, &rng]() {
                  return Triplet{dist(rng), dist(rng), dist(rng),
                                 make_random_translation(rng),
                                 make_random_translation(rng)};
                });
  return output;
}

template <typename T, typename RandomFunc>
T make_triplet_interaction(std::shared_ptr<Geometry> geometry,
                           const std::size_t n, RandomFunc &rng) {
  return make_interaction_impl<T>(
      geometry, make_triplet_input_params(geometry->atoms_per_cell, n, rng));
}

template <typename RandomFunc>
QuadrupletUpdateData make_quadruplet_update_params(const std::size_t n,
                                                   RandomFunc &rng) {
  return QuadrupletUpdateData{make_random_scalarfield(n, rng)};
}

template <typename RandomFunc>
QuadrupletInputData
make_quadruplet_input_params(const std::size_t atoms_per_cell,
                             const std::size_t n, RandomFunc &rng) {
  std::uniform_int_distribution<> dist(0, atoms_per_cell - 1);
  QuadrupletInputData output{make_quadruplet_update_params(n, rng),
                             quadrupletfield(n)};
  std::generate(begin(output.quadruplets_), end(output.quadruplets_),
                [&dist, &rng]() {
                  return Quadruplet{dist(rng),
                                    dist(rng),
                                    dist(rng),
                                    dist(rng),
                                    make_random_translation(rng),
                                    make_random_translation(rng),
                                    make_random_translation(rng)};
                });
  return output;
}

template <typename T, typename RandomFunc>
T make_quadruplet_interaction(std::shared_ptr<Geometry> geometry,
                              const std::size_t n, RandomFunc &rng) {
  return make_interaction_impl<T>(
      geometry, make_quadruplet_input_params(geometry->atoms_per_cell, n, rng));
}

template <typename Agg, std::size_t I>
using interaction_t = typename Agg::template interaction_t<I>;

template <typename Agg, typename Interaction, typename RandomFunc>
typename Interaction::update_data make_update_params(const std::size_t n,
                                                     RandomFunc &rng) {
  if constexpr (std::is_same_v<interaction_t<Agg, 0>, Interaction>)
    return make_site_update_params<RandomFunc>(n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 1>, Interaction>)
    return make_pair_update_params<RandomFunc>(n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 2>, Interaction>)
    return make_triplet_update_params<RandomFunc>(n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 3>, Interaction>)
    return make_quadruplet_update_params<RandomFunc>(n, rng);
  else
    static_assert(false);
};

template <typename Agg, typename Interaction, typename RandomFunc>
typename Interaction::input_data
make_input_params(const std::size_t atoms_per_cell, const std::size_t n,
                  RandomFunc &rng) {
  if constexpr (std::is_same_v<interaction_t<Agg, 0>, Interaction>)
    return make_site_input_params<RandomFunc>(atoms_per_cell, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 1>, Interaction>)
    return make_pair_input_params<RandomFunc>(atoms_per_cell, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 2>, Interaction>)
    return make_triplet_input_params<RandomFunc>(atoms_per_cell, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 3>, Interaction>)
    return make_quadruplet_input_params<RandomFunc>(atoms_per_cell, n, rng);
  else
    static_assert(false);
};

template <typename Agg, typename Interaction, typename RandomFunc>
Interaction make_interaction(std::shared_ptr<Geometry> geometry,
                             const std::size_t n, RandomFunc &rng) {
  if constexpr (std::is_same_v<interaction_t<Agg, 0>, Interaction>)
    return make_site_interaction<Interaction, RandomFunc>(geometry, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 1>, Interaction>)
    return make_pair_interaction<Interaction, RandomFunc>(geometry, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 2>, Interaction>)
    return make_triplet_interaction<Interaction, RandomFunc>(geometry, n, rng);
  else if constexpr (std::is_same_v<interaction_t<Agg, 3>, Interaction>)
    return make_quadruplet_interaction<Interaction, RandomFunc>(geometry, n,
                                                                rng);
  else
    static_assert(false);
};

template <typename Agg, typename RandomFunc>
Agg make_aggregator(std::shared_ptr<Geometry> geometry,
                    const std::array<std::size_t, 4> n, RandomFunc &rng) {
  return Agg(geometry,
             make_interaction<Agg, interaction_t<Agg, 0>>(geometry, n[0], rng),
             make_interaction<Agg, interaction_t<Agg, 1>>(geometry, n[1], rng),
             make_interaction<Agg, interaction_t<Agg, 2>>(geometry, n[2], rng),
             make_interaction<Agg, interaction_t<Agg, 3>>(geometry, n[3], rng));
};

template <typename Agg>
using update_parameter_set =
    std::tuple<typename interaction_t<Agg, 0>::update_data,
               typename interaction_t<Agg, 1>::update_data,
               typename interaction_t<Agg, 2>::update_data,
               typename interaction_t<Agg, 3>::update_data>;

template <typename Agg, typename RandomFunc>
update_parameter_set<Agg>
make_update_parameter_set(const std::array<std::size_t, 4> n, RandomFunc &rng) {
  return std::make_tuple(
      make_update_params<Agg, interaction_t<Agg, 0>>(n[0], rng),
      make_update_params<Agg, interaction_t<Agg, 1>>(n[1], rng),
      make_update_params<Agg, interaction_t<Agg, 2>>(n[2], rng),
      make_update_params<Agg, interaction_t<Agg, 3>>(n[3], rng));
}

template <typename Agg, std::size_t I = 0>
void updateParameters(Agg &aggregator,
                      const update_parameter_set<Agg> &params) {
  if constexpr (I < std::tuple_size_v<update_parameter_set<Agg>>) {
    aggregator.template updateParameters<interaction_t<Agg, I>>(
        std::get<I>(params));
    updateParameters<Agg, I + 1>(aggregator, params);
  }
}

template <typename RandomFunc>
vectorfield make_spins(const std::size_t nos, RandomFunc &rng) {
  return make_random_normal_vectorfield(nos, rng);
};

template <typename RandomFunc>
void generate_spins(vectorfield::iterator begin, vectorfield::iterator end,
                    RandomFunc &rng) {
  generate_random_normal_vectorfield(begin, end, rng);
};

template <typename Agg> int run_benchmark(const int argc, const char *argv[]) {
  std::array<std::size_t, 4> n_interactions{3, 5, 5, 5};
  std::mt19937 rng(1293175);

  auto geometry = make_geometry({25, 25, 25}, 10);
  if (!geometry) {
    fprintf(stderr, "No Geometry created\n");
    return EXIT_FAILURE;
  }

  auto agg = make_aggregator<Agg>(geometry, n_interactions, rng);
  vectorfield spins = make_spins(geometry->nos, rng);
  scalarfield energy(geometry->nos, 0.0);

  generate_spins(begin(spins), end(spins), rng);
  std::fill(begin(energy), end(energy), 0.0);

  ankerl::nanobench::Bench().minEpochIterations(10).run(
      "Energy", [&]() { agg.Energy(spins, energy); });

  ankerl::nanobench::Bench().minEpochIterations(10).run("Update", [&]() {
    updateParameters<Agg>(agg,
                          make_update_parameter_set<Agg>(n_interactions, rng));
  });

  return EXIT_SUCCESS;
};

} // namespace
