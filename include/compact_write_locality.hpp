#pragma once
#include "natives.hpp"
#include <algorithm>
#include <cstdlib>
#include <execution>
#include <memory>
#include <optional>

namespace CompactWriteLocality {

class SiteInteraction {
public:
  struct Index {
    std::size_t iatom;
    int ispin;
  };
  using input_data = SiteInputData;
  using update_data = SiteUpdateData;

  SiteInteraction(intfield indices, scalarfield magnitude,
                  vectorfield direction)
      : indices_{std::move(indices)}, magnitude_{std::move(magnitude)},
        direction_{std::move(direction)} {};
  SiteInteraction(const input_data &data)
      : SiteInteraction(data.indices_, data.magnitude_, data.direction_){};

  using index_t = std::optional<Index>;

  scalar Energy(const index_t &indices, const vectorfield &spins) const;

  template <typename IndexVector>
  void applyGeometry(const Geometry &geometry, IndexVector &indices) const;

  template <typename IndexTuple> void clearData(IndexTuple &indices) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  intfield indices_;
  scalarfield magnitude_;
  vectorfield direction_;
};

class PairInteraction {
public:
  struct Index {
    std::size_t ipair;
    int ispin, jspin;
  };
  using input_data = PairInputData;
  using update_data = PairUpdateData;

  PairInteraction(const pairfield &pairs, const scalarfield &magnitude)
      : pairs_{pairs}, magnitude_{magnitude} {};
  PairInteraction(const input_data &data)
      : PairInteraction(data.pairs_, data.magnitude_){};

  using index_t = field<Index>;

  scalar Energy(const index_t &indices, const vectorfield &spins) const;

  template <typename IndexVector>
  void applyGeometry(const Geometry &geometry, IndexVector &indices) const;

  template <typename IndexTuple> void clearData(IndexTuple &indices) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  pairfield pairs_;
  scalarfield magnitude_;
};

class TripletInteraction {
public:
  struct Index {
    std::size_t itriplet;
    int ispin, jspin, kspin;
  };
  using input_data = TripletInputData;
  using update_data = TripletUpdateData;

  TripletInteraction(const tripletfield &tripltes, const scalarfield &magnitude)
      : triplets_{tripltes}, magnitude_{magnitude} {};
  TripletInteraction(const input_data &data)
      : TripletInteraction(data.triplets_, data.magnitude_){};

  using index_t = field<Index>;
  scalar Energy(const index_t &indices, const vectorfield &spins) const;

  template <typename IndexVector>
  void applyGeometry(const Geometry &geometry, IndexVector &indices) const;

  template <typename IndexContainer>
  void clearData(IndexContainer &element) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  tripletfield triplets_;
  scalarfield magnitude_;
};

class QuadrupletInteraction {
public:
  struct Index {
    std::size_t iquad;
    int ispin, jspin, kspin, lspin;
  };

  using input_data = QuadrupletInputData;
  using update_data = QuadrupletUpdateData;

  QuadrupletInteraction(quadrupletfield quad, scalarfield magnitude)
      : quadruplets_{std::move(quad)}, magnitude_{std::move(magnitude)} {};
  QuadrupletInteraction(const input_data &data)
      : QuadrupletInteraction(data.quadruplets_, data.magnitude_){};

  using index_t = field<Index>;

  scalar Energy(const index_t &indices, const vectorfield &spins) const;

  template <typename IndexVector>
  void applyGeometry(const Geometry &geometry, IndexVector &indices) const;

  template <typename IndexTuple> void clearData(IndexTuple &indices) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  quadrupletfield quadruplets_;
  scalarfield magnitude_;
};

template <typename... Interactions> class Aggregator {
public:
  Aggregator(std::shared_ptr<Geometry> geometry, Interactions... interactions)
      : interactions_(interactions...), indices_{}, geometry{} {
    setGeometry(geometry);
  };

  void Energy(const vectorfield &spins, scalarfield &energy) {
    auto transform = [&spins, &interactions =
                                  interactions_](const index_tuple_t &item) {
      return std::apply(
          [&spins, &item](const auto &...interaction) {
            return (
                interaction.Energy(
                    std::get<index_type<decltype(interaction)>>(item), spins) +
                ...);
          },
          interactions);
    };

    std::transform(std::execution::par, cbegin(indices_), cend(indices_),
                   begin(energy), transform);
  }

  void setGeometry(std::shared_ptr<Geometry> g) {
    geometry = std::move(g);

    if (!geometry) {
      indices_.clear();
      return;
    }

    if (indices_.size() != geometry->nos) {
      indices_ = field<index_tuple_t>(
          geometry->nos, std::make_tuple(typename Interactions::index_t{}...));
    } else {
      std::for_each(
          std::execution::par, begin(indices_), end(indices_),
          [&int_ = interactions_](auto &e) {
            std::apply([&e](const auto &...i) { (i.clearData(e), ...); }, int_);
          });
    }
    std::apply(
        [this](auto &...e) { (e.applyGeometry(*geometry, indices_), ...); },
        interactions_);
  }

  template <typename Interaction>
  void setParameters(const typename Interaction::input_data &parameters) {
    if (!geometry)
      return;

    const auto &interaction = [this, &parameters]() {
      auto &interaction = std::get<Interaction>(interactions_);
      interaction.setParameters(parameters);
      return interaction;
    }();

    std::for_each(std::execution::par, begin(indices_), end(indices_),
                  [&interaction](auto &e) { interaction.clearData(e); });

    std::get<Interaction>(interactions_).applyGeometry(*geometry, indices_);
  }

  template <typename Interaction>
  void updateParameters(const typename Interaction::update_data &parameters) {
    std::get<Interaction>(interactions_).updateParameters(parameters);
  }

private:
  using index_tuple_t = std::tuple<typename Interactions::index_t...>;
  template <typename T> using index_type = typename std::decay_t<T>::index_t;

  std::tuple<Interactions...> interactions_;
  field<index_tuple_t> indices_;
  std::shared_ptr<Geometry> geometry;

public:
  template <std::size_t I>
  using interaction_t = std::tuple_element_t<I, decltype(interactions_)>;
};

} // namespace CompactWriteLocality
