#pragma once
#include "natives.hpp"
#include <algorithm>
#include <cstdlib>
#include <execution>
#include <memory>
#include <optional>

namespace WriteLocality {

class SiteInteraction {
public:
  struct Data {
    Data() = delete;
    int i;
    scalar magnitude;
    Vector3 direction;
  };
  using input_data = SiteInputData;
  using update_data = SiteUpdateData;

  SiteInteraction(intfield indices, scalarfield magnitude,
                  vectorfield direction)
      : indices_{std::move(indices)}, magnitude_{std::move(magnitude)},
        direction_{std::move(direction)} {};
  SiteInteraction(const input_data &data)
      : SiteInteraction(data.indices_, data.magnitude_, data.direction_){};

  using data_t = std::optional<Data>;

  scalar Energy(const data_t &data_, const vectorfield &spins) const;

  template <typename DataVector>
  void applyParameters(const Geometry &geometry, DataVector &data) const;

  template <typename DataTuple> void clearData(DataTuple &data) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  intfield indices_;
  scalarfield magnitude_;
  vectorfield direction_;
};

class PairInteraction {
public:
  struct Data {
    int i, j;
    scalar magnitude;
  };
  using input_data = PairInputData;
  using update_data = PairUpdateData;

  PairInteraction(const pairfield &pairs, const scalarfield &magnitude)
      : pairs_{pairs}, magnitude_{magnitude} {};
  PairInteraction(const input_data &data)
      : PairInteraction(data.pairs_, data.magnitude_){};

  using data_t = field<Data>;

  scalar Energy(const data_t &data_, const vectorfield &spins) const;

  template <typename DataVector>
  void applyParameters(const Geometry &geometry, DataVector &data) const;

  template <typename DataTuple> void clearData(DataTuple &data) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  pairfield pairs_;
  scalarfield magnitude_;
};

class TripletInteraction {
public:
  struct Data {
    int i, j, k;
    scalar magnitude;
  };
  using input_data = TripletInputData;
  using update_data = TripletUpdateData;

  TripletInteraction(const tripletfield &tripltes, const scalarfield &magnitude)
      : triplets_{tripltes}, magnitude_{magnitude} {};
  TripletInteraction(const input_data &data)
      : TripletInteraction(data.triplets_, data.magnitude_){};

  using data_t = field<Data>;
  scalar Energy(const data_t &data_, const vectorfield &spins) const;

  template <typename DataVector>
  void applyParameters(const Geometry &geometry, DataVector &data) const;

  template <typename DataTuple> void clearData(DataTuple &element) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  tripletfield triplets_;
  scalarfield magnitude_;
};

class QuadrupletInteraction : public Owned {
public:
  struct Data {
    int i, j, k, l;
    scalar magnitude;
  };

  using input_data = QuadrupletInputData;
  using update_data = QuadrupletUpdateData;

  QuadrupletInteraction(quadrupletfield quad, scalarfield magnitude)
      : quadruplets_{std::move(quad)}, magnitude_{std::move(magnitude)} {};
  QuadrupletInteraction(const input_data &data)
      : QuadrupletInteraction(data.quadruplets_, data.magnitude_){};

  using data_t = field<Data>;

  scalar Energy(const data_t &data_, const vectorfield &spins) const;

  template <typename DataVector>
  void applyParameters(const Geometry &geometry, DataVector &data) const;

  template <typename DataTuple> void clearData(DataTuple &data) const;

  void setParameters(const input_data &parameters);

  void updateParameters(const update_data &parameters);

private:
  quadrupletfield quadruplets_;
  scalarfield magnitude_;
};

template <typename... Interactions> class Aggregator {
public:
  Aggregator(std::shared_ptr<Geometry> geometry, Interactions... interactions)
      : interactions_(interactions...), geometry{} {
    setGeometry(geometry);
  };

  void Energy(const vectorfield &spins, scalarfield &energy) {
    auto transform = [&spins,
                      &interactions = interactions_](const data_tuple_t &item) {
      return std::apply(
          [&spins, &item](const auto &...interaction) {
            return (
                interaction.Energy(
                    std::get<data_type<decltype(interaction)>>(item), spins) +
                ...);
          },
          interactions);
    };

    std::transform(std::execution::par, cbegin(data_), cend(data_),
                   begin(energy), transform);
  }

  void setGeometry(const std::shared_ptr<Geometry> g) {
    geometry = std::move(g);

    if (!geometry) {
      data_.clear();
      return;
    }

    if (data_.size() != geometry->nos) {
      data_ = field<data_tuple_t>(
          geometry->nos, std::make_tuple(typename Interactions::data_t{}...));
    } else {
      std::for_each(
          std::execution::par, begin(data_), end(data_),
          [&int_ = interactions_](auto &e) {
            std::apply([&e](const auto &...i) { (i.clearData(e), ...); }, int_);
          });
    }
    std::apply(
        [this](auto &...e) { (e.applyParameters(*geometry, data_), ...); },
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

    std::for_each(std::execution::par, begin(data_), end(data_),
                  [&interaction](auto &e) { interaction.clearData(e); });

    std::get<Interaction>(interactions_).applyParameters(*geometry, data_);
  }

  template <typename Interaction>
  void updateParameters(const typename Interaction::update_data &parameters) {
    if (!geometry)
      return;

    const auto &interaction = [this, &parameters]() {
      auto &interaction = std::get<Interaction>(interactions_);
      interaction.updateParameters(parameters);
      return interaction;
    }();

    std::for_each(std::execution::par, begin(data_), end(data_),
                  [&interaction](auto &e) { interaction.clearData(e); });

    std::get<Interaction>(interactions_).applyParameters(*geometry, data_);
  }

private:
  using data_tuple_t = std::tuple<typename Interactions::data_t...>;
  template <typename T> using data_type = typename std::decay_t<T>::data_t;

  std::tuple<Interactions...> interactions_;
  field<data_tuple_t> data_;
  std::shared_ptr<Geometry> geometry;

public:
  template <std::size_t I>
  using interaction_t = std::tuple_element_t<I, decltype(interactions_)>;
};

} // namespace WriteLocality
