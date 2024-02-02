#pragma once
#include "natives.hpp"

namespace ReadLocality {

class SiteInteraction : public Owned {
public:
  using input_data = SiteInputData;
  using update_data = SiteUpdateData;

  SiteInteraction(intfield indices, scalarfield magnitude,
                  vectorfield direction)
      : indices_{std::move(indices)}, magnitude_{std::move(magnitude)},
        direction_{std::move(direction)} {};
  SiteInteraction(const input_data &data)
      : SiteInteraction(data.indices_, data.magnitude_, data.direction_){};

  void Energy(const vectorfield &spins, scalarfield &energy);

  void setParameters(const input_data& parameters);
  void updateParameters(const update_data& parameters);

private:
  intfield indices_;
  scalarfield magnitude_;
  vectorfield direction_;

  void updateParametersImpl(intfield indices, scalarfield magnitude,
                            vectorfield direction);
};

class PairInteraction : public Owned {
public:
  using input_data = PairInputData;
  using update_data = PairUpdateData;

  PairInteraction(pairfield pairs, scalarfield magnitude)
      : pairs_{std::move(pairs)}, magnitude_{std::move(magnitude)} {};
  PairInteraction(const input_data &data)
      : PairInteraction(data.pairs_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy);

  void setParameters(const input_data& parameters);
  void updateParameters(const update_data& parameters);

private:
  pairfield pairs_;
  scalarfield magnitude_;
};

class TripletInteraction : public Owned {
public:
  using input_data = TripletInputData;
  using update_data = TripletUpdateData;

  TripletInteraction(tripletfield triplets, scalarfield magnitude)
      : triplets_{std::move(triplets)}, magnitude_{std::move(magnitude)} {};
  TripletInteraction(const input_data &data)
      : TripletInteraction(data.triplets_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy);

  void setParameters(const input_data& parameters);
  void updateParameters(const update_data& parameters);

private:
  tripletfield triplets_;
  scalarfield magnitude_;
};

class QuadrupletInteraction : public Owned {
public:
  using input_data = QuadrupletInputData;
  using update_data = QuadrupletUpdateData;

  QuadrupletInteraction(quadrupletfield quad, scalarfield magnitude)
      : quadruplets_{std::move(quad)}, magnitude_{std::move(magnitude)} {};
  QuadrupletInteraction(const input_data &data)
      : QuadrupletInteraction(data.quadruplets_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy);

  void setParameters(const input_data& parameters);
  void updateParameters(const update_data& parameters);

private:
  quadrupletfield quadruplets_;
  scalarfield magnitude_;
};

template <typename... Interactions> class Aggregator : public Owner {
public:
  Aggregator(std::shared_ptr<Geometry> geometry, Interactions... interactions)
      : Owner{}, interactions_(interactions...) {
    std::apply([this](auto &...e) { (e.setOwner(this), ...); }, interactions_);
    setGeometry(geometry);
  };

  void Energy(const vectorfield &spins, scalarfield &energy) {
    std::apply(
        [&spins, &energy](auto &...e) { (e.Energy(spins, energy), ...); },
        interactions_);
  }

  void setGeometry(std::shared_ptr<Geometry> pGeometry) {
    geometry = pGeometry;
  };

  template <typename Interaction>
  void setParameters(const typename Interaction::input_data &parameters) {
    std::get<Interaction>(interactions_).setParameters(parameters);
  }

  template <typename Interaction>
  void updateParameters(const typename Interaction::update_data &parameters) {
    std::get<Interaction>(interactions_).updateParameters(parameters);
  }

private:
  std::tuple<Interactions...> interactions_;

public:
  template <std::size_t I>
  using interaction_t = std::tuple_element_t<I, decltype(interactions_)>;
};

} // namespace ReadLocality
