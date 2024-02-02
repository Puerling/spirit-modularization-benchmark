#pragma once
#include "natives.hpp"
#include <algorithm>
#include <string_view>

namespace Polymorphic {

struct ABC : public Owned {
  virtual ~ABC() = default;

  void virtual Energy(const vectorfield &spins, scalarfield &energy) = 0;
  std::unique_ptr<ABC> virtual clone(Owner *new_owner) = 0;
  std::string_view virtual Name() = 0;
};

template <typename Derived> struct Base : public ABC {
  std::unique_ptr<ABC> clone(Owner *new_owner) final {
    auto copy = std::make_unique<Derived>(static_cast<const Derived &>(*this));
    copy->setOwner(new_owner);
    return copy;
  }
  std::string_view Name() final { return Derived::name; };
};

class SiteInteraction : public Base<SiteInteraction> {
public:
  using input_data = SiteInputData;
  using update_data = SiteUpdateData;

  SiteInteraction(intfield indices, scalarfield magnitude,
                  vectorfield direction)
      : indices_{std::move(indices)}, magnitude_{std::move(magnitude)},
        direction_{std::move(direction)} {};
  SiteInteraction(const input_data &data)
      : SiteInteraction(data.indices_, data.magnitude_, data.direction_){};

  void Energy(const vectorfield &spins, scalarfield &energy) final;

  void setParameters(const input_data &parameters);
  void updateParameters(const update_data &parameters);

  static constexpr std::string_view name = "SiteInteraction";

private:
  intfield indices_;
  scalarfield magnitude_;
  vectorfield direction_;
};

class PairInteraction : public Base<PairInteraction> {
public:
  using input_data = PairInputData;
  using update_data = PairUpdateData;

  PairInteraction(pairfield pairs, scalarfield magnitude)
      : pairs_{std::move(pairs)}, magnitude_{std::move(magnitude)} {};
  PairInteraction(const input_data &data)
      : PairInteraction(data.pairs_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy) final;

  void setParameters(const input_data &parameters);
  void updateParameters(const update_data &parameters);

  static constexpr std::string_view name = "PairInteraction";

private:
  pairfield pairs_;
  scalarfield magnitude_;
};

class TripletInteraction : public Base<TripletInteraction> {
public:
  using input_data = TripletInputData;
  using update_data = TripletUpdateData;

  TripletInteraction(tripletfield tripltes, scalarfield magnitude)
      : triplets_{std::move(tripltes)}, magnitude_{std::move(magnitude)} {};
  TripletInteraction(const input_data &data)
      : TripletInteraction(data.triplets_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy) final;

  void setParameters(const input_data &parameters);
  void updateParameters(const update_data &parameters);

  static constexpr std::string_view name = "TripletInteraction";

private:
  tripletfield triplets_;
  scalarfield magnitude_;
};

class QuadrupletInteraction : public Base<QuadrupletInteraction> {
public:
  using input_data = QuadrupletInputData;
  using update_data = QuadrupletUpdateData;

  QuadrupletInteraction(quadrupletfield quad, scalarfield magnitude)
      : quadruplets_{std::move(quad)}, magnitude_{std::move(magnitude)} {};
  QuadrupletInteraction(const input_data &data)
      : QuadrupletInteraction(data.quadruplets_, data.magnitude_){};

  void Energy(const vectorfield &spins, scalarfield &energy) final;

  void setParameters(const input_data &parameters);
  void updateParameters(const update_data &parameters);

  static constexpr std::string_view name = "QuadrupletInteraction";

private:
  quadrupletfield quadruplets_;
  scalarfield magnitude_;
};

template <typename... Interactions> class Aggregator : public Owner {
public:
  friend void swap(Aggregator &first, Aggregator &second) noexcept {
    using std::swap;
    if (&first == &second) {
      return;
    }

    swap(first.geometry, second.geometry);
    swap(first.interactions_, second.interactions_);
    for (auto &interaction : first.interactions_) {
      interaction->setOwner(&first);
    }
    for (auto &interaction : second.interactions_) {
      interaction->setOwner(&second);
    }
  };

  Aggregator(std::shared_ptr<Geometry> geometry, Interactions... interactions)
      : Owner{}, interactions_{} {
    setGeometry(geometry);

    (interactions_.emplace_back(std::make_unique<Interactions>(interactions)),
     ...);
    for (auto &interaction : interactions_) {
      interaction->setOwner(this);
    }
  };

  Aggregator() = default;
  Aggregator(const Aggregator &other) {
    for (auto &interaction : other.interactions_) {
      interactions_.emplace_back(interaction->clone(this));
    }
    setGeometry(other.geometry);
  }

  Aggregator &operator=(Aggregator other) {
    swap(*this, other);
    return *this;
  }

  Aggregator(Aggregator &&other) noexcept : Aggregator() { swap(*this, other); }

  Aggregator &operator=(Aggregator &&other) noexcept {
    swap(*this, other);
    return *this;
  }

  void Energy(const vectorfield &spins, scalarfield &energy) {
    for (auto &interaction : interactions_) {
      interaction->Energy(spins, energy);
    }
  }

  void setGeometry(std::shared_ptr<Geometry> pGeometry) {
    geometry = pGeometry;
  };

  template <typename Interaction>
  void setParameters(const typename Interaction::input_data &parameters) {
    auto it = std::find_if(
        begin(interactions_), end(interactions_),
        [](const auto &i) { return i->Name() == Interaction::name; });

    if (it != end(interactions_))
      dynamic_cast<Interaction *>((*it).get())->setParameters(parameters);
  }

  template <typename Interaction>
  void updateParameters(const typename Interaction::update_data &parameters) {
    auto it = std::find_if(
        begin(interactions_), end(interactions_),
        [](const auto &i) { return i->Name() == Interaction::name; });

    if (it != end(interactions_))
      dynamic_cast<Interaction *>((*it).get())->updateParameters(parameters);
  }

private:
  std::vector<std::unique_ptr<ABC>> interactions_;

public:
  template <std::size_t I>
  using interaction_t = std::tuple_element_t<I, std::tuple<Interactions...>>;
};

} // namespace Polymorphic
