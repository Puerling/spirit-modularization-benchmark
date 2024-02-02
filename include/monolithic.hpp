#include "natives.hpp"

namespace Monolithic {

struct SiteInteraction {
  using input_data = SiteInputData;
  using update_data = SiteUpdateData;

  SiteInteraction(intfield indices, scalarfield c, vectorfield direction)
      : indices_{indices}, magnitude_{c}, direction_{direction} {};
  SiteInteraction(const input_data &data)
      : SiteInteraction(data.indices_, data.magnitude_, data.direction_){};

  intfield indices_;
  scalarfield magnitude_;
  vectorfield direction_;
};

struct PairInteraction {
  using input_data = PairInputData;
  using update_data = PairUpdateData;

  PairInteraction(pairfield pairs, scalarfield c)
      : pairs_{pairs}, magnitude_{c} {};
  PairInteraction(const input_data &data)
      : PairInteraction(data.pairs_, data.magnitude_){};

  pairfield pairs_;
  scalarfield magnitude_;
};

struct TripletInteraction {
  using input_data = TripletInputData;
  using update_data = TripletUpdateData;

  TripletInteraction(tripletfield triplet, scalarfield c)
      : triplets_{triplet}, magnitude_{c} {};
  TripletInteraction(const input_data &data)
      : TripletInteraction(data.triplets_, data.magnitude_){};

  tripletfield triplets_;
  scalarfield magnitude_;
};

struct QuadrupletInteraction {
  using input_data = QuadrupletInputData;
  using update_data = QuadrupletUpdateData;

  QuadrupletInteraction(quadrupletfield quad, scalarfield c)
      : quadruplets_{quad}, magnitude_{c} {};
  QuadrupletInteraction(const input_data &data)
      : QuadrupletInteraction(data.quadruplets_, data.magnitude_){};

  quadrupletfield quadruplets_;
  scalarfield magnitude_;
};

template <typename... Interactions> class Aggregator;

template <>
class Aggregator<SiteInteraction, PairInteraction, TripletInteraction,
                 QuadrupletInteraction> {
public:
  Aggregator(std::shared_ptr<Geometry> geometry, SiteInteraction site,
             PairInteraction pair, TripletInteraction triplet,
             QuadrupletInteraction quad)
      : geometry{geometry}, site_indices{site.indices_},
        site_magnitudes{site.magnitude_}, site_directions{site.direction_},
        pair_pairs{pair.pairs_}, pair_magnitudes{pair.magnitude_},
        triplet_triplets{triplet.triplets_},
        triplet_magnitudes{triplet.magnitude_},
        quadruplet_quadruplets{quad.quadruplets_},
        quadruplet_magnitudes{quad.magnitude_} {};

  void Energy(const vectorfield &spins, scalarfield &energy);

  void setGeometry(std::shared_ptr<Geometry> pGeometry) {
    geometry = pGeometry;
  };

  template <typename Interaction>
  void setParameters(const typename Interaction::input_data &parameters);

  template <typename Interaction>
  void updateParameters(const typename Interaction::update_data &parameters);

  template <std::size_t I>
  using interaction_t = std::tuple_element_t<
      I, std::tuple<SiteInteraction, PairInteraction, TripletInteraction,
                    QuadrupletInteraction>>;

private:
  std::shared_ptr<Geometry> geometry;
  intfield site_indices;
  scalarfield site_magnitudes;
  vectorfield site_directions;

  pairfield pair_pairs;
  scalarfield pair_magnitudes;

  tripletfield triplet_triplets;
  scalarfield triplet_magnitudes;

  quadrupletfield quadruplet_quadruplets;
  scalarfield quadruplet_magnitudes;
};

};
