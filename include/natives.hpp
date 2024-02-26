#pragma once
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

struct Geometry;

struct Owner {
  std::shared_ptr<Geometry> geometry;
};

class Owned {
protected:
  Owner *owner = nullptr;

public:
  void setOwner(Owner *new_owner) { owner = new_owner; }
};

#ifdef SPIRIT_USE_CUDA

#include "Managed_Allocator.hpp"
using scalar = float;
template <typename T> using field = std::vector<T, managed_allocator<T>>;

struct Pair {
  int i, j;
  int translations[3];

  template<typename... Array, typename = std::enable_if_t<sizeof...(Array) == 1>>
  static Pair create(int i, int j, Array&&... t){
    return Pair{i, j, {t[0], t[1], t[2]}...};
  }
};

struct Triplet {
  int i, j, k;
  int d_j[3], d_k[3];

  template<typename... Array, typename = std::enable_if_t<sizeof...(Array) == 2>>
  static Triplet create(int i, int j, int k, Array&&... t){
    return Triplet{i, j, k, {t[0], t[1], t[2]}...};
  }
};

struct Quadruplet {
  int i, j, k, l;
  int d_j[3], d_k[3], d_l[3];

  template<typename... Array, typename = std::enable_if_t<sizeof...(Array) == 3>>
  static Quadruplet create(int i, int j, int k, int l, Array&&... t){
    return Quadruplet{i, j, k, l, {t[0], t[1], t[2]}...};
  }
};

#else

using scalar = double;
template <typename T> using field = std::vector<T>;

struct Pair {
  int i, j;
  std::array<int, 3> translations;

  template<typename... Array, typename = std::enable_if<sizeof...(Array) == 3>>
  static Pair create(int i, int j, Array&&... t){
    return Pair{i, j, {t[0], t[1], t[2]}...};
  }
};

struct Triplet {
  int i, j, k;
  std::array<int, 3> d_j, d_k;

  template<typename... Array, typename = std::enable_if<sizeof...(Array) == 3>>
  static Triplet create(int i, int j, int k, Array&&... t){
    return Triplet{i, j, k, {t[0], t[1], t[2]}...};
  }
};

struct Quadruplet {
  int i, j, k, l;
  std::array<int, 3> d_j, d_k, d_l;

  template<typename... Array, typename = std::enable_if<sizeof...(Array) == 3>>
  static Quadruplet create(int i, int j, int k, int l, Array&&... t){
    return Quadruplet{i, j, k, l, {t[0], t[1], t[2]}...};
  }
};

#endif

using Vector3 = Eigen::Matrix<scalar, 3, 1>;

using pairfield = field<Pair>;
using tripletfield = field<Triplet>;
using quadrupletfield = field<Quadruplet>;
using intfield = field<int>;
using scalarfield = field<scalar>;
using vectorfield = field<Vector3>;
using sizefield = field<std::size_t>;

struct Geometry {
  Geometry(sizefield n_cells, std::size_t atoms_per_cell)
      : n_cells_total{n_cells[0] * n_cells[1] * n_cells[2]},
        atoms_per_cell{atoms_per_cell}, nos{n_cells_total * atoms_per_cell},
        n_cells{n_cells} {};
  std::size_t n_cells_total;
  std::size_t atoms_per_cell;
  std::size_t nos;
  sizefield n_cells;
};

struct SiteUpdateData {
  scalarfield magnitude_;
  vectorfield direction_;
};

struct SiteInputData : public SiteUpdateData {
  intfield indices_;
};

struct PairUpdateData {
  scalarfield magnitude_;
};

struct PairInputData : public PairUpdateData {
  pairfield pairs_;
};

struct TripletUpdateData {
  scalarfield magnitude_;
};

struct TripletInputData : public TripletUpdateData {
  tripletfield triplets_;
};

struct QuadrupletUpdateData {
  scalarfield magnitude_;
};

struct QuadrupletInputData : public QuadrupletUpdateData {
  quadrupletfield quadruplets_;
};

inline scalar dot(const Vector3 &a, const Vector3 &b) { return a.dot(b); }

inline Vector3 normalized(const Vector3 &a) { return a.normalized(); }

inline Vector3 cross(const Vector3 &a, const Vector3 &b) { return a.cross(b); }

#ifdef SPIRIT_USE_CUDA
__device__ __inline__ int idx_from_pair(const int ispin, const std::size_t n_cells[3],
                                    int N, const Pair &pair,
                                    bool invert = false)
#else
inline int idx_from_pair(const int ispin, const sizefield &n_cells, int N,
                         const Pair &pair, bool invert = false)
#endif
{
  // Invalid index if atom type of spin i is not correct
  if (pair.i != ispin % N)
    return -1;

  // Number of cells
  const int &Na = n_cells[0];
  const int &Nb = n_cells[1];
  const int &Nc = n_cells[2];

  // Invalid index if translations reach out over the lattice bounds
  if (std::abs(pair.translations[0]) > Na ||
      std::abs(pair.translations[1]) > Nb ||
      std::abs(pair.translations[2]) > Nc)
    return -1;

  // Translations (cell) of spin i
  const int nic = ispin / (N * Na * Nb);
  const int nib = (ispin - nic * N * Na * Nb) / (N * Na);
  const int nia = (ispin - nic * N * Na * Nb - nib * N * Na) / N;

  const int pm = invert ? -1 : 1;
  // Translations (cell) of spin j (possibly outside of non-periodical domain)
  int nja = nia + pm * pair.translations[0];
  int njb = nib + pm * pair.translations[1];
  int njc = nic + pm * pair.translations[2];

  // Check boundary conditions: a
  if ((0 <= nja && nja < Na)) {
    // Boundary conditions fulfilled
    // Find the translations of spin j within the non-periodical domain
    if (nja < 0)
      nja += Na;
    // Calculate the correct index
    if (nja >= Na)
      nja -= Na;
  } else {
    // Boundary conditions not fulfilled
    return -1;
  }

  // Check boundary conditions: b
  if ((0 <= njb && njb < Nb)) {
    // Boundary conditions fulfilled
    // Find the translations of spin j within the non-periodical domain
    if (njb < 0)
      njb += Nb;
    // Calculate the correct index
    if (njb >= Nb)
      njb -= Nb;
  } else {
    // Boundary conditions not fulfilled
    return -1;
  }

  // Check boundary conditions: c
  if ((0 <= njc && njc < Nc)) {
    // Boundary conditions fulfilled
    // Find the translations of spin j within the non-periodical domain
    if (njc < 0)
      njc += Nc;
    // Calculate the correct index
    if (njc >= Nc)
      njc -= Nc;
  } else {
    // Boundary conditions not fulfilled
    return -1;
  }

  // Return the index of spin j according to it's translations
  return pair.j + (nja)*N + (njb)*N * Na + (njc)*N * Na * Nb;
}
