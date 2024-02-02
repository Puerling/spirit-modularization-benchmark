#pragma once
#include <array>
#include <cmath>
#include <memory>
#include <vector>

using tri_int = std::array<std::size_t, 3>;

struct Geometry {
  Geometry(tri_int n_cells, std::size_t atoms_per_cell)
      : n_cells_total{n_cells[0] * n_cells[1] * n_cells[2]},
        atoms_per_cell{atoms_per_cell}, nos{n_cells_total * atoms_per_cell},
        n_cells{n_cells} {};
  std::size_t n_cells_total;
  std::size_t atoms_per_cell;
  std::size_t nos;
  tri_int n_cells;
};

struct Owner {
  std::shared_ptr<Geometry> geometry;
};

class Owned {
protected:
  Owner *owner = nullptr;

public:
  void setOwner(Owner *new_owner) { owner = new_owner; }
};

using scalar = double;
template <typename T> using field = std::vector<T>;

struct Pair {
  int i, j;
  std::array<int, 3> translations;
};

struct Triplet {
  int i, j, k;
  std::array<int, 3> d_j, d_k;
};

struct Quadruplet {
  int i, j, k, l;
  std::array<int, 3> d_j, d_k, d_l;
};

using Vector3 = std::array<scalar, 3>;

using pairfield = field<Pair>;
using tripletfield = field<Triplet>;
using quadrupletfield = field<Quadruplet>;
using intfield = field<int>;
using scalarfield = field<scalar>;
using vectorfield = field<Vector3>;

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

inline scalar dot(const Vector3 &a, const Vector3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vector3 normalized(const Vector3 &a) {
  const scalar norm = std::sqrt(dot(a, a));
  if (norm > 1e-6)
    return Vector3{a[0] / norm, a[1] / norm, a[2] / norm};
  else
    return a;
}

inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
  return Vector3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0]};
}

inline int idx_from_translations(const tri_int &n_cells, const int n_cell_atoms,
                                 const std::array<int, 3> &translations) {
  const auto &Na = n_cells[0];
  const auto &Nb = n_cells[1];
  const auto &N = n_cell_atoms;

  const auto &da = translations[0];
  const auto &db = translations[1];
  const auto &dc = translations[2];

  return da * N + db * N * Na + dc * N * Na * Nb;
}

inline int idx_from_pair(const int ispin, const tri_int &n_cells, int N,
                         const Pair &pair, bool invert = false) {
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
