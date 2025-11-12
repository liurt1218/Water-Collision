#ifndef ARRAY3D_H_
#define ARRAY3D_H_

#include <cassert>

// To disable assert*() calls, uncomment this line:
// #define NDEBUG

template <typename T>
class Array3D {
 public:
  // Allocates a 3D array with |nx| rows, |ny| columns, and depth |nz|.
  inline Array3D(std::size_t nx, std::size_t ny, std::size_t nz);

  // Sets |*this| = |other|.
  // We use this function instead of operator= to prevent any automatic calls to
  // operator= that may occur e.g. when returning an Array3D from a function and
  // assigning the returned Array3D to an already instantiated Array3D object.
  inline void SetEqualTo(const Array3D& other);

  // Deallocates the array.
  inline ~Array3D();

  std::size_t nx() const { return nx_; }
  std::size_t ny() const { return ny_; }
  std::size_t nz() const { return nz_; }

  // Returns the element at index (|i|, |j|, |k|) of this array.
  inline const T& operator()(std::size_t i, std::size_t j, std::size_t k) const;

  // Returns a modifiable reference to the element at index (|i|, |j|,
  // |k|) of this array.
  inline T& operator()(std::size_t i, std::size_t j, std::size_t k);

  // Sets all elements of this 3D array equal to |value|.
  inline void operator=(const T& value);

  // Adds |scalar| * |arr| to |*this|.
  // |arr| and |*this| must have identical dimensions.
  inline void PlusEquals(double scalar, const Array3D<T>& arr);

  // Sets |*this| = |arr1| + |scalar| * |arr2|.
  //
  // |arr1|, |arr2|, and |*this| must have identical dimensions.
  //
  // It's okay if |arr1| and/or |arr2| and/or |*this| are the same array since
  // each element will be modified one at a time, not affecting any other
  // element.
  inline void EqualsPlusTimes(const Array3D<T>& arr1, double scalar,
                              const Array3D<T>& arr2);

 private:
  // Don't allow copy constructor to be called from outside this class.
  Array3D(const Array3D& other);

  // Don't allow copy-assignment operator to be called from outside this class.
  Array3D& operator=(const Array3D& other);

  // Number of rows of data this array stores (x or i direction)
  const std::size_t nx_;

  // Number of columns of data this array stores (y or j direction)
  const std::size_t ny_;

  // Depth of data this array stores (z or k direction)
  const std::size_t nz_;

  // Size of a single stack of this array's data, for convenience
  const std::size_t ny_nz_;

  // The actual 3D data this array stores
  T* data_;
};

template <class T>
inline Array3D<T>::Array3D(std::size_t nx, std::size_t ny, std::size_t nz)
    : nx_(nx), ny_(ny), nz_(nz), ny_nz_(ny * nz), data_(new T[nx * ny_nz_]) {}

template <class T>
inline void Array3D<T>::SetEqualTo(const Array3D<T>& other) {
  T* data_pointer = data_;
  const T* other_data_pointer = other.data_;
  for (std::size_t i = 0; i < nx_ * ny_nz_; i++) {
    (*data_pointer) = (*other_data_pointer);
    data_pointer++;
    other_data_pointer++;
  }
}

template <class T>
inline Array3D<T>::~Array3D() {
  // The only constructor for the class instantiates |data_|, so there is
  // no need to check if |data_| is NULL before deleting it.
  delete[] data_;
}

template <class T>
inline const T& Array3D<T>::operator()(std::size_t i, std::size_t j,
                                       std::size_t k) const {
  return data_[i * ny_nz_ + j * nz_ + k];
}

template <class T>
inline T& Array3D<T>::operator()(std::size_t i, std::size_t j, std::size_t k) {
  return data_[i * ny_nz_ + j * nz_ + k];
}

template <class T>
inline void Array3D<T>::operator=(const T& value) {
  T* data_pointer = data_;
  for (std::size_t i = 0; i < nx_ * ny_nz_; i++) {
    (*data_pointer) = value;
    data_pointer++;
  }
}

template <class T>
inline void Array3D<T>::PlusEquals(double scalar, const Array3D<T>& arr) {
  // |arr| and |*this| must have identical dimensions.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      for (std::size_t k = 0; k < nz_; k++) {
        (*this)(i, j, k) += scalar * arr(i, j, k);
      }
    }
  }
}

template <class T>
inline void Array3D<T>::EqualsPlusTimes(const Array3D<T>& arr1, double scalar,
                                        const Array3D<T>& arr2) {
  // |arr1|, |arr2|, and |*this| must have identical dimensions.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      for (std::size_t k = 0; k < nz_; k++) {
        (*this)(i, j, k) = arr1(i, j, k) + scalar * arr2(i, j, k);
      }
    }
  }
}

// Returns the element-wise "dot product" of |a1| and |a2|.
// |a1| and |a2| must have identical dimensions.
inline double Dot(const Array3D<double>& a1, const Array3D<double>& a2) {
  double dot = 0.0;

  for (std::size_t i = 0; i < a1.nx(); i++) {
    for (std::size_t j = 0; j < a1.ny(); j++) {
      for (std::size_t k = 0; k < a1.nz(); k++) {
        dot += a1(i, j, k) * a2(i, j, k);
      }
    }
  }

  return dot;
}

#endif  // ARRAY3D_H_
