#ifndef ARRAY3D_H_
#define ARRAY3D_H_

template <typename T>
class Array3D {
 public:
  // Allocates a 3D array with |nx| rows, |ny| columns, and depth |nz|.
  inline Array3D(std::size_t nx, std::size_t ny, std::size_t nz);

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

 private:
  // Don't allow copy constructor to be called.
  Array3D(const Array3D& other);

  // Don't allow copy-assignment operator to be called.
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

#endif  // ARRAY3D_H_
