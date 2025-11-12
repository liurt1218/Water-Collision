#include <iostream>

#include "Array3D.h"

// Sets some values in the |*table_ptr| 3D array, assumed to have be at least
// 2 x 4 x 5.
void SetValues(Array3D<int>* table_ptr) {
  (*table_ptr)(0, 0, 0) = 7;
  (*table_ptr)(1, 3, 4) = 12;
}

// Gets the value at indices (i, j, k) in a const |*table_ptr|. No bounds
// checking of indices is done.
int GetValue(const Array3D<int>* table_ptr, std::size_t i, std::size_t j,
             std::size_t k) {
  return (*table_ptr)(i, j, k);
}

// Reports whether all values in |*table_ptr| are -19.
void CheckValue(const Array3D<int>* table_ptr) {
  for (std::size_t i = 0; i < table_ptr->nx(); i++) {
    for (std::size_t j = 0; j < table_ptr->ny(); j++) {
      for (std::size_t k = 0; k < table_ptr->nz(); k++) {
        if ((*table_ptr)(i, j, k) != -19) {
          std::cout << "Element (" << i << ", " << j << ", " << k
                    << ") is not -19!" << std::endl;
          return;
        }
      }
    }
  }

  std::cout << "All elements were -19, as expected." << std::endl;
}

// Reports whether all values are as expected after calling PlusEquals(..).
void CheckPlusEquals(const Array3D<int>* table_ptr) {
  for (std::size_t i = 0; i < table_ptr->nx(); i++) {
    for (std::size_t j = 0; j < table_ptr->ny(); j++) {
      for (std::size_t k = 0; k < table_ptr->nz(); k++) {
        if ((*table_ptr)(i, j, k) != 190) {
          std::cout << "Element (" << i << ", " << j << ", " << k
                    << ") is not 190!" << std::endl;
          return;
        }
      }
    }
  }

  std::cout << "All elements were 190, as expected." << std::endl;
}

// Reports whether all values are as expected after calling EqualsPlusTimes(..).
void CheckEqualsPlusTimes(const Array3D<int>* table_ptr) {
  for (std::size_t i = 0; i < table_ptr->nx(); i++) {
    for (std::size_t j = 0; j < table_ptr->ny(); j++) {
      for (std::size_t k = 0; k < table_ptr->nz(); k++) {
        if ((*table_ptr)(i, j, k) != -380) {
          std::cout << "Element (" << i << ", " << j << ", " << k
                    << ") is not -380!" << std::endl;
          return;
        }
      }
    }
  }

  std::cout << "All elements were -380, as expected." << std::endl;
}

int main(int argc, char** argv) {
  // Create a 3 x 4 x 5 array of integers.
  Array3D<int> table(3, 4, 5);

  // Check that accessors for array dimensions are working.
  std::cout << "Array is " << table.nx() << " x " << table.ny() << " x "
            << table.nz() << "." << std::endl
            << std::endl;

  // Use the modifiable operator() of the Array3D class to set values of
  // some elements of the 3D array.
  SetValues(&table);

  // Use the const accessor operator() of the Array3D class to retrieve
  // the values of those same elements of the 3D array to verify that the
  // setting and getting of values worked.
  std::cout << "table(0, 0, 0) = " << GetValue(&table, 0, 0, 0) << std::endl;
  std::cout << "table(1, 3, 4) = " << GetValue(&table, 1, 3, 4) << std::endl;

  // Use the operator=(const T& value) method in the Array3D class to assign a
  // single value to every element of the array.
  table = -19;
  CheckValue(&table);

  // Use void Array3D<T>::PlusEquals(double, const Array3D<T>&).
  table.PlusEquals(-11, table);
  CheckPlusEquals(&table);

  // Use void Array3D<T>::EqualsPlusTimes(const Array3D<T>&, double, const
  // Array3D<T>&).
  table.EqualsPlusTimes(table, -3, table);
  CheckEqualsPlusTimes(&table);

  // Test copying one Array3D to another one with the same dimensions.
  Array3D<int> table2(3, 4, 5);
  table2.SetEqualTo(table);
  CheckEqualsPlusTimes(&table2);  // just checking that table2 == table now

  // Test computing dot product of two Array3Ds.
  Array3D<double> table3(3, 4, 5);
  Array3D<double> table4(3, 4, 5);
  table3 = -1;
  table4 = 7;
  double dot_product = Dot(table3, table4);
  if (dot_product == -7 * 3 * 4 * 5) {
    std::cout << "Dot product is correct!" << std::endl;
  } else {
    std::cout << "ERROR: Dot product is wrong!" << std::endl;
  }

  return 0;
}
