/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file PetscVector.hpp
 */

#ifndef LAI_PETSCVECTOR_HPP_
#define LAI_PETSCVECTOR_HPP_

#include "common/DataTypes.hpp"

#include <petscvec.h>

namespace geosx
{

inline PetscInt * toPetscInt( globalIndex * const index )
{
  return reinterpret_cast<PetscInt*>(index);
}

inline PetscInt const * toPetscInt( globalIndex const * const index )
{
  return reinterpret_cast<PetscInt const*>(index);
}

/**
 * \class PetscVector
 * \brief This class creates and provides basic support for Vec 
 *        vector object type used in PETSc.
 */
class PetscVector
{
  static_assert( sizeof(PetscInt)==sizeof(globalIndex), "sizeof(PetscInt) != sizeof(localIndex)");
  static_assert( std::is_same<PetscScalar, real64>::value, "PetscScalar != real64" );

 public:
  //! @name Constructor/Destructor Methods
  //@{

  /**
   * @brief Empty vector constructor.
   *
   * Create an empty (distributed) vector.
   * 
   */
  PetscVector();

  /**
   * @brief Copy constructor.
   *
   * \param vec Vector to be copied.
   *
   */
  PetscVector( PetscVector const & vec );

  /* Construct from Petsc vector */
  explicit PetscVector( Vec vec );

  /**
   * @brief Virtual destructor.
   */
  ~PetscVector();
  //@}

  //! @name Create Methods
  //@{

  /**
   * @brief Create a vector based on a previous vector.
   *
   * \param vector an already formed PetscVector.
   *
   */
  void create( PetscVector const & src );

  /**
   * @brief Create a vector based on local number of elements. Creates a CPU MPI vector.
   *
   * Create a vector based on local number of elements.  Global size is
   * the sum across processors.  For specifying a global size and having
   * automatic partitioning, see createGlobal().
   *
   * \param localSize local number of elements.
   *
   */
  void createWithLocalSize( localIndex const localSize, MPI_Comm const & comm = MPI_COMM_WORLD );

  /**
   * @brief Create a vector based on global number of elements. Creates a CPU MPI vector.
   *
   * Create a vector based on global number of elements. Every processors
   * gets the same number of local elements except proc 0, which gets any
   * remainder elements as well if the split can't be done evenly.
   *
   * \param globalSize Global number of elements.
   *
   */
  void createWithGlobalSize( globalIndex const globalSize, MPI_Comm const & comm = MPI_COMM_WORLD );

  /**
   * @brief Construct parallel vector from a local array. Creates a CPU MPI vector.
   *
   * Create a vector from local data, must assemble vector after use.
   *
   * \param localValues local data to put into vector
   *
   */
  void create( array1d<real64> const & localValues, MPI_Comm const & comm = MPI_COMM_WORLD );

  //@}
  //! @name Open / close
  //@{

  /**
   * @brief Empty function for PETSc implementation. May be required by other libraries.
   *
   */
  void open();

  /**
   * @brief Assemble vector
   *
   */
  void close();

  //@}
  //! @name Add/Set Methods
  //@

  /**
   * @brief Set vector value.
   *
   * Set vector value at given element.
   *
   * \param globalRow global row index
   * \param value Value to add at given row.
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void set( globalIndex const globalRow,
            real64 const value );

  /**
   * @brief Add into vector value.
   *
   * Add into vector value at given element.
   *
   * \param globalRow global row.
   * \param value Values to add in given row.
   * 
   * NOTE: set() and add() can't be interchanged without assembly.
   *
   */
  void add( globalIndex const globalRow,
            real64 const value );

  /**
   * @brief Set vector values.
   *
   * Set vector values at given elements.
   *
   * \param globalIndices global row indices.
   * \param values Values to add in given rows.
   * \param size Number of elements
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void set( globalIndex const * globalIndices,
            real64 const * values,
            localIndex size );

  /**
   * @brief Add vector values.
   *
   * Add vector values at given elements.
   *
   * \param globalIndices global row indices.
   * \param values Values to add in given rows.
   * \param size Number of elements
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void add( globalIndex const * globalIndices,
            real64 const * values,
            localIndex size );

  /**
   * @brief Set vector values using array1d
   *
   * Set vector values at given elements.
   *
   * \param globalIndices global row indices.
   * \param values Values to add in given rows.
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void set( array1d<globalIndex> const & globalIndices,
            array1d<real64> const & values );


  /**
   * @brief Add into vector values using array1d
   *
   * Add into vector values at given rows.
   *
   * \param globalIndices global rows indices
   * \param values Values to add in given rows.
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void add( array1d<globalIndex> const & globalIndices,
            array1d<real64> const & values );

  /**
   * @brief Set all elements to a constant value.
   *
   * \param value Values to set vector elements to.
   *
   * NOTE: set() and add() can't be interchanged without assembly.
   * 
   */
  void set( real64 const value );

  /**
   * @brief Set vector elements to zero.
   *
   */
  void zero();

  /**
   * @brief Set vector elements to random entries.
   *
   */
  void rand( unsigned long seed = 1984 );

  //@}

  //! @name Algebraic Operations
  //@{

  /**
   * @brief Multiply all elements by scalingFactor.
   *
   * \param scalingFactor Scaling Factor.
   *
   */
  void scale( real64 const scalingFactor );

  /**
   * @brief Dot product with the vector vec.
   *
   * \param vec EpetraVector to dot-product with.
   *
   */
  real64 dot( PetscVector const &vec );

  /**
   * @brief Update vector <tt>y</tt> as <tt>y</tt> = <tt>x</tt>.
   *
   * @note The naming convention follows the BLAS library.
   *
   * \param x PetscVector to copy.
   *
   */
  void copy(PetscVector const &x);

  /**
   * @brief Update vector <tt>y</tt> as <tt>y</tt> = <tt>alpha*x + y</tt>.
   *
   * @note The naming convention follows the logic of the BLAS library.
   *
   * \param alpha Scaling factor for added vector.
   * \param x Vector to add.
   *
   */
  void axpy( real64 const alpha,
             PetscVector const &x );

  /**
   * @brief Update vector <tt>y</tt> as <tt>y</tt> = <tt>alpha*x + beta*y</tt>.
   *
   * @note The naming convention follows the logic of the BLAS library.
   *
   * \param alpha Scaling factor for added vector.
   * \param x Vector to add.
   * \param beta Scaling factor for self vector.
   *
   */
  void axpby( real64 const alpha, 
              PetscVector &x, 
              real64 const beta);

  /**
   * @brief 1-norm of the vector.
   *
   */
  real64 norm1() const;

  /**
   * @brief 2-norm of the vector.
   *
   */
  real64 norm2() const;

  /**
   * @brief Infinity-norm of the vector.
   *
   */
  real64 normInf() const;

  //@}

  //! @name Accessor Methods
  //@{

  /**
   * @brief Returns the global of the vector.
   */
  globalIndex globalSize() const;

  /**
   * @brief Returns the local size of the vector.
   */
  localIndex localSize() const;

  /**
   * @brief Returns the index of the first global row owned by that processor.
   */
  globalIndex ilower() const;

  /**
   * @brief Returns the next index after last global row owned by that processor.
   *
   * @note The intention is for [ilower; iupper) to be used as a half-open index range
   */
  globalIndex iupper() const;

   /**
   * @brief Returns a single element. 
   * 
   * \param globalRow Global location of element to return
   */
  real64 get(globalIndex globalRow) const;

  /**
   * @brief Returns array of values at globalIndices of the vector.
   * 
   * \param globalIndices Global index array of local portion of the vector
   * \param values Array of values of local portion of the vector
   * 
   * NOTE: not yet implemented
   */
  void get( array1d<globalIndex> const & globalIndices,
            array1d<real64> & values ) const;

  /**
   * @brief Returns a const pointer to the underlying Vec.
   */
  const Vec* unwrappedPointer() const;

  /**
   * @brief Returns a non-const pointer to the underlying Vec.
   */
  Vec* unwrappedPointer();

  /**
   * @brief Returns underlying constant PETSc vector.
   */
  Vec getConstVec() const;

  /**
   * @brief Returns underying PETSc vector.
   */
  Vec getVec();

  /**
   * @brief Returns the matrix MPI communicator.
   */
  MPI_Comm getComm() const;

  //@}

  //! @name I/O Methods
  //@{

  /**
   * @brief Print the vector in PETSc format to the terminal.
   */
  void print() const;

  /**
   * @brief Write the vector to a matlab-compatible file
   * 
   * \param filename Name of output file
   * \param mtxFormat True if Matrix Market file format, false for Matlab
   */
  void write( string const & filename,
              bool const mtxFormat = true ) const;

  /**
   * Map a global row index to local row index. 
   * Error if requesting processor does not own row index. 
   * 
   * \param index Global index of row
   */
  localIndex getLocalRowID( globalIndex const index ) const;

  /**
   * Extract a view of the local portion of the array.
   * 
   * \param localVector Pointer to array of local values. Caller allocates memory. 
   */
  void extractLocalVector( real64 ** localVector ) const;

  //@}

 protected:
  
  // Underlying Petsc Vec
  Vec m_vec;
};

} // end geosx namespace

#endif /* LAI_PETSCVECTOR_HPP_ */