/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2019 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2019 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All right reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file HypreKernels.hpp
 */
#ifndef GEOSX_LINEARALGEBRA_INTERFACES_HYPREKERNELS_HPP_
#define GEOSX_LINEARALGEBRA_INTERFACES_HYPREKERNELS_HPP_

#include "common/DataTypes.hpp"
#include "common/GEOS_RAJA_Interface.hpp"
#include "linearAlgebra/interfaces/hypre/HypreUtils.hpp"

#include <_hypre_parcsr_mv.h>

namespace geosx
{

namespace hypre
{

inline void scaleMatrixValues( hypre_CSRMatrix const * const mat,
                               real64 const factor )
{
  if( isEqual( factor, 1.0 ) )
  {
    return;
  }
  HYPRE_Real * const va = hypre_CSRMatrixData( mat );
  forAll< hypre::execPolicy >( hypre_CSRMatrixNumNonzeros( mat ), [=] GEOSX_HYPRE_HOST_DEVICE ( HYPRE_Int const i )
  {
    va[i] *= factor;
  } );
}

inline void scaleMatrixRows( hypre_CSRMatrix const * const mat,
                             hypre_Vector const * const vec )
{
  HYPRE_Int const * const ia = hypre_CSRMatrixI( mat );
  HYPRE_Real * const va = hypre_CSRMatrixData( mat );
  HYPRE_Real const * const scalingFactors = hypre_VectorData( vec );

  forAll< hypre::execPolicy >( hypre_CSRMatrixNumRows( mat ), [=] GEOSX_HYPRE_HOST_DEVICE ( HYPRE_Int const localRow )
  {
    real64 const factor = scalingFactors[localRow];
    if( !isEqual( factor, 1.0 ) )
    {
      for( HYPRE_Int j = ia[localRow]; j < ia[localRow + 1]; ++j )
      {
        va[j] *= factor;
      }
    }
  } );
}

template< typename F, typename R >
inline void rescaleMatrixRows( hypre_ParCSRMatrix * const mat,
                               arrayView1d< globalIndex const > const & rowIndices,
                               F transform,
                               R reduce )
{
  hypre_CSRMatrix const * const csr_diag = hypre_ParCSRMatrixDiag( mat );
  HYPRE_Int const * const ia_diag = hypre_CSRMatrixI( csr_diag );
  HYPRE_Real * const va_diag = hypre_CSRMatrixData( csr_diag );

  hypre_CSRMatrix const * const csr_offd = hypre_ParCSRMatrixOffd( mat );
  HYPRE_Int const * const ia_offd = hypre_CSRMatrixI( csr_offd );
  HYPRE_Real * const va_offd = hypre_CSRMatrixData( csr_offd );

  HYPRE_BigInt const firstLocalRow = hypre_ParCSRMatrixFirstRowIndex( mat );

  auto const reducer = [=] GEOSX_HYPRE_HOST_DEVICE ( double acc, double v ){ return reduce( acc, transform( v ) ); };
  forAll< hypre::execPolicy >( rowIndices.size(), [=] GEOSX_HYPRE_HOST_DEVICE ( localIndex const i )
  {
    HYPRE_Int const localRow = LvArray::integerConversion< HYPRE_Int >( rowIndices[i] - firstLocalRow );
    GEOSX_ASSERT( 0 <= localRow && localRow < hypre_CSRMatrixNumRows( csr_diag ) );

    HYPRE_Real scale = 0.0;
    for( HYPRE_Int k = ia_diag[localRow]; k < ia_diag[localRow + 1]; ++k )
    {
      scale = reducer( scale, va_diag[k] );
    }
    if( hypre_CSRMatrixNumCols( csr_offd ) > 0 )
    {
      for( HYPRE_Int k = ia_offd[localRow]; k < ia_offd[localRow + 1]; ++k )
      {
        scale = reducer( scale, va_offd[k] );
      }
    }

    for( HYPRE_Int k = ia_diag[localRow]; k < ia_diag[localRow + 1]; ++k )
    {
      va_diag[k] /= scale;
    }
    if( hypre_CSRMatrixNumCols( csr_offd ) > 0 )
    {
      for( HYPRE_Int k = ia_offd[localRow]; k < ia_offd[localRow + 1]; ++k )
      {
        va_offd[k] /= scale;
      }
    }
  } );
}

template< typename T, typename INDEX >
void
GEOSX_HYPRE_HOST_DEVICE
makeSortedPermutation( T const * const values,
                       INDEX const size,
                       INDEX * const perm )
{
  for( INDEX i = 0; i < size; ++i )
  {
    perm[i] = i; // std::iota
  }
  auto const comp = [values] GEOSX_HYPRE_HOST_DEVICE ( INDEX const i, INDEX const j ) { return values[i] < values[j]; };
  LvArray::sortedArrayManipulation::makeSorted( perm, perm + size, comp );
}

inline void addEntriesRestricted( hypre_CSRMatrix const * const src,
                                  hypre_CSRMatrix const * const dst,
                                  real64 const scale )
{
  GEOSX_LAI_ASSERT( src != nullptr );
  GEOSX_LAI_ASSERT( dst != nullptr );
  GEOSX_LAI_ASSERT_EQ( hypre_CSRMatrixNumRows( src ), hypre_CSRMatrixNumRows( dst ) );

  if( isZero( scale ) )
  {
    return;
  }

  HYPRE_Int const * const src_ia  = hypre_CSRMatrixI( src );
  HYPRE_Int const * const src_ja  = hypre_CSRMatrixJ( src );
  HYPRE_Real const * const src_va = hypre_CSRMatrixData( src );

  HYPRE_Int const * const dst_ia  = hypre_CSRMatrixI( dst );
  HYPRE_Int const * const dst_ja  = hypre_CSRMatrixJ( dst );
  HYPRE_Real * const dst_va       = hypre_CSRMatrixData( dst );

  // Allocate contiguous memory to store sorted column permutations of each row
  array1d< HYPRE_Int > const src_permutation( hypre_CSRMatrixNumNonzeros( src ) );
  array1d< HYPRE_Int > const dst_permutation( hypre_CSRMatrixNumNonzeros( dst ) );

  // Each thread adds one row of src into dst
  forAll< hypre::execPolicy >( hypre_CSRMatrixNumRows( dst ),
                               [=,
                                src_permutation = src_permutation.toView(),
                                dst_permutation = dst_permutation.toView()] GEOSX_HYPRE_HOST_DEVICE ( HYPRE_Int const localRow )
  {
    HYPRE_Int const src_offset = src_ia[localRow];
    HYPRE_Int const src_length = src_ia[localRow + 1] - src_offset;
    HYPRE_Int const * const src_indices = src_ja + src_offset;
    HYPRE_Real const * const src_values = src_va + src_offset;
    HYPRE_Int * const src_perm = src_permutation.data() + src_offset;

    HYPRE_Int const dst_offset = dst_ia[localRow];
    HYPRE_Int const dst_length = dst_ia[localRow + 1] - dst_offset;
    HYPRE_Int const * const dst_indices = dst_ja + dst_offset;
    HYPRE_Real * const dst_values = dst_va + dst_offset;
    HYPRE_Int * const dst_perm = dst_permutation.data() + dst_offset;

    // Since hypre does not store columns in sorted order, create a sorted "view" of src and dst rows
    // TODO: it would be nice to cache the permutation arrays somewhere to avoid recomputing
    makeSortedPermutation( src_indices, src_length, src_perm );
    makeSortedPermutation( dst_indices, dst_length, dst_perm );

    // Add entries looping through them in sorted column order, skipping src entries not in dst
    for( HYPRE_Int i = 0, j = 0; i < dst_length && j < src_length; ++i )
    {
      while( j < src_length && src_indices[src_perm[j]] < dst_indices[dst_perm[i]] ) ++j;
      if( j < src_length && src_indices[src_perm[j]] == dst_indices[dst_perm[i]] )
      {
        dst_values[dst_perm[i]] += scale * src_values[src_perm[j++]];
      }
    }
  } );
}

template< bool SKIP_DIAG >
void clampMatrixEntries( hypre_CSRMatrix const * const mat,
                         real64 const lo,
                         real64 const hi )
{
  HYPRE_Int const * const ia = hypre_CSRMatrixI( mat );
  HYPRE_Real * const va = hypre_CSRMatrixData( mat );

  forAll< hypre::execPolicy >( hypre_CSRMatrixNumRows( mat ), [=] GEOSX_HYPRE_HOST_DEVICE ( localIndex const localRow )
  {
    // Hypre stores diagonal element at the beginning of each row, we assume it's always present
    for( HYPRE_Int k = ia[localRow] + SKIP_DIAG; k < ia[localRow+1]; ++k )
    {
      va[k] = LvArray::math::min( hi, LvArray::math::max( lo, va[k] ) );
    }
  } );
}

template< typename F, typename R >
inline void computeRowsSums( hypre_ParCSRMatrix const * const mat,
                             hypre_ParVector * const vec,
                             F transform,
                             R reduce )
{
  hypre_CSRMatrix const * const csr_diag = hypre_ParCSRMatrixDiag( mat );
  HYPRE_Int const * const ia_diag = hypre_CSRMatrixI( csr_diag );
  HYPRE_Real const * const va_diag = hypre_CSRMatrixData( csr_diag );

  hypre_CSRMatrix const * const csr_offd = hypre_ParCSRMatrixOffd( mat );
  HYPRE_Int const * const ia_offd = hypre_CSRMatrixI( csr_offd );
  HYPRE_Real const * const va_offd = hypre_CSRMatrixData( csr_offd );

  HYPRE_Real * const values = hypre_VectorData( hypre_ParVectorLocalVector( vec ) );

  auto const reducer = [=] GEOSX_HYPRE_HOST_DEVICE ( double acc, double v ){ return reduce( acc, transform( v ) ); };
  forAll< hypre::execPolicy >( hypre_CSRMatrixNumRows( csr_diag ), [=] GEOSX_HYPRE_HOST_DEVICE ( HYPRE_Int const localRow )
  {
    HYPRE_Real sum = 0.0;
    for( HYPRE_Int k = ia_diag[localRow]; k < ia_diag[localRow + 1]; ++k )
    {
      sum = reducer( sum, va_diag[k] );
    }
    if( hypre_CSRMatrixNumCols( csr_offd ) > 0 )
    {
      for( HYPRE_Int k = ia_offd[localRow]; k < ia_offd[localRow + 1]; ++k )
      {
        sum = reducer( sum, va_offd[k] );
      }
    }
    values[localRow] = sum;
  } );
}

} // namespace hypre

} // namespace geosx

#endif //GEOSX_LINEARALGEBRA_INTERFACES_HYPREKERNELS_HPP_
