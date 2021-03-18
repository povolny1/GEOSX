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
 * @file HypreUtils.cpp
 */

#include "HypreUtils.hpp"

#include "linearAlgebra/interfaces/hypre/HypreVector.hpp"

#include <_hypre_parcsr_ls.h>

namespace geosx
{

namespace hypre
{

HYPRE_Int DummySetup( HYPRE_Solver,
                      HYPRE_ParCSRMatrix,
                      HYPRE_ParVector,
                      HYPRE_ParVector )
{
  return 0;
}

HYPRE_Int SuperLUDistSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x )
{
  GEOSX_UNUSED_VAR( A );
  return hypre_SLUDistSolve( solver, b, x );
}

HYPRE_Int SuperLUDistDestroy( HYPRE_Solver solver )
{
  return hypre_SLUDistDestroy( solver );
}

/**
 * @brief Holds temporary data needed by hypre's relaxation methods.
 *
 * Many implementations dispatched to by hypre_BoomerAMGRelax crash unless Vtemp parameter
 * is provided (a temporary storage vector). We want the temporary vector to persist across
 * multiple applications of the relaxation method. In order to make it conform to hypre's
 * solver interface, we allocate this vector during setup and disguise it as HYPRE_Solver.
 * Hence the multiple reinterpret_casts (replacing C-style casts used in hypre) required
 * to convert back and forth between HYPRE_Solver and pointer to actual data struct.
 *
 * We also have to manage memory the "old" way (raw new/delete), because a single pointer
 * (HYPRE_Solver, which is an opaque pointer type) is all we got to manage the struct with.
 */
struct RelaxationData
{
  HYPRE_Int type = -1;
  HypreVector vtemp;
  HypreVector norms;
};

HYPRE_Int RelaxationCreate( HYPRE_Solver & solver,
                            HYPRE_Int const type )
{
  RelaxationData * const data = new RelaxationData;
  data->type = type;
  solver = reinterpret_cast< HYPRE_Solver >( data );
  return 0;
}

HYPRE_Int RelaxationSetup( HYPRE_Solver solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector b,
                           HYPRE_ParVector x )
{
  GEOSX_UNUSED_VAR( b, x );
  RelaxationData * const data = reinterpret_cast< RelaxationData * >( solver );
  data->vtemp.createWithLocalSize( hypre_ParCSRMatrixNumRows( A ), hypre_ParCSRMatrixComm( A ) );
  data->norms.createWithLocalSize( hypre_ParCSRMatrixNumRows( A ), hypre_ParCSRMatrixComm( A ) );

  // L1-smoothers needs row L1-norms precomputed
  if( data->type == 7 || data->type == 8 || data->type == 18 )
  {
    hypre_CSRMatrix * const csr_diag = hypre_ParCSRMatrixDiag( A );
    hypre_CSRMatrix * const csr_offd = hypre_ParCSRMatrixOffd( A );
    HYPRE_Real * const values = hypre_VectorData( hypre_ParVectorLocalVector( data->norms.unwrapped() ) );
    hypre_CSRMatrixComputeRowSum( csr_diag, nullptr, nullptr, values, 1, 1.0, "set" );
    if( hypre_CSRMatrixNumCols( csr_offd ) > 0 )
    {
      hypre_CSRMatrixComputeRowSum( csr_offd, nullptr, nullptr, values, 1, 1.0, "add" );
    }
  }
  return 0;
}

HYPRE_Int RelaxationSolve( HYPRE_Solver solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector b,
                           HYPRE_ParVector x )
{
  GEOSX_UNUSED_VAR( solver );
  RelaxationData * const data = reinterpret_cast< RelaxationData * >( solver );
  return hypre_BoomerAMGRelax( A, b, nullptr, data->type, 0, 1.0, 1.0, data->norms.extractLocalVector(), x, data->vtemp.unwrapped(), nullptr );
}

HYPRE_Int RelaxationDestroy( HYPRE_Solver solver )
{
  RelaxationData * const data = reinterpret_cast< RelaxationData * >( solver );
  delete data;
  return 0;
}

} // namespace hypre

} // namespace geosx
