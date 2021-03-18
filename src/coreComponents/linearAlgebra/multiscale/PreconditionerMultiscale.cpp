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
 * @file PreconditionerMultiscale.cpp
 */

#include "PreconditionerMultiscale.hpp"

#include "linearAlgebra/interfaces/InterfaceTypes.hpp"
#include "linearAlgebra/multiscale/LevelBuilderBase.hpp"

namespace geosx
{

template< typename LAI >
PreconditionerMultiscale< LAI >::PreconditionerMultiscale( LinearSolverParameters params,
                                                           MeshLevel & mesh )
  : Base(),
  m_params( std::move( params.multiscale ) ),
  m_mesh( mesh )
{}

template< typename LAI >
PreconditionerMultiscale< LAI >::~PreconditionerMultiscale() = default;

template< typename LAI >
void PreconditionerMultiscale< LAI >::createLevels( Matrix const & mat,
                                                    DofManager const & dofManager )
{
  GEOSX_MARK_FUNCTION;
  m_levels.clear();

  // create fine level
  m_levels.emplace_back();
  Level & fine = m_levels[0];
  fine.builder = multiscale::LevelBuilderBase< LAI >::createInstance( m_params.fieldName + "_ms_level0", m_params );
  fine.builder->initializeFineLevel( m_mesh, dofManager, m_params.fieldName, mat.getComm() );
  fine.matrix = &mat;

  // create coarse levels
  for( integer level_index = 1; level_index < m_params.maxLevels; ++level_index )
  {
    m_levels.emplace_back();
    Level & coarse = m_levels[level_index];
    coarse.builder = multiscale::LevelBuilderBase< LAI >::createInstance( m_params.fieldName + "_ms_level" + std::to_string( level_index ), m_params );
    coarse.builder->initializeCoarseLevel( *m_levels[level_index - 1].builder );
    coarse.matrix = &coarse.builder->matrix();

    if( coarse.matrix->numGlobalRows() <= m_params.minGlobalDof )
    {
      break;
    }
  }

  // create smoothers
  for( size_t level_index = 0; level_index < m_levels.size() - 1; ++level_index )
  {
    Level & level = m_levels[level_index];
    // TODO: smoother options from input
    LinearSolverParameters smoother_params;
    smoother_params.preconditionerType = m_params.smootherType;
    if( m_params.preOrPostSmoothing == LinearSolverParameters::AMG::PreOrPost::pre
        || m_params.preOrPostSmoothing == LinearSolverParameters::AMG::PreOrPost::both )
    {
      level.presmoother = LAI::createPreconditioner( smoother_params );
    }
    if( m_params.preOrPostSmoothing == LinearSolverParameters::AMG::PreOrPost::post
        || m_params.preOrPostSmoothing == LinearSolverParameters::AMG::PreOrPost::both )
    {
      level.postsmoother = LAI::createPreconditioner( smoother_params );
    }
    // TODO: pre/post smoother could be the same object
  }

  // create vectors
  for( size_t level_index = 0; level_index < m_levels.size(); ++level_index )
  {
    Level & level = m_levels[level_index];
    level.rhs.createWithLocalSize( level.matrix->numLocalRows(), level.matrix->getComm() );
    level.sol.createWithLocalSize( level.matrix->numLocalRows(), level.matrix->getComm() );
    level.tmp.createWithLocalSize( level.matrix->numLocalRows(), level.matrix->getComm() );
  }

  // create coarse solver
  {
    LinearSolverParameters coarseParams;
    coarseParams.preconditionerType = m_params.coarseType;
    m_coarse_solver = LAI::createPreconditioner( coarseParams );
  }
}

template< typename LAI >
void PreconditionerMultiscale< LAI >::setup( Matrix const & mat )
{
  GEOSX_MARK_FUNCTION;

  Base::setup( mat );

  if( !m_initialized )
  {
    GEOSX_ERROR_IF( mat.dofManager() == nullptr, "PreconditionerMultiscale: DofManager is required" );
    createLevels( mat, *mat.dofManager() );
    m_initialized = true;
  }

  // compute level operators
  for( integer level_index = 1; level_index < m_params.maxLevels; ++level_index )
  {
    m_levels[level_index].builder->compute( *m_levels[level_index - 1].matrix );
  }

  // compute level smoothers
  for( size_t level_index = 0; level_index < m_levels.size() - 1; ++level_index )
  {
    Level & level = m_levels[level_index];
    if( level.presmoother )
    {
      m_levels[level_index].presmoother->setup( *m_levels[level_index].matrix );
    }
    if( level.postsmoother )
    {
      m_levels[level_index].postsmoother->setup( *m_levels[level_index].matrix );
    }
  }

  // setup coarse solver
  m_coarse_solver->setup( *m_levels.back().matrix );
}

template< typename LAI >
void PreconditionerMultiscale< LAI >::apply( Vector const & src,
                                             Vector & dst ) const
{
  // TODO: remove hardcoded V-cycle, abstract into a separate component

  m_levels[0].rhs.copy( src );
  int const numLevels = LvArray::integerConversion< int >( m_levels.size() );

  // down phase
  for( int level_index = 0; level_index < numLevels - 1; ++level_index )
  {
    Level const & fine = m_levels[level_index];
    Level const & coarse = m_levels[level_index + 1];
    fine.sol.zero();
    if( fine.presmoother )
    {
      for( integer s = 0; s < m_params.numSmootherSweeps; ++s )
      {
        fine.presmoother->apply( fine.rhs, fine.tmp );
        fine.sol.axpy( 1.0, fine.tmp );
        fine.matrix->residual( fine.tmp, fine.rhs, fine.rhs );
      }
    }
    coarse.builder->restriction().apply( fine.rhs, coarse.rhs );
  }

  // coarse level solve
  m_coarse_solver->apply( m_levels.back().rhs, m_levels.back().sol );

  // up phase
  for( int level_index = numLevels - 2; level_index >= 0; --level_index )
  {
    Level const & fine = m_levels[level_index];
    Level const & coarse = m_levels[level_index + 1];
    coarse.builder->prolongation().apply( coarse.sol, fine.tmp );
    fine.sol.axpy( 1.0, fine.tmp );
    fine.matrix->residual( fine.tmp, fine.rhs, fine.rhs );
    if( fine.postsmoother )
    {
      for( integer s = 0; s < m_params.numSmootherSweeps; ++s )
      {
        fine.postsmoother->apply( fine.rhs, fine.tmp );
        fine.sol.axpy( 1.0, fine.tmp );
        fine.matrix->residual( fine.tmp, fine.rhs, fine.rhs );
      }
    }
  }

  dst.copy( m_levels[0].sol );
}

// -----------------------
// Explicit Instantiations
// -----------------------
#ifdef GEOSX_USE_TRILINOS
template class PreconditionerMultiscale< TrilinosInterface >;
#endif

#ifdef GEOSX_USE_HYPRE
template class PreconditionerMultiscale< HypreInterface >;
#endif

#ifdef GEOSX_USE_PETSC
template class PreconditionerMultiscale< PetscInterface >;
#endif

} // namespace geosx
