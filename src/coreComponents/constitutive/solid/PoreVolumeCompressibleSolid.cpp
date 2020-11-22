/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file PoreVolumeCompressibleSolid.cpp
 */

#include "PoreVolumeCompressibleSolid.hpp"

namespace geosx
{

using namespace dataRepository;

namespace constitutive
{


PoreVolumeCompressibleSolid::PoreVolumeCompressibleSolid( std::string const & name, Group * const parent ):
  ConstitutiveBase( name, parent )
{
  registerWrapper( viewKeyStruct::compressibilityString, &m_compressibility )->
    setInputFlag( InputFlags::REQUIRED )->
    setDescription( "Solid compressibility" );

  registerWrapper( viewKeyStruct::referencePressureString, &m_referencePressure )->
    setInputFlag( InputFlags::REQUIRED )->
    setDescription( "Reference pressure for fluid compressibility" );

  registerWrapper( viewKeyStruct::poreVolumeMultiplierString, &m_poreVolumeMultiplier )->
    setDefaultValue( 1.0 );

  registerWrapper( viewKeyStruct::dPVMult_dPresString, &m_dPVMult_dPressure );
}

PoreVolumeCompressibleSolid::~PoreVolumeCompressibleSolid() = default;

std::unique_ptr< ConstitutiveBase >
PoreVolumeCompressibleSolid::deliverClone( string const & name,
                                           Group * const parent ) const
{
  std::unique_ptr< ConstitutiveBase > clone = ConstitutiveBase::deliverClone( name, parent );

  PoreVolumeCompressibleSolid * const
  newConstitutiveRelation = dynamic_cast< PoreVolumeCompressibleSolid * >(clone.get());

  newConstitutiveRelation->m_poreVolumeRelation = this->m_poreVolumeRelation;

  return clone;
}

void PoreVolumeCompressibleSolid::allocateConstitutiveData( dataRepository::Group * const parent,
                                                            localIndex const numConstitutivePointsPerParentIndex )
{
  ConstitutiveBase::allocateConstitutiveData( parent, numConstitutivePointsPerParentIndex );

  this->resize( parent->size() );

  m_poreVolumeMultiplier.resize( parent->size(), numConstitutivePointsPerParentIndex );
  m_dPVMult_dPressure.resize( parent->size(), numConstitutivePointsPerParentIndex );
  m_poreVolumeMultiplier.setValues< serialPolicy >( 1.0 );
}

void PoreVolumeCompressibleSolid::PostProcessInput()
{
  if( m_compressibility < 0.0 )
  {
    string const message = "An invalid value of fluid bulk modulus (" + std::to_string( m_compressibility ) + ") is specified";
    GEOSX_ERROR( message );
  }
  m_poreVolumeRelation.SetCoefficients( m_referencePressure, 1.0, m_compressibility );
}

void PoreVolumeCompressibleSolid::StateUpdatePointPressure( real64 const & pres,
                                                            localIndex const k,
                                                            localIndex const q )
{
  m_poreVolumeRelation.Compute( pres, m_poreVolumeMultiplier[k][q], m_dPVMult_dPressure[k][q] );
}

void PoreVolumeCompressibleSolid::StateUpdateBatchPressure( arrayView1d< real64 const > const & pres,
                                                            arrayView1d< real64 const > const & dPres )
{
  localIndex const numElems = m_poreVolumeMultiplier.size( 0 );
  localIndex const numQuad  = m_poreVolumeMultiplier.size( 1 );

  GEOSX_ASSERT_EQ( pres.size(), numElems );
  GEOSX_ASSERT_EQ( dPres.size(), numElems );

  ExponentialRelation< real64, ExponentApproximationType::Linear > const relation = m_poreVolumeRelation;
  GEOSX_UNUSED_VAR( relation );

  arrayView2d< real64 > const & pvmult = m_poreVolumeMultiplier;
  arrayView2d< real64 > const & dPVMult_dPres = m_dPVMult_dPressure;

  localIndex const numEntries = 29;

  real64 const x[numEntries] = {
    26.5e5, 70e5, 289e5, 309e5, 312e5, 330e5, 332e5, 359e5, 360e5, 370e5,
    380e5, 382e5, 383e5, 390e5, 430e5, 432e5, 433e5, 450e5, 490e5, 491e5,
    500e5, 600e5, 750e5, 900e5, 1000e5, 1100e5, 1120e5, 1150e5, 1200e5
  };

  real64 const y[numEntries] = {
    0.9605, 0.9624, 0.9717, 0.9726, 0.9727, 0.9735, 0.9736, 0.9746, 0.9747, 0.9751,
    0.9755, 0.9756, 0.9756, 0.9759, 0.9775, 0.9775, 0.9776, 0.9782, 0.9798, 0.9798,
    0.9802, 0.9839, 0.9892, 0.9941, 0.9972, 1.0000, 1.0005, 1.0014, 1.0027
  };

  forAll< parallelDevicePolicy<> >( numElems, [=] GEOSX_HOST_DEVICE ( localIndex const k )
  {
    for( localIndex q = 0; q < numQuad; ++q )
    {
      //relation.Compute( pres[k] + dPres[k], pvmult[k][q], dPVMult_dPres[k][q] );
      real64 const p = pres[k] + dPres[k];
      localIndex lowerIndex = 0;
      localIndex upperIndex = 1;
      localIndex intervalFound = 0;
      for( localIndex i = 0; i < numEntries-1; ++i )
      {
        intervalFound = ( p > x[lowerIndex] && p <= x[upperIndex] );
        if( intervalFound )
        {
          real64 const a = (y[upperIndex]-y[lowerIndex]) / (x[upperIndex]-x[lowerIndex]);
          real64 const b = y[lowerIndex];
          pvmult[k][q] = a * (p-x[lowerIndex]) + b;
          dPVMult_dPres[k][q] = a;
        }
        intervalFound = 0;
        lowerIndex++;
        upperIndex++;
      }
    }
  } );
}

REGISTER_CATALOG_ENTRY( ConstitutiveBase, PoreVolumeCompressibleSolid, std::string const &, Group * const )
}
} /* namespace geosx */
