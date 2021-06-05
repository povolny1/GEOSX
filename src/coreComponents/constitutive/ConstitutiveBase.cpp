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
 * @file ConstitutiveBase.cpp
 */



#include "ConstitutiveBase.hpp"

namespace geosx
{
using namespace dataRepository;
namespace constitutive
{

ConstitutiveBase::ConstitutiveBase( string const & name,
                                    Group * const parent ):
  Group( name, parent ),
  m_numQuadraturePoints( 1 ),
  m_constitutiveDataGroup( nullptr )
{
  setInputFlags( InputFlags::OPTIONAL_NONUNIQUE );
}

ConstitutiveBase::~ConstitutiveBase()
{}



ConstitutiveBase::CatalogInterface::CatalogType & ConstitutiveBase::getCatalog()
{
  static ConstitutiveBase::CatalogInterface::CatalogType catalog;
  return catalog;
}

void ConstitutiveBase::allocateConstitutiveData( dataRepository::Group & parent,
                                                 localIndex const numConstitutivePointsPerParentIndex )
{
  m_numQuadraturePoints = numConstitutivePointsPerParentIndex;
  m_constitutiveDataGroup = &parent;

  for( auto & group : this->getSubGroups() )
  {
    for( auto & wrapper : group.second->wrappers() )
    {
      if( wrapper.second->sizedFromParent() )
      {
        string const & wrapperName = wrapper.first;
        parent.registerWrapper( makeFieldName( this->getName(), wrapperName ), wrapper.second->clone( wrapperName, parent ) ).
          setRestartFlags( RestartFlags::NO_WRITE );
      }
    }
  }

  for( auto & wrapper : this->wrappers() )
  {
    if( wrapper.second->sizedFromParent() )
    {
      string const wrapperName = wrapper.first;
      /*
      bool const skip = ( makeFieldName( this->getName(), wrapperName ) == "rock_BiotCoefficient" ||
                          makeFieldName( this->getName(), wrapperName ) == "rock_density" ||
                          makeFieldName( this->getName(), wrapperName ) == "rock_shearModulus" ||
                          makeFieldName( this->getName(), wrapperName ) == "rock_bulkModulus" ) &&
                        parent.hasWrapper( makeFieldName( this->getName(), wrapperName ) );
      if( !skip )
      {
      */
        parent.registerWrapper( makeFieldName( this->getName(), wrapperName ), wrapper.second->clone( wrapperName, parent ) ).
          setRestartFlags( RestartFlags::NO_WRITE );
      //}
    }
  }

  this->resize( parent.size() );

  /*
  for( auto & wrapper : this->wrappers() )
  {
    if( wrapper.second->sizedFromParent() )
    {
      string const wrapperName = wrapper.first;
      bool const populateDensity = ( makeFieldName( this->getName(), wrapperName ) == "rock_density" ) &&
                                   parent.hasWrapper( makeFieldName( this->getName(), wrapperName ) );
      if( populateDensity )
      {
        Wrapper< array1d< real64 > > & wrapperFromCPG = parent.getWrapper< array1d< real64 >, string >( makeFieldName( this->getName(), wrapperName ) );
        wrapperFromCPG.setPlotLevel( dataRepository::PlotLevel::LEVEL_0 );
        arrayView1d< real64 > valFromCPG = parent.getReference< array1d< real64 > >( makeFieldName( this->getName(), wrapperName ) );
        arrayView2d< real64 > valFromModel = this->getReference< array2d< real64 > >( wrapperName );
        for( localIndex k = 0; k < valFromCPG.size(); ++k )
        {
          for( localIndex q = 0; q < m_numQuadraturePoints; ++q )
          {
            valFromModel( k, q ) = valFromCPG( k );
          }
        }
      }
      bool const populateMechParams = ( makeFieldName( this->getName(), wrapperName ) == "rock_BiotCoefficient" ||
                                        makeFieldName( this->getName(), wrapperName ) == "rock_shearModulus" ||
                                        makeFieldName( this->getName(), wrapperName ) == "rock_bulkModulus" ) &&
                                      parent.hasWrapper( makeFieldName( this->getName(), wrapperName ) );
      if( populateMechParams )
      {
        Wrapper< array1d< real64 > > & wrapperFromCPG = parent.getWrapper< array1d< real64 >, string >( makeFieldName( this->getName(), wrapperName ) );
        wrapperFromCPG.setPlotLevel( dataRepository::PlotLevel::LEVEL_0 );
        arrayView1d< real64 > valFromCPG = parent.getReference< array1d< real64 > >( makeFieldName( this->getName(), wrapperName ) );
        arrayView1d< real64 > valFromModel = this->getReference< array1d< real64 > >( wrapperName );
        for( localIndex k = 0; k < valFromCPG.size(); ++k )
        {
          valFromModel( k ) = valFromCPG( k );
        }
      }

    }
  }
  */
}

std::unique_ptr< ConstitutiveBase >
ConstitutiveBase::deliverClone( string const & name,
                                Group * const parent ) const
{
  std::unique_ptr< ConstitutiveBase >
  newModel = ConstitutiveBase::CatalogInterface::factory( this->getCatalogName(), name, parent );

  newModel->forWrappers( [&]( WrapperBase & wrapper )
  {
    wrapper.copyWrapper( this->getWrapperBase( wrapper.getName() ) );
  } );

  return newModel;
}


}
} /* namespace geosx */
