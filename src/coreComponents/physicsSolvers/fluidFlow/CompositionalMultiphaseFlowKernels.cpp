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
 * @file CompositionalMultiphaseFlowKernels.cpp
 */

#include <physicsSolvers/multiphysics/CompositionalMultiphaseReservoir.hpp>
#include "CompositionalMultiphaseFlowKernels.hpp"

#include "finiteVolume/CellElementStencilTPFA.hpp"
#include "finiteVolume/FaceElementStencil.hpp"

namespace geosx
{

namespace CompositionalMultiphaseFlowKernels
{

/******************************** ComponentFractionKernel ********************************/

template< localIndex NC >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
ComponentFractionKernel::
  compute( arraySlice1d< real64 const > const compDens,
           arraySlice1d< real64 const > const dCompDens,
           arraySlice1d< real64 > const compFrac,
           arraySlice2d< real64 > const dCompFrac_dCompDens )
{
  real64 totalDensity = 0.0;

  for( localIndex ic = 0; ic < NC; ++ic )
  {
    totalDensity += compDens[ic] + dCompDens[ic];
  }

  real64 const totalDensityInv = 1.0 / totalDensity;

  for( localIndex ic = 0; ic < NC; ++ic )
  {
    compFrac[ic] = (compDens[ic] + dCompDens[ic]) * totalDensityInv;
    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dCompFrac_dCompDens[ic][jc] = -compFrac[ic] * totalDensityInv;
    }
    dCompFrac_dCompDens[ic][ic] += totalDensityInv;
  }
}

template< localIndex NC >
void
ComponentFractionKernel::
  launch( localIndex const size,
          arrayView2d< real64 const > const & compDens,
          arrayView2d< real64 const > const & dCompDens,
          arrayView2d< real64 > const & compFrac,
          arrayView3d< real64 > const & dCompFrac_dCompDens )
{
  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const a )
  {
    compute< NC >( compDens[a],
                   dCompDens[a],
                   compFrac[a],
                   dCompFrac_dCompDens[a] );
  } );
}

template< localIndex NC >
void
ComponentFractionKernel::
  launch( SortedArrayView< localIndex const > const & targetSet,
          arrayView2d< real64 const > const & compDens,
          arrayView2d< real64 const > const & dCompDens,
          arrayView2d< real64 > const & compFrac,
          arrayView3d< real64 > const & dCompFrac_dCompDens )
{
  forAll< parallelDevicePolicy<> >( targetSet.size(), [=] GEOSX_HOST_DEVICE ( localIndex const i )
  {
    localIndex const a = targetSet[ i ];
    compute< NC >( compDens[a],
                   dCompDens[a],
                   compFrac[a],
                   dCompFrac_dCompDens[a] );
  } );
}

#define INST_ComponentFractionKernel( NC ) \
  template \
  void ComponentFractionKernel:: \
    launch< NC >( localIndex const size, \
                  arrayView2d< real64 const > const & compDens, \
                  arrayView2d< real64 const > const & dCompDens, \
                  arrayView2d< real64 > const & compFrac, \
                  arrayView3d< real64 > const & dCompFrac_dCompDens ); \
  template \
  void ComponentFractionKernel:: \
    launch< NC >( SortedArrayView< localIndex const > const & targetSet, \
                  arrayView2d< real64 const > const & compDens, \
                  arrayView2d< real64 const > const & dCompDens, \
                  arrayView2d< real64 > const & compFrac, \
                  arrayView3d< real64 > const & dCompFrac_dCompDens )

INST_ComponentFractionKernel( 1 );
INST_ComponentFractionKernel( 2 );
INST_ComponentFractionKernel( 3 );
INST_ComponentFractionKernel( 4 );
INST_ComponentFractionKernel( 5 );

#undef INST_ComponentFractionKernel

/******************************** PhaseVolumeFractionKernel ********************************/

template< localIndex NC, localIndex NP >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
PhaseVolumeFractionKernel::
  compute( arraySlice1d< real64 const > const & compDens,
           arraySlice1d< real64 const > const & dCompDens,
           arraySlice2d< real64 const > const & dCompFrac_dCompDens,
           arraySlice1d< real64 const > const & phaseDens,
           arraySlice1d< real64 const > const & dPhaseDens_dPres,
           arraySlice2d< real64 const > const & dPhaseDens_dComp,
           arraySlice1d< real64 const > const & phaseFrac,
           arraySlice1d< real64 const > const & dPhaseFrac_dPres,
           arraySlice2d< real64 const > const & dPhaseFrac_dComp,
           arraySlice1d< real64 > const & phaseVolFrac,
           arraySlice1d< real64 > const & dPhaseVolFrac_dPres,
           arraySlice2d< real64 > const & dPhaseVolFrac_dComp )
{
  real64 work[NC];

  // compute total density from component partial densities
  real64 totalDensity = 0.0;
  real64 const dTotalDens_dCompDens = 1.0;
  for( localIndex ic = 0; ic < NC; ++ic )
  {
    totalDensity += compDens[ic] + dCompDens[ic];
  }

  for( localIndex ip = 0; ip < NP; ++ip )
  {
    // Expression for volume fractions: S_p = (nu_p / rho_p) * rho_t
    real64 const phaseDensInv = 1.0 / phaseDens[ip];

    // compute saturation and derivatives except multiplying by the total density
    phaseVolFrac[ip] = phaseFrac[ip] * phaseDensInv;

    dPhaseVolFrac_dPres[ip] =
      (dPhaseFrac_dPres[ip] - phaseVolFrac[ip] * dPhaseDens_dPres[ip]) * phaseDensInv;

    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dPhaseVolFrac_dComp[ip][jc] =
        (dPhaseFrac_dComp[ip][jc] - phaseVolFrac[ip] * dPhaseDens_dComp[ip][jc]) * phaseDensInv;
    }

    // apply chain rule to convert derivatives from global component fractions to densities
    applyChainRuleInPlace( NC, dCompFrac_dCompDens, dPhaseVolFrac_dComp[ip], work );

    // now finalize the computation by multiplying by total density
    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dPhaseVolFrac_dComp[ip][jc] *= totalDensity;
      dPhaseVolFrac_dComp[ip][jc] += phaseVolFrac[ip] * dTotalDens_dCompDens;
    }

    phaseVolFrac[ip] *= totalDensity;
    dPhaseVolFrac_dPres[ip] *= totalDensity;
  }
}

template< localIndex NC, localIndex NP >
void PhaseVolumeFractionKernel::
  launch( localIndex const size,
          arrayView2d< real64 const > const & compDens,
          arrayView2d< real64 const > const & dCompDens,
          arrayView3d< real64 const > const & dCompFrac_dCompDens,
          arrayView3d< real64 const > const & phaseDens,
          arrayView3d< real64 const > const & dPhaseDens_dPres,
          arrayView4d< real64 const > const & dPhaseDens_dComp,
          arrayView3d< real64 const > const & phaseFrac,
          arrayView3d< real64 const > const & dPhaseFrac_dPres,
          arrayView4d< real64 const > const & dPhaseFrac_dComp,
          arrayView2d< real64 > const & phaseVolFrac,
          arrayView2d< real64 > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 > const & dPhaseVolFrac_dComp )
{
  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const a )
  {
    compute< NC, NP >( compDens[a],
                       dCompDens[a],
                       dCompFrac_dCompDens[a],
                       phaseDens[a][0],
                       dPhaseDens_dPres[a][0],
                       dPhaseDens_dComp[a][0],
                       phaseFrac[a][0],
                       dPhaseFrac_dPres[a][0],
                       dPhaseFrac_dComp[a][0],
                       phaseVolFrac[a],
                       dPhaseVolFrac_dPres[a],
                       dPhaseVolFrac_dComp[a] );
  } );
}

template< localIndex NC, localIndex NP >
void PhaseVolumeFractionKernel::
  launch( SortedArrayView< localIndex const > const & targetSet,
          arrayView2d< real64 const > const & compDens,
          arrayView2d< real64 const > const & dCompDens,
          arrayView3d< real64 const > const & dCompFrac_dCompDens,
          arrayView3d< real64 const > const & phaseDens,
          arrayView3d< real64 const > const & dPhaseDens_dPres,
          arrayView4d< real64 const > const & dPhaseDens_dComp,
          arrayView3d< real64 const > const & phaseFrac,
          arrayView3d< real64 const > const & dPhaseFrac_dPres,
          arrayView4d< real64 const > const & dPhaseFrac_dComp,
          arrayView2d< real64 > const & phaseVolFrac,
          arrayView2d< real64 > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 > const & dPhaseVolFrac_dComp )
{
  forAll< parallelDevicePolicy<> >( targetSet.size(), [=] GEOSX_HOST_DEVICE ( localIndex const i )
  {
    localIndex const a = targetSet[ i ];
    compute< NC, NP >( compDens[a],
                       dCompDens[a],
                       dCompFrac_dCompDens[a],
                       phaseDens[a][0],
                       dPhaseDens_dPres[a][0],
                       dPhaseDens_dComp[a][0],
                       phaseFrac[a][0],
                       dPhaseFrac_dPres[a][0],
                       dPhaseFrac_dComp[a][0],
                       phaseVolFrac[a],
                       dPhaseVolFrac_dPres[a],
                       dPhaseVolFrac_dComp[a] );
  } );
}

#define INST_PhaseVolumeFractionKernel( NC, NP ) \
  template \
  void \
  PhaseVolumeFractionKernel:: \
    launch< NC, NP >( localIndex const size, \
                      arrayView2d< real64 const > const & compDens, \
                      arrayView2d< real64 const > const & dCompDens, \
                      arrayView3d< real64 const > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const > const & phaseDens, \
                      arrayView3d< real64 const > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const > const & phaseFrac, \
                      arrayView3d< real64 const > const & dPhaseFrac_dPres, \
                      arrayView4d< real64 const > const & dPhaseFrac_dComp, \
                      arrayView2d< real64 > const & phaseVolFrac, \
                      arrayView2d< real64 > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 > const & dPhaseVolFrac_dComp ); \
  template \
  void \
  PhaseVolumeFractionKernel:: \
    launch< NC, NP >( SortedArrayView< localIndex const > const & targetSet, \
                      arrayView2d< real64 const > const & compDens, \
                      arrayView2d< real64 const > const & dCompDens, \
                      arrayView3d< real64 const > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const > const & phaseDens, \
                      arrayView3d< real64 const > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const > const & phaseFrac, \
                      arrayView3d< real64 const > const & dPhaseFrac_dPres, \
                      arrayView4d< real64 const > const & dPhaseFrac_dComp, \
                      arrayView2d< real64 > const & phaseVolFrac, \
                      arrayView2d< real64 > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 > const & dPhaseVolFrac_dComp )

INST_PhaseVolumeFractionKernel( 1, 1 );
INST_PhaseVolumeFractionKernel( 2, 1 );
INST_PhaseVolumeFractionKernel( 3, 1 );
INST_PhaseVolumeFractionKernel( 4, 1 );
INST_PhaseVolumeFractionKernel( 5, 1 );

INST_PhaseVolumeFractionKernel( 1, 2 );
INST_PhaseVolumeFractionKernel( 2, 2 );
INST_PhaseVolumeFractionKernel( 3, 2 );
INST_PhaseVolumeFractionKernel( 4, 2 );
INST_PhaseVolumeFractionKernel( 5, 2 );

INST_PhaseVolumeFractionKernel( 1, 3 );
INST_PhaseVolumeFractionKernel( 2, 3 );
INST_PhaseVolumeFractionKernel( 3, 3 );
INST_PhaseVolumeFractionKernel( 4, 3 );
INST_PhaseVolumeFractionKernel( 5, 3 );

#undef INST_PhaseVolumeFractionKernel

/******************************** PhaseMobilityKernel ********************************/

template< localIndex NC, localIndex NP >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
PhaseMobilityKernel::
compute( arraySlice2d< real64 const > const & dCompFrac_dCompDens,
         arraySlice1d< real64 const > const & GEOSX_UNUSED_PARAM(phaseDens),
         arraySlice1d< real64 const > const & GEOSX_UNUSED_PARAM(dPhaseDens_dPres),
         arraySlice2d< real64 const > const & GEOSX_UNUSED_PARAM(dPhaseDens_dComp),
         arraySlice1d< real64 const > const & phaseVisc,
         arraySlice1d< real64 const > const & dPhaseVisc_dPres,
         arraySlice2d< real64 const > const & dPhaseVisc_dComp,
         arraySlice1d< real64 const > const & phaseRelPerm,
         arraySlice2d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac,
         arraySlice1d< real64 const > const & dPhaseVolFrac_dPres,
         arraySlice2d< real64 const > const & dPhaseVolFrac_dComp,
         arraySlice1d< real64 > const & phaseMob,
         arraySlice1d< real64 > const & dPhaseMob_dPres,
         arraySlice2d< real64 > const & dPhaseMob_dComp )
{
  real64 dRelPerm_dC[NC];
  real64 dVisc_dC[NC];

  for( localIndex ip = 0; ip < NP; ++ip )
  {
    real64 const viscosity = phaseVisc[ip];
    real64 const dVisc_dP = dPhaseVisc_dPres[ip];
    applyChainRule( NC, dCompFrac_dCompDens, dPhaseVisc_dComp[ip], dVisc_dC );

    real64 const relPerm = phaseRelPerm[ip];
    real64 dRelPerm_dP = 0.0;
    for( localIndex ic = 0; ic < NC; ++ic )
    {
      dRelPerm_dC[ic] = 0.0;
    }

    for( localIndex jp = 0; jp < NP; ++jp )
    {
      real64 const dRelPerm_dS = dPhaseRelPerm_dPhaseVolFrac[ip][jp];
      dRelPerm_dP += dRelPerm_dS * dPhaseVolFrac_dPres[jp];

      for( localIndex jc = 0; jc < NC; ++jc )
      {
        dRelPerm_dC[jc] += dRelPerm_dS * dPhaseVolFrac_dComp[jp][jc];
      }
    }

    real64 const mobility = relPerm  / viscosity;

    phaseMob[ip] = mobility;
    dPhaseMob_dPres[ip] = dRelPerm_dP / viscosity
                          - mobility * dVisc_dP / viscosity;

    // compositional derivatives
    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dPhaseMob_dComp[ip][jc] = dRelPerm_dC[jc] / viscosity
                                - mobility * dVisc_dC[jc] / viscosity;
    }
  }
}

// weighted
/*
template< localIndex NC, localIndex NP >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
PhaseMobilityKernel::
  compute( arraySlice2d< real64 const > const & dCompFrac_dCompDens,
           arraySlice1d< real64 const > const & phaseDens,
           arraySlice1d< real64 const > const & dPhaseDens_dPres,
           arraySlice2d< real64 const > const & dPhaseDens_dComp,
           arraySlice1d< real64 const > const & phaseVisc,
           arraySlice1d< real64 const > const & dPhaseVisc_dPres,
           arraySlice2d< real64 const > const & dPhaseVisc_dComp,
           arraySlice1d< real64 const > const & phaseRelPerm,
           arraySlice2d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac,
           arraySlice1d< real64 const > const & dPhaseVolFrac_dPres,
           arraySlice2d< real64 const > const & dPhaseVolFrac_dComp,
           arraySlice1d< real64 > const & phaseMob,
           arraySlice1d< real64 > const & dPhaseMob_dPres,
           arraySlice2d< real64 > const & dPhaseMob_dComp )
{
  real64 dRelPerm_dC[NC];
  real64 dDens_dC[NC];
  real64 dVisc_dC[NC];

  for( localIndex ip = 0; ip < NP; ++ip )
  {
    real64 const density = phaseDens[ip];
    real64 const dDens_dP = dPhaseDens_dPres[ip];
    applyChainRule( NC, dCompFrac_dCompDens, dPhaseDens_dComp[ip], dDens_dC );

      std::cerr << "Checking dDens_dC " << dDens_dC[0] << std::endl;

    real64 const viscosity = phaseVisc[ip];
    real64 const dVisc_dP = dPhaseVisc_dPres[ip];
    applyChainRule( NC, dCompFrac_dCompDens, dPhaseVisc_dComp[ip], dVisc_dC );

    real64 const relPerm = phaseRelPerm[ip];
    real64 dRelPerm_dP = 0.0;
    for( localIndex ic = 0; ic < NC; ++ic )
    {
      dRelPerm_dC[ic] = 0.0;
    }

    for( localIndex jp = 0; jp < NP; ++jp )
    {
      real64 const dRelPerm_dS = dPhaseRelPerm_dPhaseVolFrac[ip][jp];
      dRelPerm_dP += dRelPerm_dS * dPhaseVolFrac_dPres[jp];

      for( localIndex jc = 0; jc < NC; ++jc )
      {
        dRelPerm_dC[jc] += dRelPerm_dS * dPhaseVolFrac_dComp[jp][jc];
      }
    }

    real64 const mobility = relPerm * density / viscosity;

    phaseMob[ip] = mobility;
    dPhaseMob_dPres[ip] = dRelPerm_dP * density / viscosity
                          + mobility * (dDens_dP / density - dVisc_dP / viscosity);

    // compositional derivatives
    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dPhaseMob_dComp[ip][jc] = dRelPerm_dC[jc] * density / viscosity
                                + mobility * (dDens_dC[jc] / density - dVisc_dC[jc] / viscosity);
    }
  }
}
*/

template< localIndex NC, localIndex NP >
void PhaseMobilityKernel::
  launch( localIndex const size,
          arrayView3d< real64 const > const & dCompFrac_dCompDens,
          arrayView3d< real64 const > const & phaseDens,
          arrayView3d< real64 const > const & dPhaseDens_dPres,
          arrayView4d< real64 const > const & dPhaseDens_dComp,
          arrayView3d< real64 const > const & phaseVisc,
          arrayView3d< real64 const > const & dPhaseVisc_dPres,
          arrayView4d< real64 const > const & dPhaseVisc_dComp,
          arrayView3d< real64 const > const & phaseRelPerm,
          arrayView4d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac,
          arrayView2d< real64 const > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const > const & dPhaseVolFrac_dComp,
          arrayView2d< real64 > const & phaseMob,
          arrayView2d< real64 > const & dPhaseMob_dPres,
          arrayView3d< real64 > const & dPhaseMob_dComp )
{
  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const a )
  {
    compute< NC, NP >( dCompFrac_dCompDens[a],
                       phaseDens[a][0],
                       dPhaseDens_dPres[a][0],
                       dPhaseDens_dComp[a][0],
                       phaseVisc[a][0],
                       dPhaseVisc_dPres[a][0],
                       dPhaseVisc_dComp[a][0],
                       phaseRelPerm[a][0],
                       dPhaseRelPerm_dPhaseVolFrac[a][0],
                       dPhaseVolFrac_dPres[a],
                       dPhaseVolFrac_dComp[a],
                       phaseMob[a],
                       dPhaseMob_dPres[a],
                       dPhaseMob_dComp[a] );
  } );
}

template< localIndex NC, localIndex NP >
void PhaseMobilityKernel::
  launch( SortedArrayView< localIndex const > const & targetSet,
          arrayView3d< real64 const > const & dCompFrac_dCompDens,
          arrayView3d< real64 const > const & phaseDens,
          arrayView3d< real64 const > const & dPhaseDens_dPres,
          arrayView4d< real64 const > const & dPhaseDens_dComp,
          arrayView3d< real64 const > const & phaseVisc,
          arrayView3d< real64 const > const & dPhaseVisc_dPres,
          arrayView4d< real64 const > const & dPhaseVisc_dComp,
          arrayView3d< real64 const > const & phaseRelPerm,
          arrayView4d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac,
          arrayView2d< real64 const > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const > const & dPhaseVolFrac_dComp,
          arrayView2d< real64 > const & phaseMob,
          arrayView2d< real64 > const & dPhaseMob_dPres,
          arrayView3d< real64 > const & dPhaseMob_dComp )
{
  forAll< parallelDevicePolicy<> >( targetSet.size(), [=] GEOSX_HOST_DEVICE ( localIndex const i )
  {
    localIndex const a = targetSet[ i ];
    compute< NC, NP >( dCompFrac_dCompDens[a],
                       phaseDens[a][0],
                       dPhaseDens_dPres[a][0],
                       dPhaseDens_dComp[a][0],
                       phaseVisc[a][0],
                       dPhaseVisc_dPres[a][0],
                       dPhaseVisc_dComp[a][0],
                       phaseRelPerm[a][0],
                       dPhaseRelPerm_dPhaseVolFrac[a][0],
                       dPhaseVolFrac_dPres[a],
                       dPhaseVolFrac_dComp[a],
                       phaseMob[a],
                       dPhaseMob_dPres[a],
                       dPhaseMob_dComp[a] );
  } );
}

#define INST_PhaseMobilityKernel( NC, NP ) \
  template \
  void \
  PhaseMobilityKernel:: \
    launch< NC, NP >( localIndex const size, \
                      arrayView3d< real64 const > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const > const & phaseDens, \
                      arrayView3d< real64 const > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const > const & phaseVisc, \
                      arrayView3d< real64 const > const & dPhaseVisc_dPres, \
                      arrayView4d< real64 const > const & dPhaseVisc_dComp, \
                      arrayView3d< real64 const > const & phaseRelPerm, \
                      arrayView4d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac, \
                      arrayView2d< real64 const > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 const > const & dPhaseVolFrac_dComp, \
                      arrayView2d< real64 > const & phaseMob, \
                      arrayView2d< real64 > const & dPhaseMob_dPres, \
                      arrayView3d< real64 > const & dPhaseMob_dComp ); \
  template \
  void \
  PhaseMobilityKernel:: \
    launch< NC, NP >( SortedArrayView< localIndex const > const & targetSet, \
                      arrayView3d< real64 const > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const > const & phaseDens, \
                      arrayView3d< real64 const > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const > const & phaseVisc, \
                      arrayView3d< real64 const > const & dPhaseVisc_dPres, \
                      arrayView4d< real64 const > const & dPhaseVisc_dComp, \
                      arrayView3d< real64 const > const & phaseRelPerm, \
                      arrayView4d< real64 const > const & dPhaseRelPerm_dPhaseVolFrac, \
                      arrayView2d< real64 const > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 const > const & dPhaseVolFrac_dComp, \
                      arrayView2d< real64 > const & phaseMob, \
                      arrayView2d< real64 > const & dPhaseMob_dPres, \
                      arrayView3d< real64 > const & dPhaseMob_dComp )

INST_PhaseMobilityKernel( 1, 1 );
INST_PhaseMobilityKernel( 2, 1 );
INST_PhaseMobilityKernel( 3, 1 );
INST_PhaseMobilityKernel( 4, 1 );
INST_PhaseMobilityKernel( 5, 1 );

INST_PhaseMobilityKernel( 1, 2 );
INST_PhaseMobilityKernel( 2, 2 );
INST_PhaseMobilityKernel( 3, 2 );
INST_PhaseMobilityKernel( 4, 2 );
INST_PhaseMobilityKernel( 5, 2 );

INST_PhaseMobilityKernel( 1, 3 );
INST_PhaseMobilityKernel( 2, 3 );
INST_PhaseMobilityKernel( 3, 3 );
INST_PhaseMobilityKernel( 4, 3 );
INST_PhaseMobilityKernel( 5, 3 );

#undef INST_PhaseMobilityKernel

/******************************** AccumulationKernel ********************************/

template< localIndex NC >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
AccumulationKernel::
  compute( localIndex const numPhases,
           real64 const & volume,
           real64 const & porosityOld,
           real64 const & porosityRef,
           real64 const & pvMult,
           real64 const & dPvMult_dPres,
           arraySlice2d< real64 const > const & dCompFrac_dCompDens,
           arraySlice1d< real64 const > const & phaseVolFracOld,
           arraySlice1d< real64 const > const & phaseVolFrac,
           arraySlice1d< real64 const > const & dPhaseVolFrac_dPres,
           arraySlice2d< real64 const > const & dPhaseVolFrac_dCompDens,
           arraySlice1d< real64 const > const & phaseDensOld,
           arraySlice1d< real64 const > const & phaseDens,
           arraySlice1d< real64 const > const & dPhaseDens_dPres,
           arraySlice2d< real64 const > const & dPhaseDens_dComp,
           arraySlice2d< real64 const > const & phaseCompFracOld,
           arraySlice2d< real64 const > const & phaseCompFrac,
           arraySlice2d< real64 const > const & dPhaseCompFrac_dPres,
           arraySlice3d< real64 const > const & dPhaseCompFrac_dComp,
           real64 ( & localAccum )[NC],
           real64 ( & localAccumJacobian )[NC][NC + 1] )
{
  localIndex constexpr NDOF = NC + 1;
  localIndex const NP = numPhases;

  // temporary work arrays
  real64 dPhaseAmount_dC[NC];
  real64 dPhaseCompFrac_dC[NC];

  // reset the local values
  for( localIndex i = 0; i < NC; ++i )
  {
    localAccum[i] = 0.0;
    for( localIndex j = 0; j < NDOF; ++j )
    {
      localAccumJacobian[i][j] = 0.0;
    }
  }

  // compute fluid-independent (pore volume) part
  real64 const volNew = volume;
  real64 const volOld = volume;
  real64 const dVol_dP = 0.0; // used in poroelastic solver

  real64 const poroNew = porosityRef * pvMult;
  real64 const poroOld = porosityOld;
  real64 const dPoro_dP = porosityRef * dPvMult_dPres;

  real64 const poreVolNew = volNew * poroNew;
  real64 const poreVolOld = volOld * poroOld;
  real64 const dPoreVol_dP = dVol_dP * poroNew + volNew * dPoro_dP;

  // sum contributions to component accumulation from each phase
  for( localIndex ip = 0; ip < NP; ++ip )
  {
    real64 const phaseAmountNew = poreVolNew * phaseVolFrac[ip] * phaseDens[ip];
    real64 const phaseAmountOld = poreVolOld * phaseVolFracOld[ip] * phaseDensOld[ip];

    real64 const dPhaseAmount_dP = dPoreVol_dP * phaseVolFrac[ip] * phaseDens[ip]
                                   + poreVolNew * (dPhaseVolFrac_dPres[ip] * phaseDens[ip]
                                                   + phaseVolFrac[ip] * dPhaseDens_dPres[ip]);

    // assemble density dependence
    applyChainRule( NC, dCompFrac_dCompDens, dPhaseDens_dComp[ip], dPhaseAmount_dC );
    for( localIndex jc = 0; jc < NC; ++jc )
    {
      dPhaseAmount_dC[jc] = dPhaseAmount_dC[jc] * phaseVolFrac[ip]
                            + phaseDens[ip] * dPhaseVolFrac_dCompDens[ip][jc];
      dPhaseAmount_dC[jc] *= poreVolNew;
    }

    // ic - index of component whose conservation equation is assembled
    // (i.e. row number in local matrix)
    for( localIndex ic = 0; ic < NC; ++ic )
    {
      real64 const phaseCompAmountNew = phaseAmountNew * phaseCompFrac[ip][ic];
      real64 const phaseCompAmountOld = phaseAmountOld * phaseCompFracOld[ip][ic];

      real64 const dPhaseCompAmount_dP = dPhaseAmount_dP * phaseCompFrac[ip][ic]
                                         + phaseAmountNew * dPhaseCompFrac_dPres[ip][ic];

      localAccum[ic] += phaseCompAmountNew - phaseCompAmountOld;
      localAccumJacobian[ic][0] += dPhaseCompAmount_dP;

      // jc - index of component w.r.t. whose compositional var the derivative is being taken
      // (i.e. col number in local matrix)

      // assemble phase composition dependence
      applyChainRule( NC, dCompFrac_dCompDens, dPhaseCompFrac_dComp[ip][ic], dPhaseCompFrac_dC );
      for( localIndex jc = 0; jc < NC; ++jc )
      {
        real64 const dPhaseCompAmount_dC = dPhaseCompFrac_dC[jc] * phaseAmountNew
                                           + phaseCompFrac[ip][ic] * dPhaseAmount_dC[jc];
        localAccumJacobian[ic][jc + 1] += dPhaseCompAmount_dC;
      }
    }
  }
}

template< localIndex NC >
void
AccumulationKernel::
  launch( localIndex const numPhases,
          localIndex const size,
          globalIndex const rankOffset,
          arrayView1d< globalIndex const > const & dofNumber,
          arrayView1d< integer const > const & elemGhostRank,
          arrayView1d< real64 const > const & volume,
          arrayView1d< real64 const > const & porosityOld,
          arrayView1d< real64 const > const & porosityRef,
          arrayView2d< real64 const > const & pvMult,
          arrayView2d< real64 const > const & dPvMult_dPres,
          arrayView3d< real64 const > const & dCompFrac_dCompDens,
          arrayView2d< real64 const > const & phaseVolFracOld,
          arrayView2d< real64 const > const & phaseVolFrac,
          arrayView2d< real64 const > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const > const & dPhaseVolFrac_dCompDens,
          arrayView2d< real64 const > const & phaseDensOld,
          arrayView3d< real64 const > const & phaseDens,
          arrayView3d< real64 const > const & dPhaseDens_dPres,
          arrayView4d< real64 const > const & dPhaseDens_dComp,
          arrayView3d< real64 const > const & phaseCompFracOld,
          arrayView4d< real64 const > const & phaseCompFrac,
          arrayView4d< real64 const > const & dPhaseCompFrac_dPres,
          arrayView5d< real64 const > const & dPhaseCompFrac_dComp,
          CRSMatrixView< real64, globalIndex const > const & localMatrix,
          arrayView1d< real64 > const & localRhs )
{
  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const ei )
  {
    if( elemGhostRank[ei] >= 0 )
      return;

    localIndex constexpr NDOF = NC + 1;

    real64 localAccum[NC];
    real64 localAccumJacobian[NC][NDOF];

    compute< NC >( numPhases,
                   volume[ei],
                   porosityOld[ei],
                   porosityRef[ei],
                   pvMult[ei][0],
                   dPvMult_dPres[ei][0],
                   dCompFrac_dCompDens[ei],
                   phaseVolFracOld[ei],
                   phaseVolFrac[ei],
                   dPhaseVolFrac_dPres[ei],
                   dPhaseVolFrac_dCompDens[ei],
                   phaseDensOld[ei],
                   phaseDens[ei][0],
                   dPhaseDens_dPres[ei][0],
                   dPhaseDens_dComp[ei][0],
                   phaseCompFracOld[ei],
                   phaseCompFrac[ei][0],
                   dPhaseCompFrac_dPres[ei][0],
                   dPhaseCompFrac_dComp[ei][0],
                   localAccum,
                   localAccumJacobian );

    // set DOF indices for this block
    localIndex const localRow = dofNumber[ei] - rankOffset;
    globalIndex dofIndices[NDOF];
    for( localIndex idof = 0; idof < NDOF; ++idof )
    {
      dofIndices[idof] = dofNumber[ei] + idof;
    }

    // TODO: apply equation/variable change transformation(s)

    // add contribution to residual and jacobian
    for( localIndex i = 0; i < NC; ++i )
    {
      localRhs[localRow + i] += localAccum[i];
      localMatrix.addToRow< serialAtomic >( localRow + i,
                                            dofIndices,
                                            localAccumJacobian[i],
                                            NDOF );
    }
  } );
}

#define INST_AccumulationKernel( NC ) \
  template \
  void \
  AccumulationKernel:: \
    launch< NC >( localIndex const numPhases, \
                  localIndex const size, \
                  globalIndex const rankOffset, \
                  arrayView1d< globalIndex const > const & dofNumber, \
                  arrayView1d< integer const > const & elemGhostRank, \
                  arrayView1d< real64 const > const & volume, \
                  arrayView1d< real64 const > const & porosityOld, \
                  arrayView1d< real64 const > const & porosityRef, \
                  arrayView2d< real64 const > const & pvMult, \
                  arrayView2d< real64 const > const & dPvMult_dPres, \
                  arrayView3d< real64 const > const & dCompFrac_dCompDens, \
                  arrayView2d< real64 const > const & phaseVolFracOld, \
                  arrayView2d< real64 const > const & phaseVolFrac, \
                  arrayView2d< real64 const > const & dPhaseVolFrac_dPres, \
                  arrayView3d< real64 const > const & dPhaseVolFrac_dCompDens, \
                  arrayView2d< real64 const > const & phaseDensOld, \
                  arrayView3d< real64 const > const & phaseDens, \
                  arrayView3d< real64 const > const & dPhaseDens_dPres, \
                  arrayView4d< real64 const > const & dPhaseDens_dComp, \
                  arrayView3d< real64 const > const & phaseCompFracOld, \
                  arrayView4d< real64 const > const & phaseCompFrac, \
                  arrayView4d< real64 const > const & dPhaseCompFrac_dPres, \
                  arrayView5d< real64 const > const & dPhaseCompFrac_dComp, \
                  CRSMatrixView< real64, globalIndex const > const & localMatrix, \
                  arrayView1d< real64 > const & localRhs )

INST_AccumulationKernel( 1 );
INST_AccumulationKernel( 2 );
INST_AccumulationKernel( 3 );
INST_AccumulationKernel( 4 );
INST_AccumulationKernel( 5 );

#undef INST_AccumulationKernel

/******************************** VolumeBalanceKernel ********************************/

template< localIndex NC, localIndex NUM_ELEMS, localIndex MAX_STENCIL, bool IS_UT_FORM >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
FluxKernel::
compute( localIndex const numPhases,
           localIndex const stencilSize,
           arraySlice1d< localIndex const > const seri,
           arraySlice1d< localIndex const > const sesri,
           arraySlice1d< localIndex const > const sei,
           arraySlice1d< real64 const > const stencilWeights,
           ElementViewConst< arrayView1d< real64 const > > const & pres,
           ElementViewConst< arrayView1d< real64 const > > const & dPres,
           ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
           ElementViewConst< arrayView2d< real64 const > > const & phaseMob,
           ElementViewConst< arrayView2d< real64 const > > const & dPhaseMob_dPres,
           ElementViewConst< arrayView3d< real64 const > > const & dPhaseMob_dComp,
           ElementViewConst< arrayView2d< real64 const > > const & dPhaseVolFrac_dPres,
           ElementViewConst< arrayView3d< real64 const > > const & dPhaseVolFrac_dComp,
           ElementViewConst< arrayView3d< real64 const > > const & dCompFrac_dCompDens,
           ElementViewConst< arrayView3d< real64 const > > const & phaseDens,
           ElementViewConst< arrayView3d< real64 const > > const & dPhaseDens_dPres,
           ElementViewConst< arrayView4d< real64 const > > const & dPhaseDens_dComp,
           ElementViewConst< arrayView3d< real64 const > > const & phaseMassDens,
           ElementViewConst< arrayView3d< real64 const > > const & dPhaseMassDens_dPres,
           ElementViewConst< arrayView4d< real64 const > > const & dPhaseMassDens_dComp,
           ElementViewConst< arrayView4d< real64 const > > const & phaseCompFrac,
           ElementViewConst< arrayView4d< real64 const > > const & dPhaseCompFrac_dPres,
           ElementViewConst< arrayView5d< real64 const > > const & dPhaseCompFrac_dComp,
           ElementViewConst< arrayView3d< real64 const > > const & phaseCapPressure,
           ElementViewConst< arrayView4d< real64 const > > const & dPhaseCapPressure_dPhaseVolFrac,
           integer const capPressureFlag,
           real64 const dt,
           arraySlice1d< real64 > const localFlux,
           arraySlice2d< real64 > const localFluxJacobian )
{
  localIndex constexpr NDOF = NC + 1;
  localIndex const NP = numPhases;

  real64 compFlux[NC]{};
  real64 dCompFlux_dP[MAX_STENCIL][NC]{};
  real64 dCompFlux_dC[MAX_STENCIL][NC][NC]{};

  real64 totFlux_unw{};
  real64 dTotFlux_dP[MAX_STENCIL]{};
  real64 dTotFlux_dC[MAX_STENCIL][NC]{};

  // loop over phases, compute and upwind phase flux and sum contributions to each component's flux
  for( localIndex ip = 0; ip < NP; ++ip )
  {

    real64 phaseFlux{};
    real64 dPhaseFlux_dP[MAX_STENCIL]{};
    real64 dPhaseFlux_dC[MAX_STENCIL][NC]{};

    localIndex k_up = -1;

    CompositionalMultiphaseFlowUpwindHelperKernels::formPPUVelocity<NC,NUM_ELEMS,MAX_STENCIL>(
           NP,
           ip,
           stencilSize,
           seri,
           sesri,
           sei,
           stencilWeights,
           pres,
           dPres,
           gravCoef,
           phaseMob,
           dPhaseMob_dPres,
           dPhaseMob_dComp,
           dPhaseVolFrac_dPres,
           dPhaseVolFrac_dComp,
           dCompFrac_dCompDens,
           phaseMassDens,
           dPhaseMassDens_dPres,
           dPhaseMassDens_dComp,
           phaseCapPressure,
           dPhaseCapPressure_dPhaseVolFrac,
           capPressureFlag,
           k_up,
           phaseFlux,
           dPhaseFlux_dP,
           dPhaseFlux_dC
      );

    // updateing phase Flux
    totFlux_unw += phaseFlux;

    for( localIndex ke = 0; ke < stencilSize; ++ke )
    {
      dTotFlux_dP[ke] += dPhaseFlux_dP[ke];

      for( localIndex jc = 0; jc < NC; ++jc )
      {
        dTotFlux_dC[ke][jc] += dPhaseFlux_dC[ke][jc];
      }
    }

//    mdensmultiply
    CompositionalMultiphaseFlowUpwindHelperKernels::mdensMultiply(
      ip,
      k_up,
      stencilSize,
      seri,
      sesri,
      sei,
      dCompFrac_dCompDens,
      phaseDens,
      dPhaseDens_dPres,
      dPhaseDens_dComp,
      phaseFlux,
      dPhaseFlux_dP,
      dPhaseFlux_dC);

    if( !IS_UT_FORM ) // skip  if you intend to use fixed total velocity formulation
    {
      CompositionalMultiphaseFlowUpwindHelperKernels::formPhaseComp( ip,
                                                                     k_up,
                                                                     stencilSize,
                                                                     seri,
                                                                     sesri,
                                                                     sei,
                                                                     phaseCompFrac,
                                                                     dPhaseCompFrac_dPres,
                                                                     dPhaseCompFrac_dComp,
                                                                     dCompFrac_dCompDens,
                                                                     phaseFlux,
                                                                     dPhaseFlux_dP,
                                                                     dPhaseFlux_dC,
                                                                     compFlux,
                                                                     dCompFlux_dP,
                                                                     dCompFlux_dC );

    }
  }
  // *** end of upwinding

  //if total flux formulation
  if( IS_UT_FORM )
  {
    for( localIndex ip = 0; ip < NP; ++ip )
    {
      // choose upstream cell
      // create local work arrays
      real64 phaseFlux{};
      real64 dPhaseFlux_dP[MAX_STENCIL]{};
      real64 dPhaseFlux_dC[MAX_STENCIL][NC]{};

      real64 phaseFluxV{};
      real64 dPhaseFluxV_dP[MAX_STENCIL]{};
      real64 dPhaseFluxV_dC[MAX_STENCIL][NC]{};

      real64 fflow{};
      real64 dFflow_dP[MAX_STENCIL]{};
      real64 dFflow_dC[MAX_STENCIL][NC]{};

      real64 gravHead{};
      real64 dGravHead_dP[NUM_ELEMS]{};
      real64 dGravHead_dC[NUM_ELEMS][NC]{};

      real64 dProp_dC[NC]{};

      //Fetch gravHead for phase i defined as \rho_i g dz/dx
      CompositionalMultiphaseFlowUpwindHelperKernels::formGravHead( ip,
                                                                    stencilSize,
                                                                    seri,
                                                                    sesri,
                                                                    sei,
                                                                    stencilWeights,
                                                                    gravCoef,
                                                                    dCompFrac_dCompDens,
                                                                    phaseMassDens,
                                                                    dPhaseMassDens_dPres,
                                                                    dPhaseMassDens_dComp,
                                                                    gravHead,
                                                                    dGravHead_dP,
                                                                    dGravHead_dC,
                                                                    dProp_dC );

      // and the fractional flow for viscous part as \lambda_i^{up}/\sum_{NP}(\lambda_j^{up}) with up decided upon
      // the Upwind strategy
      localIndex k_up = -1;
      CompositionalMultiphaseFlowUpwindHelperKernels::formFracFlow< NC, NUM_ELEMS, MAX_STENCIL,
        CompositionalMultiphaseFlowUpwindHelperKernels::term::Viscous,
        CompositionalMultiphaseFlowUpwindHelperKernels::PU >( NP,
                                                              ip,
                                                              stencilSize,
                                                              seri,
                                                              sesri,
                                                              sei,
                                                              stencilWeights,
                                                              totFlux_unw, //in fine should be a ElemnetViewConst once seq form are in place
                                                              pres,
                                                              dPres,
                                                              gravCoef,
                                                              dCompFrac_dCompDens,
                                                              phaseMassDens,
                                                              dPhaseMassDens_dPres,
                                                              dPhaseMassDens_dComp,
                                                              phaseMob,
                                                              dPhaseMob_dPres,
                                                              dPhaseMob_dComp,
                                                              k_up,
                                                              fflow,
                                                              dFflow_dP,
                                                              dFflow_dC );

//mdensmultiply
      CompositionalMultiphaseFlowUpwindHelperKernels::mdensMultiply(
                                                                  ip,
                                                                  k_up,
                                                                  stencilSize,
                                                                  seri,
                                                                  sesri,
                                                                  sei,
                                                                  dCompFrac_dCompDens,
                                                                  phaseDens,
                                                                  dPhaseDens_dPres,
                                                                  dPhaseDens_dComp,
                                                                  fflow,
                                                                  dFflow_dP,
                                                                  dFflow_dC
      );

      // Assembling the viscous flux (and derivatives) from fractional flow and total velocity as \phi_{\mu} = f_i^{up,\mu} uT
      phaseFluxV = fflow * totFlux_unw;
      for( localIndex ke = 0; ke < stencilSize; ++ke )
      {
        dPhaseFluxV_dP[ke] += dFflow_dP[ke] * totFlux_unw;

        for( localIndex jc = 0; jc < NC; ++jc )
        {
          dPhaseFluxV_dC[ke][jc] += dFflow_dC[ke][jc] * totFlux_unw;
        }
      }

      //NON-FIXED UT -- to be canceled out if considered fixed
      for( localIndex ke = 0; ke < stencilSize; ++ke )
      {
        dPhaseFluxV_dP[ke] += fflow * dTotFlux_dP[ke];

        for( localIndex jc = 0; jc < NC; ++jc )
        {
          dPhaseFluxV_dC[ke][jc] += fflow * dTotFlux_dC[ke][jc];
        }
      }

      phaseFlux += phaseFluxV;
      for( localIndex ke = 0; ke < stencilSize; ++ke )
      {
        dPhaseFlux_dP[ke] += dPhaseFluxV_dP[ke];
        for( localIndex ic = 0; ic < NC; ++ic )
          dPhaseFlux_dC[ke][ic] += dPhaseFluxV_dC[ke][ic];
      }

      // Distributing the viscous flux of phase i onto component
      CompositionalMultiphaseFlowUpwindHelperKernels::formPhaseComp( ip,
                                                                     k_up,
                                                                     stencilSize,
                                                                     seri,
                                                                     sesri,
                                                                     sei,
                                                                     phaseCompFrac,
                                                                     dPhaseCompFrac_dPres,
                                                                     dPhaseCompFrac_dComp,
                                                                     dCompFrac_dCompDens,
                                                                     phaseFluxV,
                                                                     dPhaseFluxV_dP,
                                                                     dPhaseFluxV_dC,
                                                                     compFlux,
                                                                     dCompFlux_dP,
                                                                     dCompFlux_dC );
      /***           GRAVITY TERM                ***/
      for( localIndex jp = 0; jp < NP; ++jp )
      {
        if( ip != jp )
        {
          real64 phaseFluxG{};
          real64 dPhaseFluxG_dP[MAX_STENCIL]{};
          real64 dPhaseFluxG_dC[MAX_STENCIL][NC]{};

          localIndex k_up_g = -1;
          localIndex k_up_og = -1;

          real64 gravHeadOther{};
          real64 dGravHeadOther_dP[NUM_ELEMS]{};
          real64 dGravHeadOther_dC[NUM_ELEMS][NC]{};
          real64 dPropOther_dC[NC]{};

          //Fetch gravHead for phase j!=i defined as \rho_j g dz/dx
          CompositionalMultiphaseFlowUpwindHelperKernels::formGravHead( jp,
                                                                        stencilSize,
                                                                        seri,
                                                                        sesri,
                                                                        sei,
                                                                        stencilWeights,
                                                                        gravCoef,
                                                                        dCompFrac_dCompDens,
                                                                        phaseMassDens,
                                                                        dPhaseMassDens_dPres,
                                                                        dPhaseMassDens_dComp,
                                                                        gravHeadOther,
                                                                        dGravHeadOther_dP,
                                                                        dGravHeadOther_dC,
                                                                        dPropOther_dC );

          // and the fractional flow for gravitational part as \lambda_i^{up}/\sum_{NP}(\lambda_k^{up}) with up decided upon
          // the Upwind strategy
          CompositionalMultiphaseFlowUpwindHelperKernels::formFracFlow< NC, NUM_ELEMS, MAX_STENCIL,
            CompositionalMultiphaseFlowUpwindHelperKernels::term::Gravity,
            CompositionalMultiphaseFlowUpwindHelperKernels::PU >( NP,
                                                                  ip,
                                                                  stencilSize,
                                                                  seri,
                                                                  sesri,
                                                                  sei,
                                                                  stencilWeights,
                                                                  totFlux_unw, //in fine should be a ElemnetViewConst once seq form are in place
                                                                  pres,
                                                                  dPres,
                                                                  gravCoef,
                                                                  dCompFrac_dCompDens,
                                                                  phaseMassDens,
                                                                  dPhaseMassDens_dPres,
                                                                  dPhaseMassDens_dComp,
                                                                  phaseMob,
                                                                  dPhaseMob_dPres,
                                                                  dPhaseMob_dComp,
                                                                  k_up_g,
                                                                  fflow,
                                                                  dFflow_dP,
                                                                  dFflow_dC );

          //mdensmultiply
          CompositionalMultiphaseFlowUpwindHelperKernels::mdensMultiply(
            ip,
            k_up_g,
            stencilSize,
            seri,
            sesri,
            sei,
            dCompFrac_dCompDens,
            phaseDens,
            dPhaseDens_dPres,
            dPhaseDens_dComp,
            fflow,
            dFflow_dP,
            dFflow_dC
            );



          //Eventually get the mobility of the second phase
          real64 mobOther{};
          real64 dMobOther_dP{};
          real64 dMobOther_dC[NC]{};

          // and the other mobility for gravitational part as \lambda_j^{up} with up decided upon
          // the Upwind strategy - Note that it should be the same as the gravitational fractional flow

          CompositionalMultiphaseFlowUpwindHelperKernels::upwindMob< NC, NUM_ELEMS,
            CompositionalMultiphaseFlowUpwindHelperKernels::term::Gravity,
            CompositionalMultiphaseFlowUpwindHelperKernels::PU >( NP,
                                                                  jp,
                                                                  stencilSize,
                                                                  seri,
                                                                  sesri,
                                                                  sei,
                                                                  stencilWeights,
                                                                  totFlux_unw, //in fine should be a ElemnetViewConst once seq form are in place
                                                                  pres,
                                                                  dPres,
                                                                  gravCoef,
                                                                  dCompFrac_dCompDens,
                                                                  phaseMassDens,
                                                                  dPhaseMassDens_dPres,
                                                                  dPhaseMassDens_dComp,
                                                                  phaseMob,
                                                                  dPhaseMob_dPres,
                                                                  dPhaseMob_dComp,
                                                                  k_up_og,
                                                                  mobOther,
                                                                  dMobOther_dP,
                                                                  dMobOther_dC );



          // Assembling gravitational flux phase-wise as \phi_{i,g} = \sum_{k\nei} \lambda_k^{up,g} f_k^{up,g} (G_i - G_k)
          phaseFluxG -= fflow * mobOther * ( gravHead - gravHeadOther );
          dPhaseFluxG_dP[k_up_og] -= fflow * dMobOther_dP * ( gravHead - gravHeadOther );
          for( localIndex jc = 0; jc < NC; ++jc )
            dPhaseFluxG_dC[k_up_og][jc] -= fflow * dMobOther_dC[jc] * ( gravHead - gravHeadOther );

          //mob related part of dFflow_dP is only upstream defined but totMob related is defined everywhere
          for( localIndex ke = 0; ke < stencilSize; ++ke )
          {
            dPhaseFluxG_dP[ke] -= dFflow_dP[ke] * mobOther * ( gravHead - gravHeadOther );

            for( localIndex jc = 0; jc < NC; ++jc )
            {
              dPhaseFluxG_dC[ke][jc] -= dFflow_dC[ke][jc] * mobOther * ( gravHead - gravHeadOther );
            }
          }

          for( localIndex ke = 0; ke < NUM_ELEMS; ++ke )
          {
            dPhaseFluxG_dP[ke] -= fflow * mobOther * ( dGravHead_dP[ke] - dGravHeadOther_dP[ke] );
            for( localIndex jc = 0; jc < NC; ++jc )
            {
              dPhaseFluxG_dC[ke][jc] -= fflow * mobOther * ( dGravHead_dC[ke][jc] - dGravHeadOther_dC[ke][jc] );
            }
          }

          // Distributing the gravitational flux of phase i onto component
          CompositionalMultiphaseFlowUpwindHelperKernels::formPhaseComp( ip,
                                                                         k_up_g,
                                                                         stencilSize,
                                                                         seri,
                                                                         sesri,
                                                                         sei,
                                                                         phaseCompFrac,
                                                                         dPhaseCompFrac_dPres,
                                                                         dPhaseCompFrac_dComp,
                                                                         dCompFrac_dCompDens,
                                                                         phaseFluxG,
                                                                         dPhaseFluxG_dP,
                                                                         dPhaseFluxG_dC,
                                                                         compFlux,
                                                                         dCompFlux_dP,
                                                                         dCompFlux_dC );

          //update phaseFlux from gravitational part
          phaseFlux += phaseFluxG;
          for( localIndex ke = 0; ke < stencilSize; ++ke )
          {
            dPhaseFlux_dP[ke] += dPhaseFluxG_dP[ke];
            for( localIndex ic = 0; ic < NC; ++ic )
              dPhaseFlux_dC[ke][ic] += dPhaseFluxG_dC[ke][ic];
          }

        }
      }


    }
  }//end if UT_FORM

  CompositionalMultiphaseFlowUpwindHelperKernels::fillLocalJacobi< NC, MAX_STENCIL, NDOF >( compFlux,
                                                                                            dCompFlux_dP,
                                                                                            dCompFlux_dC,
                                                                                            stencilSize,
                                                                                            dt,
                                                                                            localFlux,
                                                                                            localFluxJacobian );
}

template< localIndex NC, typename STENCIL_TYPE , bool IS_UT_FORM >
void
FluxKernel::
  launch( localIndex const numPhases,
          STENCIL_TYPE const & stencil,
          globalIndex const rankOffset,
          ElementViewConst< arrayView1d< globalIndex const > > const & dofNumber,
          ElementViewConst< arrayView1d< integer const > > const & ghostRank,
          ElementViewConst< arrayView1d< real64 const > > const & pres,
          ElementViewConst< arrayView1d< real64 const > > const & dPres,
          ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
          ElementViewConst< arrayView2d< real64 const > > const & phaseMob,
          ElementViewConst< arrayView2d< real64 const > > const & dPhaseMob_dPres,
          ElementViewConst< arrayView3d< real64 const > > const & dPhaseMob_dComp,
          ElementViewConst< arrayView2d< real64 const > > const & dPhaseVolFrac_dPres,
          ElementViewConst< arrayView3d< real64 const > > const & dPhaseVolFrac_dComp,
          ElementViewConst< arrayView3d< real64 const > > const & dCompFrac_dCompDens,
          ElementViewConst< arrayView3d< real64 const > > const & phaseDens,
          ElementViewConst< arrayView3d< real64 const > > const & dPhaseDens_dPres,
          ElementViewConst< arrayView4d< real64 const > > const & dPhaseDens_dComp,
          ElementViewConst< arrayView3d< real64 const > > const & phaseMassDens,
          ElementViewConst< arrayView3d< real64 const > > const & dPhaseMassDens_dPres,
          ElementViewConst< arrayView4d< real64 const > > const & dPhaseMassDens_dComp,
          ElementViewConst< arrayView4d< real64 const > > const & phaseCompFrac,
          ElementViewConst< arrayView4d< real64 const > > const & dPhaseCompFrac_dPres,
          ElementViewConst< arrayView5d< real64 const > > const & dPhaseCompFrac_dComp,
          ElementViewConst< arrayView3d< real64 const > > const & phaseCapPressure,
          ElementViewConst< arrayView4d< real64 const > > const & dPhaseCapPressure_dPhaseVolFrac,
          integer const capPressureFlag,
          real64 const dt,
          CRSMatrixView< real64, globalIndex const > const & localMatrix,
          arrayView1d< real64 > const & localRhs )
{
  typename STENCIL_TYPE::IndexContainerViewConstType const & seri = stencil.getElementRegionIndices();
  typename STENCIL_TYPE::IndexContainerViewConstType const & sesri = stencil.getElementSubRegionIndices();
  typename STENCIL_TYPE::IndexContainerViewConstType const & sei = stencil.getElementIndices();
  typename STENCIL_TYPE::WeightContainerViewConstType const & weights = stencil.getWeights();

  localIndex constexpr NUM_ELEMS   = STENCIL_TYPE::NUM_POINT_IN_FLUX;
  localIndex constexpr MAX_STENCIL = STENCIL_TYPE::MAX_STENCIL_SIZE;

  forAll< parallelDevicePolicy<> >( stencil.size(), [=] GEOSX_HOST_DEVICE ( localIndex const iconn )
  {
    // TODO: hack! for MPFA, etc. must obtain proper size from e.g. seri
    localIndex const stencilSize = MAX_STENCIL;
    localIndex constexpr NDOF = NC + 1;

    stackArray1d< real64, NUM_ELEMS * NC >                      localFlux( NUM_ELEMS * NC );
    stackArray2d< real64, NUM_ELEMS * NC * MAX_STENCIL * NDOF > localFluxJacobian( NUM_ELEMS * NC, stencilSize * NDOF );

    std::cerr << " iconn :" << iconn <<std::endl;

    FluxKernel::compute< NC, NUM_ELEMS, MAX_STENCIL, IS_UT_FORM >( numPhases,
                                                       stencilSize,
                                                       seri[iconn],
                                                       sesri[iconn],
                                                       sei[iconn],
                                                       weights[iconn],
                                                       pres,
                                                       dPres,
                                                       gravCoef,
                                                       phaseMob,
                                                       dPhaseMob_dPres,
                                                       dPhaseMob_dComp,
                                                       dPhaseVolFrac_dPres,
                                                       dPhaseVolFrac_dComp,
                                                       dCompFrac_dCompDens,
                                                                   phaseDens,
                                                                   dPhaseDens_dPres,
                                                                   dPhaseDens_dComp,
                                                       phaseMassDens,
                                                       dPhaseMassDens_dPres,
                                                       dPhaseMassDens_dComp,
                                                       phaseCompFrac,
                                                       dPhaseCompFrac_dPres,
                                                       dPhaseCompFrac_dComp,
                                                       phaseCapPressure,
                                                       dPhaseCapPressure_dPhaseVolFrac,
                                                       capPressureFlag,
                                                       dt,
                                                       localFlux,
                                                       localFluxJacobian );

    // populate dof indices
    globalIndex dofColIndices[ MAX_STENCIL * NDOF ];
    for( localIndex i = 0; i < stencilSize; ++i )
    {
      globalIndex const offset = dofNumber[seri( iconn, i )][sesri( iconn, i )][sei( iconn, i )];

      for( localIndex jdof = 0; jdof < NDOF; ++jdof )
      {
        dofColIndices[i * NDOF + jdof] = offset + jdof;
      }
    }

    // TODO: apply equation/variable change transformation(s)

    // Add to residual/jacobian
    for( localIndex i = 0; i < NUM_ELEMS; ++i )
    {
      if( ghostRank[seri( iconn, i )][sesri( iconn, i )][sei( iconn, i )] < 0 )
      {
        globalIndex const globalRow = dofNumber[seri( iconn, i )][sesri( iconn, i )][sei( iconn, i )];
        localIndex const localRow = LvArray::integerConversion< localIndex >( globalRow - rankOffset );
        GEOSX_ASSERT_GE( localRow, 0 );
        GEOSX_ASSERT_GT( localMatrix.numRows(), localRow + NC );

        for( localIndex ic = 0; ic < NC; ++ic )
        {
          RAJA::atomicAdd( parallelDeviceAtomic{}, &localRhs[localRow + ic], localFlux[i * NC + ic] );
          localMatrix.addToRowBinarySearchUnsorted< parallelDeviceAtomic >( localRow + ic,
                                                                            dofColIndices,
                                                                            localFluxJacobian[i * NC + ic].dataIfContiguous(),
                                                                            stencilSize * NDOF );
        }
      }
    }
  } );
}

#define INST_FluxKernel( NC, STENCIL_TYPE, IS_UT_FORM ) \
  template \
  void FluxKernel:: \
    launch< NC, STENCIL_TYPE, IS_UT_FORM >( localIndex const numPhases, \
                                STENCIL_TYPE const & stencil, \
                                globalIndex const rankOffset, \
                                ElementViewConst< arrayView1d< globalIndex const > > const & dofNumber, \
                                ElementViewConst< arrayView1d< integer const > > const & ghostRank, \
                                ElementViewConst< arrayView1d< real64 const > > const & pres, \
                                ElementViewConst< arrayView1d< real64 const > > const & dPres, \
                                ElementViewConst< arrayView1d< real64 const > > const & gravCoef, \
                                ElementViewConst< arrayView2d< real64 const > > const & phaseMob, \
                                ElementViewConst< arrayView2d< real64 const > > const & dPhaseMob_dPres, \
                                ElementViewConst< arrayView3d< real64 const > > const & dPhaseMob_dComp, \
                                ElementViewConst< arrayView2d< real64 const > > const & dPhaseVolFrac_dPres, \
                                ElementViewConst< arrayView3d< real64 const > > const & dPhaseVolFrac_dComp, \
                                ElementViewConst< arrayView3d< real64 const > > const & dCompFrac_dCompDens, \
                                ElementViewConst< arrayView3d< real64 const > > const & phaseDens, \
                                ElementViewConst< arrayView3d< real64 const > > const & dPhaseDens_dPres, \
                                ElementViewConst< arrayView4d< real64 const > > const & dPhaseDens_dComp, \
                                ElementViewConst< arrayView3d< real64 const > > const & phaseMassDens, \
                                ElementViewConst< arrayView3d< real64 const > > const & dPhaseMassDens_dPres, \
                                ElementViewConst< arrayView4d< real64 const > > const & dPhaseMassDens_dComp, \
                                ElementViewConst< arrayView4d< real64 const > > const & phaseCompFrac, \
                                ElementViewConst< arrayView4d< real64 const > > const & dPhaseCompFrac_dPres, \
                                ElementViewConst< arrayView5d< real64 const > > const & dPhaseCompFrac_dComp, \
                                ElementViewConst< arrayView3d< real64 const > > const & phaseCapPressure, \
                                ElementViewConst< arrayView4d< real64 const > > const & dPhaseCapPressure_dPhaseVolFrac, \
                                integer const capPressureFlag, \
                                real64 const dt, \
                                CRSMatrixView< real64, globalIndex const > const & localMatrix, \
                                arrayView1d< real64 > const & localRhs )

INST_FluxKernel( 1, CellElementStencilTPFA, false );
INST_FluxKernel( 2, CellElementStencilTPFA, false );
INST_FluxKernel( 3, CellElementStencilTPFA, false );
INST_FluxKernel( 4, CellElementStencilTPFA, false );
INST_FluxKernel( 5, CellElementStencilTPFA, false );

INST_FluxKernel( 1, CellElementStencilTPFA, true );
INST_FluxKernel( 2, CellElementStencilTPFA, true );
INST_FluxKernel( 3, CellElementStencilTPFA, true );
INST_FluxKernel( 4, CellElementStencilTPFA, true );
INST_FluxKernel( 5, CellElementStencilTPFA, true );

INST_FluxKernel( 1, FaceElementStencil, false);
INST_FluxKernel( 2, FaceElementStencil, false );
INST_FluxKernel( 3, FaceElementStencil, false );
INST_FluxKernel( 4, FaceElementStencil, false );
INST_FluxKernel( 5, FaceElementStencil, false );

INST_FluxKernel( 1, FaceElementStencil, true);
INST_FluxKernel( 2, FaceElementStencil, true );
INST_FluxKernel( 3, FaceElementStencil, true );
INST_FluxKernel( 4, FaceElementStencil, true );
INST_FluxKernel( 5, FaceElementStencil, true );

#undef INST_FluxKernel

/******************************** VolumeBalanceKernel ********************************/

template< localIndex NC, localIndex NP >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
VolumeBalanceKernel::
  compute( real64 const & volume,
           real64 const & porosityRef,
           real64 const & pvMult,
           real64 const & dPvMult_dPres,
           arraySlice1d< real64 const > const & phaseVolFrac,
           arraySlice1d< real64 const > const & dPhaseVolFrac_dPres,
           arraySlice2d< real64 const > const & dPhaseVolFrac_dCompDens,
           real64 & localVolBalance,
           real64 * const localVolBalanceJacobian )
{
  localIndex constexpr NDOF = NC + 1;

  real64 const poro     = porosityRef * pvMult;
  real64 const dPoro_dP = porosityRef * dPvMult_dPres;

  real64 const poreVol     = volume * poro;
  real64 const dPoreVol_dP = volume * dPoro_dP;

  localVolBalance = 1.0;
  for( localIndex i = 0; i < NDOF; ++i )
  {
    localVolBalanceJacobian[i] = 0.0;
  }

  // sum contributions to component accumulation from each phase
  for( localIndex ip = 0; ip < NP; ++ip )
  {
    localVolBalance -= phaseVolFrac[ip];
    localVolBalanceJacobian[0] -= dPhaseVolFrac_dPres[ip];

    for( localIndex jc = 0; jc < NC; ++jc )
    {
      localVolBalanceJacobian[jc+1] -= dPhaseVolFrac_dCompDens[ip][jc];
    }
  }

  // scale saturation-based volume balance by pore volume (for better scaling w.r.t. other equations)
  for( localIndex idof = 0; idof < NDOF; ++idof )
  {
    localVolBalanceJacobian[idof] *= poreVol;
  }
  localVolBalanceJacobian[0] += dPoreVol_dP * localVolBalance;
  localVolBalance *= poreVol;
}

template< localIndex NC, localIndex NP >
void
VolumeBalanceKernel::
  launch( localIndex const size,
          globalIndex const rankOffset,
          arrayView1d< globalIndex const > const & dofNumber,
          arrayView1d< integer const > const & elemGhostRank,
          arrayView1d< real64 const > const & volume,
          arrayView1d< real64 const > const & porosityRef,
          arrayView2d< real64 const > const & pvMult,
          arrayView2d< real64 const > const & dPvMult_dPres,
          arrayView2d< real64 const > const & phaseVolFrac,
          arrayView2d< real64 const > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const > const & dPhaseVolFrac_dCompDens,
          CRSMatrixView< real64, globalIndex const > const & localMatrix,
          arrayView1d< real64 > const & localRhs )
{
  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const ei )
  {
    if( elemGhostRank[ei] >= 0 )
      return;

    localIndex constexpr NDOF = NC + 1;

    real64 localVolBalance;
    real64 localVolBalanceJacobian[NDOF];

    compute< NC, NP >( volume[ei],
                       porosityRef[ei],
                       pvMult[ei][0],
                       dPvMult_dPres[ei][0],
                       phaseVolFrac[ei],
                       dPhaseVolFrac_dPres[ei],
                       dPhaseVolFrac_dCompDens[ei],
                       localVolBalance,
                       localVolBalanceJacobian );

    // get equation/dof indices
    localIndex const localRow = dofNumber[ei] + NC - rankOffset;
    globalIndex dofIndices[NDOF];
    for( localIndex jdof = 0; jdof < NDOF; ++jdof )
    {
      dofIndices[jdof] = dofNumber[ei] + jdof;
    }

    // TODO: apply equation/variable change transformation(s)

    // add contribution to residual and jacobian
    localRhs[localRow] += localVolBalance;
    localMatrix.addToRow< serialAtomic >( localRow,
                                          dofIndices,
                                          localVolBalanceJacobian,
                                          NDOF );
  } );
}

#define INST_VolumeBalanceKernel( NC, NP ) \
  template \
  void VolumeBalanceKernel:: \
    launch< NC, NP >( localIndex const size, \
                      globalIndex const rankOffset, \
                      arrayView1d< globalIndex const > const & dofNumber, \
                      arrayView1d< integer const > const & elemGhostRank, \
                      arrayView1d< real64 const > const & volume, \
                      arrayView1d< real64 const > const & porosityRef, \
                      arrayView2d< real64 const > const & pvMult, \
                      arrayView2d< real64 const > const & dPvMult_dPres, \
                      arrayView2d< real64 const > const & phaseVolFrac, \
                      arrayView2d< real64 const > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 const > const & dPhaseVolFrac_dCompDens, \
                      CRSMatrixView< real64, globalIndex const > const & localMatrix, \
                      arrayView1d< real64 > const & localRhs )

INST_VolumeBalanceKernel( 1, 1 );
INST_VolumeBalanceKernel( 2, 1 );
INST_VolumeBalanceKernel( 3, 1 );
INST_VolumeBalanceKernel( 4, 1 );
INST_VolumeBalanceKernel( 5, 1 );

INST_VolumeBalanceKernel( 1, 2 );
INST_VolumeBalanceKernel( 2, 2 );
INST_VolumeBalanceKernel( 3, 2 );
INST_VolumeBalanceKernel( 4, 2 );
INST_VolumeBalanceKernel( 5, 2 );

INST_VolumeBalanceKernel( 1, 3 );
INST_VolumeBalanceKernel( 2, 3 );
INST_VolumeBalanceKernel( 3, 3 );
INST_VolumeBalanceKernel( 4, 3 );
INST_VolumeBalanceKernel( 5, 3 );

#undef INST_VolumeBalanceKernel

} // namespace CompositionalMultiphaseFlowKernels

} // namespace geosx
