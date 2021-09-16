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
 * @file CompositionalMultiphaseFVMKernels.cpp
 */

#include "CompositionalMultiphaseFVMKernels.hpp"
#include "CompositionalMultiphaseFlowUpwindHelperKernels.hpp"

#include "finiteVolume/CellElementStencilTPFA.hpp"
#include "finiteVolume/SurfaceElementStencil.hpp"
#include "finiteVolume/EmbeddedSurfaceToCellStencil.hpp"
#include "finiteVolume/FaceElementToCellStencil.hpp"
#include "mesh/utilities/MeshMapUtilities.hpp"


namespace geosx
{

namespace CompositionalMultiphaseFVMKernels
{

/******************************** PhaseMobilityKernel ********************************/

template< localIndex NC, localIndex NP >
GEOSX_HOST_DEVICE
void
PhaseMobilityKernel::
  compute( arraySlice2d< real64 const, compflow::USD_COMP_DC - 1 > const & dCompFrac_dCompDens,
           arraySlice1d< real64 const, multifluid::USD_PHASE - 2 > const & GEOSX_UNUSED_PARAM( phaseDens ),
           arraySlice1d< real64 const, multifluid::USD_PHASE - 2 > const & GEOSX_UNUSED_PARAM( dPhaseDens_dPres ),
           arraySlice2d< real64 const, multifluid::USD_PHASE_DC - 2 > const & GEOSX_UNUSED_PARAM( dPhaseDens_dComp ),
           arraySlice1d< real64 const, multifluid::USD_PHASE - 2 > const & phaseVisc,
           arraySlice1d< real64 const, multifluid::USD_PHASE - 2 > const & dPhaseVisc_dPres,
           arraySlice2d< real64 const, multifluid::USD_PHASE_DC - 2 > const & dPhaseVisc_dComp,
           arraySlice1d< real64 const, relperm::USD_RELPERM - 2 > const & phaseRelPerm,
           arraySlice2d< real64 const, relperm::USD_RELPERM_DS - 2 > const & dPhaseRelPerm_dPhaseVolFrac,
           arraySlice1d< real64 const, compflow::USD_PHASE - 1 > const & phaseVolFrac,
           arraySlice1d< real64 const, compflow::USD_PHASE - 1 > const & dPhaseVolFrac_dPres,
           arraySlice2d< real64 const, compflow::USD_PHASE_DC - 1 > const & dPhaseVolFrac_dComp,
           arraySlice1d< real64, compflow::USD_PHASE - 1 > const & phaseMob,
           arraySlice1d< real64, compflow::USD_PHASE - 1 > const & dPhaseMob_dPres,
           arraySlice2d< real64, compflow::USD_PHASE_DC - 1 > const & dPhaseMob_dComp )
{
  real64 dRelPerm_dC[NC];
  real64 dVisc_dC[NC];

  for( localIndex ip = 0; ip < NP; ++ip )
  {

    // compute the phase mobility only if the phase is present
    bool const phaseExists = (phaseVolFrac[ip] > 0);
    if( !phaseExists )
    {
      phaseMob[ip] = 0.;
      dPhaseMob_dPres[ip] = 0.;
      for( localIndex jc = 0; jc < NC; ++jc )
      {
        dPhaseMob_dComp[ip][jc] = 0.;
      }
      continue;
    }

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

template< localIndex NC, localIndex NP >
void PhaseMobilityKernel::
  launch( localIndex const size,
          arrayView3d< real64 const, compflow::USD_COMP_DC > const & dCompFrac_dCompDens,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseDens,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseDens_dPres,
          arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseDens_dComp,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseVisc_dPres,
          arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseVisc_dComp,
          arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm,
          arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac,
          arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac,
          arrayView2d< real64 const, compflow::USD_PHASE > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const, compflow::USD_PHASE_DC > const & dPhaseVolFrac_dComp,
          arrayView2d< real64, compflow::USD_PHASE > const & phaseMob,
          arrayView2d< real64, compflow::USD_PHASE > const & dPhaseMob_dPres,
          arrayView3d< real64, compflow::USD_PHASE_DC > const & dPhaseMob_dComp )
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
                       phaseVolFrac[a],
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
          arrayView3d< real64 const, compflow::USD_COMP_DC > const & dCompFrac_dCompDens,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseDens,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseDens_dPres,
          arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseDens_dComp,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseVisc_dPres,
          arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseVisc_dComp,
          arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm,
          arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac,
          arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac,
          arrayView2d< real64 const, compflow::USD_PHASE > const & dPhaseVolFrac_dPres,
          arrayView3d< real64 const, compflow::USD_PHASE_DC > const & dPhaseVolFrac_dComp,
          arrayView2d< real64, compflow::USD_PHASE > const & phaseMob,
          arrayView2d< real64, compflow::USD_PHASE > const & dPhaseMob_dPres,
          arrayView3d< real64, compflow::USD_PHASE_DC > const & dPhaseMob_dComp )
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
                       phaseVolFrac[a],
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
                      arrayView3d< real64 const, compflow::USD_COMP_DC > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseDens, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseVisc_dPres, \
                      arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseVisc_dComp, \
                      arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm, \
                      arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 const, compflow::USD_PHASE_DC > const & dPhaseVolFrac_dComp, \
                      arrayView2d< real64, compflow::USD_PHASE > const & phaseMob, \
                      arrayView2d< real64, compflow::USD_PHASE > const & dPhaseMob_dPres, \
                      arrayView3d< real64, compflow::USD_PHASE_DC > const & dPhaseMob_dComp ); \
  template \
  void \
  PhaseMobilityKernel:: \
    launch< NC, NP >( SortedArrayView< localIndex const > const & targetSet, \
                      arrayView3d< real64 const, compflow::USD_COMP_DC > const & dCompFrac_dCompDens, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseDens, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseDens_dPres, \
                      arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseDens_dComp, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & dPhaseVisc_dPres, \
                      arrayView4d< real64 const, multifluid::USD_PHASE_DC > const & dPhaseVisc_dComp, \
                      arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm, \
                      arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & dPhaseVolFrac_dPres, \
                      arrayView3d< real64 const, compflow::USD_PHASE_DC > const & dPhaseVolFrac_dComp, \
                      arrayView2d< real64, compflow::USD_PHASE > const & phaseMob, \
                      arrayView2d< real64, compflow::USD_PHASE > const & dPhaseMob_dPres, \
                      arrayView3d< real64, compflow::USD_PHASE_DC > const & dPhaseMob_dComp )

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


/******************************** FluxKernel ********************************/

template< CompositionalMultiphaseFlowUpwindHelperKernels::term T >
using UpwindSchemeType = CompositionalMultiphaseFlowUpwindHelperKernels::HybridUpwind< T >;
//using UpwindSchemeType = CompositionalMultiphaseFlowUpwindHelperKernels::PhasePotentialUpwind<T>;

template< localIndex NC, localIndex MAX_NUM_ELEMS, localIndex MAX_STENCIL_SIZE, bool IS_UT_FORM >
GEOSX_HOST_DEVICE
void
FluxKernel::
  compute( localIndex const numPhases,
           localIndex const stencilSize,
           localIndex const numFluxElems,
           arraySlice1d< localIndex const > const seri,
           arraySlice1d< localIndex const > const sesri,
           arraySlice1d< localIndex const > const sei,
           real64 const (&transmissibility)[2],
           real64 const (&dTrans_dPres)[2],
           ElementViewConst< arrayView1d< real64 const > > const & pres,
           ElementViewConst< arrayView1d< real64 const > > const & dPres,
           ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
           ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseMob,
           ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseMob_dPres,
           ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseMob_dComp,
           ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseVolFrac_dPres,
           ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseVolFrac_dComp,
           ElementViewConst< arrayView3d< real64 const, compflow::USD_COMP_DC > > const & dCompFrac_dCompDens,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseDens_dPres,
           ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseDens_dComp,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseMassDens_dPres,
           ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseMassDens_dComp,
           ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac,
           ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & dPhaseCompFrac_dPres,
           ElementViewConst< arrayView5d< real64 const, multifluid::USD_PHASE_COMP_DC > > const & dPhaseCompFrac_dComp,
           ElementViewConst< arrayView3d< real64 const, cappres::USD_CAPPRES > > const & phaseCapPressure,
           ElementViewConst< arrayView4d< real64 const, cappres::USD_CAPPRES_DS > > const & dPhaseCapPressure_dPhaseVolFrac,
           integer const capPressureFlag,
           real64 const dt,
           arraySlice1d< real64 > const localFlux,
           arraySlice2d< real64 > const localFluxJacobian )
{
  localIndex constexpr NDOF = NC + 1;

//  stackArray1d< real64, NC > compFlux( NC );
//  stackArray2d< real64, MAX_STENCIL_SIZE * NC > dCompFlux_dP( stencilSize, NC );
//  stackArray3d< real64, MAX_STENCIL_SIZE * NC * NC > dCompFlux_dC( stencilSize, NC, NC );

  real64 compFlux[NC]{};
  real64 dCompFlux_dP[MAX_STENCIL_SIZE][NC]{};
  real64 dCompFlux_dC[MAX_STENCIL_SIZE][NC][NC]{};

  real64 totFlux_unw{};
  real64 dTotFlux_dP[MAX_STENCIL_SIZE]{};
  real64 dTotFlux_dC[MAX_STENCIL_SIZE][NC]{};


  using UpwindHelpers = geosx::CompositionalMultiphaseFlowUpwindHelperKernels::UpwindHelpers;

  // loop over phases, compute and upwind phase flux and sum contributions to each component's flux
  for( localIndex ip = 0; ip < numPhases; ++ip )
  {

    // create local work arrays
    real64 phaseFlux{};
    real64 dPhaseFlux_dP[MAX_STENCIL_SIZE]{};
    real64 dPhaseFlux_dC[MAX_STENCIL_SIZE][NC]{};

    localIndex k_up = -1;

    UpwindHelpers::formPPUVelocity< NC, MAX_NUM_ELEMS, MAX_STENCIL_SIZE >( numPhases,
                                                                           ip,
                                                                           stencilSize,
                                                                           seri,
                                                                           sesri,
                                                                           sei,
//      stencilWeights,
                                                                           transmissibility,
                                                                           dTrans_dPres,
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
                                                                           dPhaseFlux_dC );

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
    UpwindHelpers::mdensMultiply( ip,
                                  k_up,
                                  stencilSize,
                                  numFluxElems,
                                  seri,
                                  sesri,
                                  sei,
                                  dCompFrac_dCompDens,
                                  phaseDens,
                                  dPhaseDens_dPres,
                                  dPhaseDens_dComp,
                                  phaseFlux,
                                  dPhaseFlux_dP,
                                  dPhaseFlux_dC );

    if( !IS_UT_FORM ) // skip  if you intend to use fixed total velocity formulation
    {
      UpwindHelpers::formPhaseComp( ip,
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
    for( localIndex ip = 0; ip < numPhases; ++ip )
    {
      // choose upstream cell
      // create local work arrays
      real64 phaseFlux{};
      real64 dPhaseFlux_dP[MAX_STENCIL_SIZE]{};
      real64 dPhaseFlux_dC[MAX_STENCIL_SIZE][NC]{};

      real64 phaseFluxV{};
      real64 dPhaseFluxV_dP[MAX_STENCIL_SIZE]{};
      real64 dPhaseFluxV_dC[MAX_STENCIL_SIZE][NC]{};

      real64 fflow{};
      real64 dFflow_dP[MAX_STENCIL_SIZE]{};
      real64 dFflow_dC[MAX_STENCIL_SIZE][NC]{};


      // and the fractional flow for viscous part as \lambda_i^{up}/\sum_{NP}(\lambda_j^{up}) with up decided upon
      // the Upwind strategy
      localIndex k_up = -1;
      UpwindHelpers::formFracFlow< NC, MAX_NUM_ELEMS, MAX_STENCIL_SIZE,
        CompositionalMultiphaseFlowUpwindHelperKernels::term::Viscous,
        UpwindSchemeType >( numPhases,
                            ip,
                            stencilSize,
                            numFluxElems,
                            seri,
                            sesri,
                            sei,
//                            stencilWeights,
                            transmissibility,
                            dTrans_dPres,
                            totFlux_unw,
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
                            dPhaseVolFrac_dPres,
                            dPhaseVolFrac_dComp,
                            phaseCapPressure,
                            dPhaseCapPressure_dPhaseVolFrac,
                            capPressureFlag,
                            k_up,
                            fflow,
                            dFflow_dP,
                            dFflow_dC );

      //mdensmultiply
      UpwindHelpers::mdensMultiply( ip,
                                    k_up,
                                    stencilSize,
                                    numFluxElems,
                                    seri,
                                    sesri,
                                    sei,
                                    dCompFrac_dCompDens,
                                    phaseDens,
                                    dPhaseDens_dPres,
                                    dPhaseDens_dComp,
                                    fflow,
                                    dFflow_dP,
                                    dFflow_dC );

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

      // accumulate in the flux and its derivatives
      phaseFlux += phaseFluxV;
      for( localIndex ke = 0; ke < stencilSize; ++ke )
      {
        dPhaseFlux_dP[ke] += dPhaseFluxV_dP[ke];
        for( localIndex ic = 0; ic < NC; ++ic )
          dPhaseFlux_dC[ke][ic] += dPhaseFluxV_dC[ke][ic];
      }

      // Distributing the viscous flux of phase i onto component
      UpwindHelpers::formPhaseComp( ip,
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
      // upwind flag for main phase and other phases
      localIndex k_up_g = -1;
      localIndex k_up_og = -1;

      real64 phaseFluxG{};
      real64 dPhaseFluxG_dP[MAX_STENCIL_SIZE]{};
      real64 dPhaseFluxG_dC[MAX_STENCIL_SIZE][NC]{};

      UpwindHelpers::formPotFluxes< NC,
        CompositionalMultiphaseFlowUpwindHelperKernels::term::Gravity,
        MAX_NUM_ELEMS, MAX_STENCIL_SIZE, MAX_NUM_ELEMS, UpwindSchemeType >( numPhases,
                                                                            ip,
                                                                            stencilSize,
                                                                            numFluxElems,
                                                                            seri,
                                                                            sesri,
                                                                            sei,
//        stencilWeights,
                                                                            transmissibility,
                                                                            dTrans_dPres,
                                                                            totFlux_unw,
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
                                                                            phaseCapPressure,
                                                                            dPhaseCapPressure_dPhaseVolFrac,
                                                                            capPressureFlag,
                                                                            k_up_g,
                                                                            k_up_og,
                                                                            phaseFlux,
                                                                            dPhaseFlux_dP,
                                                                            dPhaseFlux_dC );

      // Distributing the gravitational flux of phase i onto component
      UpwindHelpers::formPhaseComp( ip,
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

      //update phaseFlux from gravitational
      phaseFlux += phaseFluxG;
      for( localIndex ke = 0; ke < stencilSize; ++ke )
      {
        dPhaseFlux_dP[ke] += dPhaseFluxG_dP[ke];
        for( localIndex ic = 0; ic < NC; ++ic )
          dPhaseFlux_dC[ke][ic] += dPhaseFluxG_dC[ke][ic];
      }


      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////Adding capillary
      /// contribution
      /***           CAPILLARY TERM                ***/
      if( capPressureFlag )
      {
        localIndex k_up_pc = -1;
        localIndex k_up_opc = -1;

        real64 phaseFluxCap{};
        real64 dPhaseFluxCap_dP[MAX_STENCIL_SIZE]{};
        real64 dPhaseFluxCap_dC[MAX_STENCIL_SIZE][NC]{};

        UpwindHelpers::formPotFluxes< NC,
          CompositionalMultiphaseFlowUpwindHelperKernels::term::Capillary,
          MAX_NUM_ELEMS, MAX_STENCIL_SIZE, MAX_STENCIL_SIZE, UpwindSchemeType >( numPhases,
                                                                                 ip,
                                                                                 stencilSize,
                                                                                 numFluxElems,
                                                                                 seri,
                                                                                 sesri,
                                                                                 sei,
//          stencilWeights,
                                                                                 transmissibility,
                                                                                 dTrans_dPres,
                                                                                 totFlux_unw,
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
                                                                                 phaseCapPressure,
                                                                                 dPhaseCapPressure_dPhaseVolFrac,
                                                                                 capPressureFlag,
                                                                                 k_up_pc,
                                                                                 k_up_opc,
                                                                                 phaseFluxCap,
                                                                                 dPhaseFluxCap_dP,
                                                                                 dPhaseFluxCap_dC );

        // Distributing the gravitational flux of phase i onto component
        UpwindHelpers::formPhaseComp( ip,
                                      k_up_pc,
                                      stencilSize,
                                      seri,
                                      sesri,
                                      sei,
                                      phaseCompFrac,
                                      dPhaseCompFrac_dPres,
                                      dPhaseCompFrac_dComp,
                                      dCompFrac_dCompDens,
                                      phaseFluxCap,
                                      dPhaseFluxCap_dP,
                                      dPhaseFluxCap_dC,
                                      compFlux,
                                      dCompFlux_dP,
                                      dCompFlux_dC );

        //update phaseFlux from gravitational and capillary part
        phaseFlux += phaseFluxCap;
        for( localIndex ke = 0; ke < stencilSize; ++ke )
        {
          dPhaseFlux_dP[ke] += dPhaseFluxCap_dP[ke];
          for( localIndex ic = 0; ic < NC; ++ic )
            dPhaseFlux_dC[ke][ic] += dPhaseFluxCap_dC[ke][ic];
        }

      }//end if capPressureFlag

//      std::cout << " \n ------ \n";

    }
  }//end if UT_FORM

  UpwindHelpers::fillLocalJacobi< NC, MAX_STENCIL_SIZE, NDOF >( compFlux,
                                                                dCompFlux_dP,
                                                                dCompFlux_dC,
                                                                stencilSize,
                                                                dt,
                                                                localFlux,
                                                                localFluxJacobian );
}

template< localIndex NC, typename STENCILWRAPPER_TYPE, bool IS_UT_FORM >
void
FluxKernel::
  launch( localIndex const numPhases,
          STENCILWRAPPER_TYPE const & stencilWrapper,
          globalIndex const rankOffset,
          ElementViewConst< arrayView1d< globalIndex const > > const & dofNumber,
          ElementViewConst< arrayView1d< integer const > > const & ghostRank,
          ElementViewConst< arrayView1d< real64 const > > const & pres,
          ElementViewConst< arrayView1d< real64 const > > const & dPres,
          ElementViewConst< arrayView3d< real64 const > > const & permeability,
          ElementViewConst< arrayView3d< real64 const > > const & dPerm_dPres,
          ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
          ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseMob,
          ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseMob_dPres,
          ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseMob_dComp,
          ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseVolFrac_dPres,
          ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseVolFrac_dComp,
          ElementViewConst< arrayView3d< real64 const, compflow::USD_COMP_DC > > const & dCompFrac_dCompDens,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseDens_dPres,
          ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseDens_dComp,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseMassDens_dPres,
          ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseMassDens_dComp,
          ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac,
          ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & dPhaseCompFrac_dPres,
          ElementViewConst< arrayView5d< real64 const, multifluid::USD_PHASE_COMP_DC > > const & dPhaseCompFrac_dComp,
          ElementViewConst< arrayView3d< real64 const, cappres::USD_CAPPRES > > const & phaseCapPressure,
          ElementViewConst< arrayView4d< real64 const, cappres::USD_CAPPRES_DS > > const & dPhaseCapPressure_dPhaseVolFrac,
          integer const capPressureFlag,
          real64 const dt,
          CRSMatrixView< real64, globalIndex const > const & localMatrix,
          arrayView1d< real64 > const & localRhs )
{
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & seri = stencilWrapper.getElementRegionIndices();
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & sesri = stencilWrapper.getElementSubRegionIndices();
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & sei = stencilWrapper.getElementIndices();

  constexpr localIndex MAX_NUM_ELEMS = STENCILWRAPPER_TYPE::NUM_POINT_IN_FLUX;
  constexpr localIndex MAX_STENCIL_SIZE = STENCILWRAPPER_TYPE::MAX_STENCIL_SIZE;

  forAll< parallelDevicePolicy<> >( stencilWrapper.size(), [=] GEOSX_HOST_DEVICE ( localIndex const iconn )
      {

        localIndex const stencilSize = meshMapUtilities::size1( sei, iconn );
        localIndex const numFluxElems = stencilWrapper.numPointsInFlux( iconn );
        constexpr localIndex NDOF = NC + 1;

        // working arrays
        stackArray1d< globalIndex, MAX_NUM_ELEMS * NDOF > dofColIndices( stencilSize * NDOF );
        stackArray1d< real64, MAX_NUM_ELEMS * NC >                      localFlux( numFluxElems * NC );
        stackArray2d< real64, MAX_NUM_ELEMS * NC * MAX_STENCIL_SIZE * NDOF > localFluxJacobian( numFluxElems * NC, stencilSize * NDOF );

        // compute transmissibility
        real64 transmissibility[STENCILWRAPPER_TYPE::MAX_NUM_OF_CONNECTIONS][2];
        real64 dTrans_dPres[STENCILWRAPPER_TYPE::MAX_NUM_OF_CONNECTIONS][2];

        stencilWrapper.computeWeights( iconn,
                                       permeability,
                                       dPerm_dPres,
                                       transmissibility,
                                       dTrans_dPres );

        FluxKernel::compute< NC, MAX_NUM_ELEMS, MAX_STENCIL_SIZE, IS_UT_FORM >( numPhases,
                                                                       stencilSize,
                                                                       numFluxElems,
                                                                       seri[iconn],
                                                                       sesri[iconn],
                                                                       sei[iconn],
                                                                       transmissibility[0],
                                                                       dTrans_dPres[0],
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
        for( localIndex i = 0; i < numFluxElems; ++i )
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
                                                                                dofColIndices.data(),
                                                                                localFluxJacobian[i * NC + ic].dataIfContiguous(),
                                                                                stencilSize * NDOF );
            }
          }
        }
      } );
}

#define INST_FluxKernel( NC, STENCILWRAPPER_TYPE, IS_UT_FORM ) \
  template \
  void FluxKernel:: \
    launch< NC, STENCILWRAPPER_TYPE, IS_UT_FORM >( localIndex const numPhases, \
                                                  STENCILWRAPPER_TYPE const & stencilWrapper, \
                                                  globalIndex const rankOffset, \
                                                  ElementViewConst< arrayView1d< globalIndex const > > const & dofNumber, \
                                                  ElementViewConst< arrayView1d< integer const > > const & ghostRank, \
                                                  ElementViewConst< arrayView1d< real64 const > > const & pres, \
                                                  ElementViewConst< arrayView1d< real64 const > > const & dPres, \
                                                  ElementViewConst< arrayView3d< real64 const > > const & permeability, \
                                                  ElementViewConst< arrayView3d< real64 const > > const & dPerm_dPres, \
                                                  ElementViewConst< arrayView1d< real64 const > > const & gravCoef, \
                                                  ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseMob, \
                                                  ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseMob_dPres, \
                                                  ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseMob_dComp, \
                                                  ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & dPhaseVolFrac_dPres, \
                                                  ElementViewConst< arrayView3d< real64 const, compflow::USD_PHASE_DC > > const & dPhaseVolFrac_dComp, \
                                                  ElementViewConst< arrayView3d< real64 const, compflow::USD_COMP_DC > > const & dCompFrac_dCompDens, \
                                                  ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens, \
                                                  ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseDens_dPres, \
                                                  ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseDens_dComp, \
                                                  ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens, \
                                                  ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & dPhaseMassDens_dPres, \
                                                  ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_DC > > const & dPhaseMassDens_dComp, \
                                                  ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac, \
                                                  ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & dPhaseCompFrac_dPres, \
                                                  ElementViewConst< arrayView5d< real64 const, multifluid::USD_PHASE_COMP_DC > > const & dPhaseCompFrac_dComp, \
                                                  ElementViewConst< arrayView3d< real64 const, cappres::USD_CAPPRES > > const & phaseCapPressure, \
                                                  ElementViewConst< arrayView4d< real64 const, cappres::USD_CAPPRES_DS > > const & dPhaseCapPressure_dPhaseVolFrac, \
                                                  integer const capPressureFlag, \
                                                  real64 const dt, \
                                                  CRSMatrixView< real64, globalIndex const > const & localMatrix, \
                                                  arrayView1d< real64 > const & localRhs )


INST_FluxKernel( 1, CellElementStencilTPFAWrapper, true );
INST_FluxKernel( 2, CellElementStencilTPFAWrapper, true );
INST_FluxKernel( 3, CellElementStencilTPFAWrapper, true  );
INST_FluxKernel( 4, CellElementStencilTPFAWrapper, true  );
INST_FluxKernel( 5, CellElementStencilTPFAWrapper, true  );

INST_FluxKernel( 1, SurfaceElementStencilWrapper, true  );
INST_FluxKernel( 2, SurfaceElementStencilWrapper, true  );
INST_FluxKernel( 3, SurfaceElementStencilWrapper, true  );
INST_FluxKernel( 4, SurfaceElementStencilWrapper, true  );
INST_FluxKernel( 5, SurfaceElementStencilWrapper, true  );

INST_FluxKernel( 1, EmbeddedSurfaceToCellStencilWrapper, true  );
INST_FluxKernel( 2, EmbeddedSurfaceToCellStencilWrapper, true  );
INST_FluxKernel( 3, EmbeddedSurfaceToCellStencilWrapper, true  );
INST_FluxKernel( 4, EmbeddedSurfaceToCellStencilWrapper, true  );
INST_FluxKernel( 5, EmbeddedSurfaceToCellStencilWrapper, true  );

INST_FluxKernel( 1, FaceElementToCellStencilWrapper, true );
INST_FluxKernel( 2, FaceElementToCellStencilWrapper, true  );
INST_FluxKernel( 3, FaceElementToCellStencilWrapper, true  );
INST_FluxKernel( 4, FaceElementToCellStencilWrapper, true  );
INST_FluxKernel( 5, FaceElementToCellStencilWrapper, true  );

INST_FluxKernel( 1, CellElementStencilTPFAWrapper, false  );
INST_FluxKernel( 2, CellElementStencilTPFAWrapper, false  );
INST_FluxKernel( 3, CellElementStencilTPFAWrapper, false  );
INST_FluxKernel( 4, CellElementStencilTPFAWrapper, false  );
INST_FluxKernel( 5, CellElementStencilTPFAWrapper, false  );

INST_FluxKernel( 1, SurfaceElementStencilWrapper, false  );
INST_FluxKernel( 2, SurfaceElementStencilWrapper, false  );
INST_FluxKernel( 3, SurfaceElementStencilWrapper, false  );
INST_FluxKernel( 4, SurfaceElementStencilWrapper, false  );
INST_FluxKernel( 5, SurfaceElementStencilWrapper, false  );

INST_FluxKernel( 1, EmbeddedSurfaceToCellStencilWrapper, false  );
INST_FluxKernel( 2, EmbeddedSurfaceToCellStencilWrapper, false  );
INST_FluxKernel( 3, EmbeddedSurfaceToCellStencilWrapper, false  );
INST_FluxKernel( 4, EmbeddedSurfaceToCellStencilWrapper, false  );
INST_FluxKernel( 5, EmbeddedSurfaceToCellStencilWrapper, false  );

INST_FluxKernel( 1, FaceElementToCellStencilWrapper, false );
INST_FluxKernel( 2, FaceElementToCellStencilWrapper, false  );
INST_FluxKernel( 3, FaceElementToCellStencilWrapper, false  );
INST_FluxKernel( 4, FaceElementToCellStencilWrapper, false  );
INST_FluxKernel( 5, FaceElementToCellStencilWrapper, false  );



#undef INST_FluxKernel


/******************************** CFLFluxKernel ********************************/

template< localIndex NC, localIndex NUM_ELEMS, localIndex MAX_STENCIL_SIZE >
GEOSX_HOST_DEVICE
void
CFLFluxKernel::
  compute( localIndex const numPhases,
           localIndex const stencilSize,
           real64 const & dt,
           arraySlice1d< localIndex const > const seri,
           arraySlice1d< localIndex const > const sesri,
           arraySlice1d< localIndex const > const sei,
           real64 const (&transmissibility)[2],
           ElementViewConst< arrayView1d< real64 const > > const & pres,
           ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
           ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseVolFrac,
           ElementViewConst< arrayView3d< real64 const, relperm::USD_RELPERM > > const & phaseRelPerm,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseVisc,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens,
           ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens,
           ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac,
           ElementView< arrayView2d< real64, compflow::USD_PHASE > > const & phaseOutflux,
           ElementView< arrayView2d< real64, compflow::USD_COMP > > const & compOutflux )
{
  // loop over phases, compute and upwind phase flux and sum contributions to each component's flux
  for( localIndex ip = 0; ip < numPhases; ++ip )
  {

    // clear working arrays
    real64 densMean{};

    // create local work arrays
    real64 presGrad{};
    real64 gravHead{};

    // calculate quantities on primary connected cells
    for( localIndex i = 0; i < NUM_ELEMS; ++i )
    {
      localIndex const er  = seri[i];
      localIndex const esr = sesri[i];
      localIndex const ei  = sei[i];

      // average density across the face
      densMean += 0.5 * phaseMassDens[er][esr][ei][0][ip];
    }

    //***** calculation of phase volumetric flux *****

    // compute potential difference MPFA-style
    for( localIndex i = 0; i < stencilSize; ++i )
    {
      localIndex const er  = seri[i];
      localIndex const esr = sesri[i];
      localIndex const ei  = sei[i];

      presGrad += transmissibility[i] * pres[er][esr][ei];
      gravHead += transmissibility[i] * densMean * gravCoef[er][esr][ei];
    }

    // *** upwinding ***

    // compute phase potential gradient
    real64 const potGrad = presGrad - gravHead;

    // choose upstream cell
    localIndex const k_up = (potGrad >= 0) ? 0 : 1;

    localIndex const er_up  = seri[k_up];
    localIndex const esr_up = sesri[k_up];
    localIndex const ei_up  = sei[k_up];

    // compute the phase flux only if the phase is present
    bool const phaseExists = (phaseVolFrac[er_up][esr_up][ei_up][ip] > 0);
    if( !phaseExists )
    {
      continue;
    }

    real64 const mobility = phaseRelPerm[er_up][esr_up][ei_up][0][ip] / phaseVisc[er_up][esr_up][ei_up][0][ip];

    // increment the phase (volumetric) outflux of the upstream cell
    real64 const absPhaseFlux = fabs( dt * mobility * potGrad );
    RAJA::atomicAdd( parallelDeviceAtomic{}, &phaseOutflux[er_up][esr_up][ei_up][ip], absPhaseFlux );

    // increment the component (mass/molar) outflux of the upstream cell
    for( localIndex ic = 0; ic < NC; ++ic )
    {
      real64 const absCompFlux = phaseCompFrac[er_up][esr_up][ei_up][0][ip][ic]
                                 * phaseDens[er_up][esr_up][ei_up][0][ip]
                                 * absPhaseFlux;
      RAJA::atomicAdd( parallelDeviceAtomic{}, &compOutflux[er_up][esr_up][ei_up][ic], absCompFlux );
    }
  }
}

template< localIndex NC, typename STENCILWRAPPER_TYPE >
void
CFLFluxKernel::
  launch( localIndex const numPhases,
          real64 const & dt,
          STENCILWRAPPER_TYPE const & stencilWrapper,
          ElementViewConst< arrayView1d< real64 const > > const & pres,
          ElementViewConst< arrayView1d< real64 const > > const & gravCoef,
          ElementViewConst< arrayView3d< real64 const > > const & permeability,
          ElementViewConst< arrayView3d< real64 const > > const & dPerm_dPres,
          ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseVolFrac,
          ElementViewConst< arrayView3d< real64 const, relperm::USD_RELPERM > > const & phaseRelPerm,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseVisc,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens,
          ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens,
          ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac,
          ElementView< arrayView2d< real64, compflow::USD_PHASE > > const & phaseOutflux,
          ElementView< arrayView2d< real64, compflow::USD_COMP > > const & compOutflux )
{
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & seri = stencilWrapper.getElementRegionIndices();
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & sesri = stencilWrapper.getElementSubRegionIndices();
  typename STENCILWRAPPER_TYPE::IndexContainerViewConstType const & sei = stencilWrapper.getElementIndices();

  localIndex constexpr NUM_ELEMS   = STENCILWRAPPER_TYPE::NUM_POINT_IN_FLUX;
  localIndex constexpr MAX_STENCIL_SIZE   = STENCILWRAPPER_TYPE::MAX_STENCIL_SIZE;

  forAll< parallelDevicePolicy<> >( stencilWrapper.size(), [=] GEOSX_HOST_DEVICE ( localIndex const iconn )
  {
    // compute transmissibility
    real64 transmissiblity[STENCILWRAPPER_TYPE::MAX_NUM_OF_CONNECTIONS][2];
    real64 dTrans_dPres[STENCILWRAPPER_TYPE::MAX_NUM_OF_CONNECTIONS][2];

    stencilWrapper.computeWeights( iconn,
                                   permeability,
                                   dPerm_dPres,
                                   transmissiblity,
                                   dTrans_dPres );

    localIndex const stencilSize = meshMapUtilities::size1( sei, iconn );

    CFLFluxKernel::compute< NC, NUM_ELEMS, MAX_STENCIL_SIZE >( numPhases,
                                                               stencilSize,
                                                               dt,
                                                               seri[iconn],
                                                               sesri[iconn],
                                                               sei[iconn],
                                                               transmissiblity[0],
                                                               pres,
                                                               gravCoef,
                                                               phaseVolFrac,
                                                               phaseRelPerm,
                                                               phaseVisc,
                                                               phaseDens,
                                                               phaseMassDens,
                                                               phaseCompFrac,
                                                               phaseOutflux,
                                                               compOutflux );
  } );
}

#define INST_CFLFluxKernel( NC, STENCILWRAPPER_TYPE ) \
  template \
  void CFLFluxKernel:: \
    launch< NC, STENCILWRAPPER_TYPE >( localIndex const numPhases, \
                                       real64 const & dt, \
                                       STENCILWRAPPER_TYPE const & stencil, \
                                       ElementViewConst< arrayView1d< real64 const > > const & pres, \
                                       ElementViewConst< arrayView1d< real64 const > > const & gravCoef, \
                                       ElementViewConst< arrayView3d< real64 const > > const & permeability, \
                                       ElementViewConst< arrayView3d< real64 const > > const & dPerm_dPres, \
                                       ElementViewConst< arrayView2d< real64 const, compflow::USD_PHASE > > const & phaseVolFrac, \
                                       ElementViewConst< arrayView3d< real64 const, relperm::USD_RELPERM > > const & phaseRelPerm, \
                                       ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseVisc, \
                                       ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseDens, \
                                       ElementViewConst< arrayView3d< real64 const, multifluid::USD_PHASE > > const & phaseMassDens, \
                                       ElementViewConst< arrayView4d< real64 const, multifluid::USD_PHASE_COMP > > const & phaseCompFrac, \
                                       ElementView< arrayView2d< real64, compflow::USD_PHASE > > const & phaseOutflux, \
                                       ElementView< arrayView2d< real64, compflow::USD_COMP > > const & compOutflux )

INST_CFLFluxKernel( 1, CellElementStencilTPFAWrapper );
INST_CFLFluxKernel( 2, CellElementStencilTPFAWrapper );
INST_CFLFluxKernel( 3, CellElementStencilTPFAWrapper );
INST_CFLFluxKernel( 4, CellElementStencilTPFAWrapper );
INST_CFLFluxKernel( 5, CellElementStencilTPFAWrapper );

INST_CFLFluxKernel( 1, SurfaceElementStencilWrapper );
INST_CFLFluxKernel( 2, SurfaceElementStencilWrapper );
INST_CFLFluxKernel( 3, SurfaceElementStencilWrapper );
INST_CFLFluxKernel( 4, SurfaceElementStencilWrapper );
INST_CFLFluxKernel( 5, SurfaceElementStencilWrapper );

INST_CFLFluxKernel( 1, EmbeddedSurfaceToCellStencilWrapper );
INST_CFLFluxKernel( 2, EmbeddedSurfaceToCellStencilWrapper );
INST_CFLFluxKernel( 3, EmbeddedSurfaceToCellStencilWrapper );
INST_CFLFluxKernel( 4, EmbeddedSurfaceToCellStencilWrapper );
INST_CFLFluxKernel( 5, EmbeddedSurfaceToCellStencilWrapper );

INST_CFLFluxKernel( 1, FaceElementToCellStencilWrapper );
INST_CFLFluxKernel( 2, FaceElementToCellStencilWrapper );
INST_CFLFluxKernel( 3, FaceElementToCellStencilWrapper );
INST_CFLFluxKernel( 4, FaceElementToCellStencilWrapper );
INST_CFLFluxKernel( 5, FaceElementToCellStencilWrapper );

#undef INST_CFLFluxKernel

/******************************** CFLKernel ********************************/

template< localIndex NP >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
CFLKernel::
  computePhaseCFL( real64 const & poreVol,
                   arraySlice1d< real64 const, compflow::USD_PHASE - 1 > phaseVolFrac,
                   arraySlice1d< real64 const, relperm::USD_RELPERM - 2 > phaseRelPerm,
                   arraySlice2d< real64 const, relperm::USD_RELPERM_DS - 2 > dPhaseRelPerm_dPhaseVolFrac,
                   arraySlice1d< real64 const, multifluid::USD_PHASE - 2 > phaseVisc,
                   arraySlice1d< real64 const, compflow::USD_PHASE - 1 > phaseOutflux,
                   real64 & phaseCFLNumber )
{
  // first, check which phases are mobile in the cell
  real64 mob[NP]{};
  localIndex mobilePhases[NP]{};
  localIndex numMobilePhases = 0;
  for( localIndex ip = 0; ip < NP; ++ip )
  {
    if( phaseVolFrac[ip] > 0 )
    {
      mob[ip] = phaseRelPerm[ip] / phaseVisc[ip];
      if( mob[ip] > minPhaseMobility )
      {
        mobilePhases[numMobilePhases] = ip;
        numMobilePhases++;
      }
    }
  }

  // then, depending on the regime, apply the appropriate CFL formula
  phaseCFLNumber = 0;

  // single-phase flow regime
  if( numMobilePhases == 1 )
  {
    phaseCFLNumber = phaseOutflux[mobilePhases[0]] / poreVol;
  }
  // two-phase flow regime
  else if( numMobilePhases == 2 )
  {
    // from Hui Cao's PhD thesis
    localIndex const ip0 = mobilePhases[0];
    localIndex const ip1 = mobilePhases[1];
    real64 const dMob_dVolFrac[2] = { dPhaseRelPerm_dPhaseVolFrac[ip0][ip0] / phaseVisc[ip0],
                                      -dPhaseRelPerm_dPhaseVolFrac[ip1][ip1] / phaseVisc[ip1] }; // using S0 = 1 - S1
    real64 const denom = 1. / ( poreVol * ( mob[ip0] + mob[ip1] ) );
    real64 const coef0 = denom * mob[ip1] / mob[ip0] * dMob_dVolFrac[ip0];
    real64 const coef1 = -denom * mob[ip0] / mob[ip1] * dMob_dVolFrac[ip1];

    phaseCFLNumber = fabs( coef0*phaseOutflux[ip0] + coef1*phaseOutflux[ip1] );
  }
  // three-phase flow regime
  else if( numMobilePhases == 3 )
  {
    // from Keith Coats, IMPES stability: Selection of stable timesteps (2003)
    real64 totalMob = 0.0;
    for( localIndex ip = 0; ip < numMobilePhases; ++ip )
    {
      totalMob += mob[ip];
    }

    real64 f[2][2]{};
    for( localIndex i = 0; i < 2; ++i )
    {
      for( localIndex j = 0; j < 2; ++j )
      {
        f[i][j]  = ( i == j )*totalMob - mob[i];
        f[i][j] /= (totalMob * mob[j]);
        real64 sum = 0;
        for( localIndex k = 0; k < 3; ++k )
        {
          sum += dPhaseRelPerm_dPhaseVolFrac[k][j] / phaseVisc[k]
                 * phaseOutflux[j];
        }
        f[i][j] *= sum;
      }
    }
    phaseCFLNumber = f[0][0] + f[1][1];
    phaseCFLNumber += sqrt( phaseCFLNumber*phaseCFLNumber - 4 * ( f[0][0]*f[1][1] - f[1][0]*f[0][1] ) );
    phaseCFLNumber = 0.5 * fabs( phaseCFLNumber ) / poreVol;
  }
}


template< localIndex NC >
GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
CFLKernel::
  computeCompCFL( real64 const & poreVol,
                  arraySlice1d< real64 const, compflow::USD_COMP - 1 > compDens,
                  arraySlice1d< real64 const, compflow::USD_COMP - 1 > compFrac,
                  arraySlice1d< real64 const, compflow::USD_COMP - 1 > compOutflux,
                  real64 & compCFLNumber )
{


  compCFLNumber = 0.0;
  for( localIndex ic = 0; ic < NC; ++ic )
  {
    if( compFrac[ic] > minComponentFraction )
    {
      real64 const compMoles = compDens[ic] * poreVol;
      real64 const CFL = compOutflux[ic] / compMoles;
      if( CFL > compCFLNumber )
      {
        compCFLNumber = CFL;
      }
    }
  }
}

template< localIndex NC, localIndex NP >
void
CFLKernel::
  launch( localIndex const size,
          arrayView1d< real64 const > const & volume,
          arrayView2d< real64 const > const & porosity,
          arrayView2d< real64 const, compflow::USD_COMP > const & compDens,
          arrayView2d< real64 const, compflow::USD_COMP > const & compFrac,
          arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac,
          arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm,
          arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac,
          arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc,
          arrayView2d< real64 const, compflow::USD_PHASE > const & phaseOutflux,
          arrayView2d< real64 const, compflow::USD_COMP > const & compOutflux,
          arrayView1d< real64 > const & phaseCFLNumber,
          arrayView1d< real64 > const & compCFLNumber,
          real64 & maxPhaseCFLNumber,
          real64 & maxCompCFLNumber )
{
  RAJA::ReduceMax< parallelDeviceReduce, real64 > subRegionPhaseCFLNumber( 0.0 );
  RAJA::ReduceMax< parallelDeviceReduce, real64 > subRegionCompCFLNumber( 0.0 );

  forAll< parallelDevicePolicy<> >( size, [=] GEOSX_HOST_DEVICE ( localIndex const ei )
  {
    real64 const poreVol = volume[ei] * porosity[ei][0];

    // phase CFL number
    real64 cellPhaseCFLNumber = 0.0;
    computePhaseCFL< NP >( poreVol,
                           phaseVolFrac[ei],
                           phaseRelPerm[ei][0],
                           dPhaseRelPerm_dPhaseVolFrac[ei][0],
                           phaseVisc[ei][0],
                           phaseOutflux[ei],
                           cellPhaseCFLNumber );
    subRegionPhaseCFLNumber.max( cellPhaseCFLNumber );
    phaseCFLNumber[ei] = cellPhaseCFLNumber;

    // component CFL number
    real64 cellCompCFLNumber = 0.0;
    computeCompCFL< NC >( poreVol,
                          compDens[ei],
                          compFrac[ei],
                          compOutflux[ei],
                          cellCompCFLNumber );
    subRegionCompCFLNumber.max( cellCompCFLNumber );
    compCFLNumber[ei] = cellCompCFLNumber;
  } );

  maxPhaseCFLNumber = subRegionPhaseCFLNumber.get();
  maxCompCFLNumber = subRegionCompCFLNumber.get();
}

#define INST_CFLKernel( NC, NP ) \
  template \
  void CFLKernel:: \
    launch< NC, NP >( localIndex const size, \
                      arrayView1d< real64 const > const & volume, \
                      arrayView2d< real64 const > const & porosity, \
                      arrayView2d< real64 const, compflow::USD_COMP > const & compDens, \
                      arrayView2d< real64 const, compflow::USD_COMP > const & compFrac, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & phaseVolFrac, \
                      arrayView3d< real64 const, relperm::USD_RELPERM > const & phaseRelPerm, \
                      arrayView4d< real64 const, relperm::USD_RELPERM_DS > const & dPhaseRelPerm_dPhaseVolFrac, \
                      arrayView3d< real64 const, multifluid::USD_PHASE > const & phaseVisc, \
                      arrayView2d< real64 const, compflow::USD_PHASE > const & phaseOutflux, \
                      arrayView2d< real64 const, compflow::USD_COMP > const & compOutflux, \
                      arrayView1d< real64 > const & phaseCFLNumber, \
                      arrayView1d< real64 > const & compCFLNumber, \
                      real64 & maxPhaseCFLNumber, \
                      real64 & maxCompCFLNumber )
INST_CFLKernel( 1, 2 );
INST_CFLKernel( 2, 2 );
INST_CFLKernel( 3, 2 );
INST_CFLKernel( 4, 2 );
INST_CFLKernel( 5, 2 );

INST_CFLKernel( 1, 3 );
INST_CFLKernel( 2, 3 );
INST_CFLKernel( 3, 3 );
INST_CFLKernel( 4, 3 );
INST_CFLKernel( 5, 3 );

#undef INST_CFLKernel

} // namespace CompositionalMultiphaseFVMKernels

} // namespace geosx
