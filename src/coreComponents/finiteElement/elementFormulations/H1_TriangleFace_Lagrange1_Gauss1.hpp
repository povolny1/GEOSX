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
 * @file H1_TriangleFace_Lagrange1_Gauss1.hpp
 */

#ifndef GEOSX_FINITEELEMENT_ELEMENTFORMULATIONS_H1TRIANGLEFACELAGRANGE1GAUSS1
#define GEOSX_FINITEELEMENT_ELEMENTFORMULATIONS_H1TRIANGLEFACELAGRANGE1GAUSS1

#include "FiniteElementBase.hpp"


namespace geosx
{
namespace finiteElement
{

/**
 * This class contains the kernel accessible functions specific to the
 * H1-conforming nodal linear triangular face finite element with a
 * 1-point Gaussian quadrature rule.
 *
 *
 *          3                                =====  ==  ==
 *           +                               Node   r   s
 *           |\_                             =====  ==  ==
 *           |  \_                           0      0   0
 *           |    \_           s             1      1   0
 *           |      \_         |             2      0   1
 *           |        \_       |             =====  ==  ==
 *           |          \      |
 *           +-----------+     *------r
 *          0              1
 *
 */
class H1_TriangleFace_Lagrange1_Gauss1 final : public FiniteElementBase
{
public:
  /// The number of nodes/support points per element.
  constexpr static localIndex numNodes = 3;

  /// The number of quadrature points per element.
  constexpr static localIndex numQuadraturePoints = 1;

  virtual ~H1_TriangleFace_Lagrange1_Gauss1() override
  {}

  GEOSX_HOST_DEVICE
  virtual localIndex getNumQuadraturePoints() const override
  {
    return numQuadraturePoints;
  }

  GEOSX_HOST_DEVICE
  static localIndex getNumQuadraturePoints( StackVariables const & GEOSX_UNUSED_PARAM( stack ) )
  {
    return numQuadraturePoints;
  }

  GEOSX_HOST_DEVICE
  virtual localIndex getNumSupportPoints() const override
  {
    return numNodes;
  }

  GEOSX_HOST_DEVICE
  static localIndex getNumSupportPoints( StackVariables const & GEOSX_UNUSED_PARAM( stack ) )
  {
    return numNodes;
  }

  /**
   * @brief Empty initialization method.
   * @param nodeManager The node manager.
   * @param edgeManager The edge manager.
   * @param faceManager The face manager.
   * @param cellSubRegion The cell sub-region for which the element has to be initialized.
   * @param initialization Object that holds initialization properties.
   */
  GEOSX_HOST_DEVICE
  static void fillInitialization( NodeManager const & GEOSX_UNUSED_PARAM( nodeManager ),
                                  EdgeManager const & GEOSX_UNUSED_PARAM( edgeManager ),
                                  FaceManager const & GEOSX_UNUSED_PARAM( faceManager ),
                                  CellElementSubRegion const & GEOSX_UNUSED_PARAM( cellSubRegion ),
                                  Initialization & GEOSX_UNUSED_PARAM( initialization )
                                  )
  {}

  /**
   * @brief Empty setup method.
   * @param cellIndex The index of the cell with respect to the cell sub region to which the element
   * has been initialized previously (see @ref initialize).
   * @param stack Object that holds stack variables.
   */
  GEOSX_HOST_DEVICE
  static void setupStack( localIndex const & GEOSX_UNUSED_PARAM( cellIndex ),
                          Initialization const & GEOSX_UNUSED_PARAM( initialization ),
                          StackVariables & GEOSX_UNUSED_PARAM( stack ) )
  {}

  /**
   * @brief Calculate shape functions values for each support point at a
   *   quadrature point.
   * @param q Index of the quadrature point.
   * @param N An array to pass back the shape function values for each support
   *          point.
   */
  GEOSX_HOST_DEVICE
  static void calcN( localIndex const q,
                     real64 ( &N )[numNodes] );

  /**
   * @brief Calculate shape functions values for each support point at a
   *   quadrature point.
   * @param q Index of the quadrature point.
   * @param stack Variables allocated on the stack as filled by @ref setupStack.
   * @param N An array to pass back the shape function values for each support
   *   point.
   */
  GEOSX_HOST_DEVICE
  GEOSX_FORCE_INLINE
  static void calcN( localIndex const q,
                     StackVariables const & GEOSX_UNUSED_PARAM( stack ),
                     real64 ( & N )[numNodes] )
  {
    return calcN( q, N );
  }

  /**
   * @brief Calculate the integration weights for a quadrature point.
   * @param q Index of the quadrature point.
   * @param X Array containing the coordinates of the support points.
   * @return The product of the quadrature rule weight and the determinate of
   *   the parent/physical transformation matrix.
   */
  GEOSX_HOST_DEVICE
  static real64 transformedQuadratureWeight( localIndex const q,
                                             real64 const (&X)[numNodes][3] );

  template< typename MATRIXTYPE >
  GEOSX_HOST_DEVICE
  GEOSX_FORCE_INLINE
  static void addGradGradStabilization( StackVariables const & stack, MATRIXTYPE & matrix )
  {}

private:
  /// The area of the element in the parent configuration.
  constexpr static real64 parentArea = 0.5;

  /// The weight of each quadrature point.
  constexpr static real64 weight = parentArea / numQuadraturePoints;

};

GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void
H1_TriangleFace_Lagrange1_Gauss1::
  calcN( localIndex const q,
         real64 (& N)[numNodes] )
{
  GEOSX_UNUSED_VAR( q );

  // single quadrature point (centroid), i.e.  r = s = 1/3
  N[0] = 1.0 / 3.0; // N0 = 1 - r - s
  N[1] = N[0];      // N1 = r
  N[2] = N[0];      // N2 = s
}

//*************************************************************************************************

GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
real64
H1_TriangleFace_Lagrange1_Gauss1::
  transformedQuadratureWeight( localIndex const q,
                               real64 const (&X)[numNodes][3] )
{
  GEOSX_UNUSED_VAR( q );
  real64 n[3] = { ( X[1][1] - X[0][1] ) * ( X[2][2] - X[0][2] ) - ( X[2][1] - X[0][1] ) * ( X[1][2] - X[0][2] ),
                  ( X[2][0] - X[0][0] ) * ( X[1][2] - X[0][2] ) - ( X[1][0] - X[0][0] ) * ( X[2][2] - X[0][2] ),
                  ( X[1][0] - X[0][0] ) * ( X[2][1] - X[0][1] ) - ( X[2][0] - X[0][0] ) * ( X[1][1] - X[0][1] )};
  return sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] ) * weight;
}

}
}
#endif //GEOSX_FINITEELEMENT_ELEMENTFORMULATIONS_H1TRIANGLEFACELAGRANGE1GAUSS1
