/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 TotalEnergies
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 *  @file Coulomb.hpp
 */

#ifndef GEOSX_CONSTITUTIVE_CONTACT_COULOMB_HPP_
#define GEOSX_CONSTITUTIVE_CONTACT_COULOMB_HPP_

#include "ContactBase.hpp"

namespace geosx
{

namespace constitutive
{

/**
 * @class CoulombUpdates
 *
 * This class is used for in-kernel contact relation updates
 */
class CoulombUpdates : public ContactBaseUpdates
{
public:

  CoulombUpdates( real64 const & cohesion,
                  real64 const & frictionCoefficient )
    : m_cohesion( cohesion ),
    m_frictionCoefficient( frictionCoefficient )
  {}

  /// Default copy constructor
  CoulombUpdates( CoulombUpdates const & ) = default;

  /// Default move constructor
  CoulombUpdates( CoulombUpdates && ) = default;

  /// Deleted default constructor
  CoulombUpdates() = delete;

  /// Deleted copy assignment operator
  CoulombUpdates & operator=( CoulombUpdates const & ) = delete;

  /// Deleted move assignment operator
  CoulombUpdates & operator=( CoulombUpdates && ) =  delete;

  /**
   * @brief Evaluate the limit tangential traction norm and return the derivative wrt normal traction
   * @param[in] normalTraction the normal traction
   * @param[out] dLimitTangentialTractionNorm_dTraction the derivative of the limit tangential traction norm wrt normal traction
   * @return the limit tangential traction norm
   */
  GEOSX_HOST_DEVICE
  inline
  virtual real64 computeLimitTangentialTractionNorm( real64 const & normalTraction,
                                                     real64 & dLimitTangentialTractionNorm_dTraction ) const override;

private:

  /// The cohesion for each upper level dimension (i.e. cell) of *this
  real64 m_cohesion;

  /// The friction coefficient for each upper level dimension (i.e. cell) of *this
  real64 m_frictionCoefficient;

};


/**
 * @class Coulomb
 *
 * Class to provide a Coulomb friction model.
 */
class Coulomb : public ContactBase
{
public:

  /**
   * constructor
   * @param[in] name name of the instance in the catalog
   * @param[in] parent the group which contains this instance
   */
  Coulomb( string const & name, Group * const parent );

  /**
   * Default Destructor
   */
  virtual ~Coulomb() override;

  /**
   * @name Static Factory Catalog members and functions
   */
  ///@{

  /**
   * @return A string that is used to register/lookup this class in the registry
   */
  static string catalogName() { return "Coulomb"; }

  virtual string getCatalogName() const override { return catalogName(); }

  ///@}

  /**
   * @struct Set of "char const *" and keys for data specified in this class.
   */
  struct viewKeyStruct : public ContactBase::viewKeyStruct
  {
    /// string/key for cohesion
    static constexpr char const * cohesionString() { return "cohesion"; }

    /// string/key for friction angle input (in radians)
    static constexpr char const * frictionAngleString() { return "frictionAngle"; }

    /// string/key for friction coefficient
    static constexpr char const * frictionCoefficientString() { return "frictionCoefficient"; }
  };

  /**
   * @brief Const accessor for cohesion
   * @return A const reference to arrayView1d<real64 const> containing the
   *         cohesions (at every element).
   */
  real64 const & cohesion() const { return m_cohesion; }

  /**
   * @brief Const accessor for friction angle
   * @return A const reference to arrayView1d<real64 const> containing the
   *         friction coefficient (at every element).
   */
  real64 const & frictionCoefficient() const { return m_frictionCoefficient; }

  /// Type of kernel wrapper for in-kernel update
  using KernelWrapper = CoulombUpdates;

  /**
   * @brief Create an update kernel wrapper.
   * @return the wrapper
   */
  KernelWrapper createKernelWrapper() const;

protected:

  virtual void postProcessInput() override;

private:

  /// The cohesion for each upper level dimension (i.e. cell) of *this
  real64 m_cohesion;

  /// The friction angle for each upper level dimension (i.e. cell) of *this
  real64 m_frictionAngle;

  /// The friction coefficient for each upper level dimension (i.e. cell) of *this
  real64 m_frictionCoefficient;
};


GEOSX_HOST_DEVICE
real64 CoulombUpdates::computeLimitTangentialTractionNorm( real64 const & normalTraction,
                                                           real64 & dLimitTangentialTractionNorm_dTraction ) const
{
  dLimitTangentialTractionNorm_dTraction = m_frictionCoefficient;
  return ( m_cohesion - normalTraction * m_frictionCoefficient );
}

} /* end namespace constitutive */

} /* end namespace geosx */

#endif /* GEOSX_CONSTITUTIVE_CONTACT_COULOMB_HPP_ */
