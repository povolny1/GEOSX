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
 * @file SolidMechanicsLagrangianFEM.hpp
 */

#ifndef SOLID_MECHANICS_LAGRANGIAN_FEM_HPP_
#define SOLID_MECHANICS_LAGRANGIAN_FEM_HPP_

#include "physicsSolvers/SolverBase.hpp"

#include "common/TimingMacros.hpp"
#include "finiteElement/Kinematics.h"
#include "mesh/MeshForLoopInterface.hpp"
#include "MPI_Communications/CommunicationTools.hpp"
#include "systemSolverInterface/LinearSolverWrapper.hpp"


namespace ML_Epetra
{ class MultiLevelPreconditioner; }

namespace geosx
{
namespace dataRepository
{
class ManagedGroup;
}
class FieldSpecificationBase;
class FiniteElementBase;
class DomainPartition;

/**
 * @class SolidMechanicsLagrangianFEM
 *
 * This class implements a finite element solution to the equations of motion.
 */
class SolidMechanicsLagrangianFEM : public SolverBase
{
public:

  /**
   * @enum timeIntegrationOption
   *
   * The options for time integration
   */
  enum class timeIntegrationOption : int
  {
    QuasiStatic,    //!< QuasiStatic
    ImplicitDynamic,//!< ImplicitDynamic
    ExplicitDynamic //!< ExplicitDynamic
  };

  /**
   * Constructor
   * @param name The name of the solver instance
   * @param parent the parent group of the solver
   */
  SolidMechanicsLagrangianFEM( const std::string& name,
                                ManagedGroup * const parent );


  SolidMechanicsLagrangianFEM( SolidMechanicsLagrangianFEM const & ) = delete;
  SolidMechanicsLagrangianFEM( SolidMechanicsLagrangianFEM && ) = default ;

  SolidMechanicsLagrangianFEM& operator=( SolidMechanicsLagrangianFEM const & ) = delete;
  SolidMechanicsLagrangianFEM& operator=( SolidMechanicsLagrangianFEM && ) = delete ;

  /**
   * destructor
   */
  virtual ~SolidMechanicsLagrangianFEM() override;

  /**
   * @return The string that may be used to generate a new instance from the SolverBase::CatalogInterface::CatalogType
   */
  static string CatalogName() { return "SolidMechanics_LagrangianFEM"; }

  virtual void InitializePreSubGroups(ManagedGroup * const rootGroup) override;

  virtual void RegisterDataOnMesh( ManagedGroup * const MeshBody ) override final;

  /**
   * Set up the ML Preconditioner appropriate for the system generated by this solver
   * @param domain The DomainParition object
   * @param MLPrec A pointer to the preconditioner object.
   */
  void SetupMLPreconditioner( DomainPartition const & domain,
                              ML_Epetra::MultiLevelPreconditioner* MLPrec );

  /**
   * @defgroup Solver Interface Functions
   *
   * These functions provide the primary interface that is required for derived classes
   */
  /**@{*/
  virtual real64 SolverStep( real64 const& time_n,
                         real64 const& dt,
                         integer const cycleNumber,
                         DomainPartition * domain ) override;

  virtual real64 ExplicitStep( real64 const& time_n,
                                   real64 const& dt,
                                   integer const cycleNumber,
                                   DomainPartition * domain ) override;

  virtual void ImplicitStepSetup( real64 const& time_n,
                              real64 const& dt,
                              DomainPartition * const domain,
                              systemSolverInterface::EpetraBlockSystem * const blockSystem ) override;

  virtual void AssembleSystem ( DomainPartition * const domain,
                                  systemSolverInterface::EpetraBlockSystem * const blockSystem,
                                  real64 const time,
                                  real64 const dt ) override;

  virtual void SolveSystem( systemSolverInterface::EpetraBlockSystem * const blockSystem,
                            SystemSolverParameters const * const params ) override;

  virtual void ApplySystemSolution( systemSolverInterface::EpetraBlockSystem const * const blockSystem,
                            real64 const scalingFactor,
                            DomainPartition * const domain  ) override;

  virtual void ApplyBoundaryConditions( DomainPartition * const domain,
                                        systemSolverInterface::EpetraBlockSystem * const blockSystem,
                                        real64 const time,
                                        real64 const dt ) override;

  virtual real64
  CalculateResidualNorm(systemSolverInterface::EpetraBlockSystem const *const blockSystem, DomainPartition *const domain) override;

  virtual void ResetStateToBeginningOfStep( DomainPartition * const domain ) override;

  virtual void ImplicitStepComplete( real64 const & time,
                                 real64 const & dt,
                                 DomainPartition * const domain ) override;

  /**@}*/

  /**
   * Function to setup/allocate the linear system blocks
   * @param domain The DomainPartition object.
   * @param blockSystem the block system object to that holds the blocks that will be constructed
   */
  void SetupSystem ( DomainPartition * const domain,
                     systemSolverInterface::EpetraBlockSystem * const blockSystem );

  /**
   * Function to set the sparsity pattern
   * @param domain The DomainParition object
   * @param sparsity Pointer to the the sparsity graph
   */
  void SetSparsityPattern( DomainPartition const * const domain,
                           Epetra_FECrsGraph * const sparsity );

  /**
   * Function to set the global DOF numbers  for this solver.
   * @param domain        The DomainParition object
   * @param numLocalRows  The number of local rows on this process
   * @param numGlobalRows The number of global rows in the system
   * @param localIndices  The local indices associated with a local row??
   * @param offset        The offset for the row/globalDOF
   */
  void SetNumRowsAndTrilinosIndices( ManagedGroup * const domain,
                                     localIndex & numLocalRows,
                                     globalIndex & numGlobalRows,
                                     localIndex_array& localIndices,
                                     localIndex offset );

  /**
   * @brief Function to select which templated kernel function to call.
   * @tparam KERNELWRAPPER A struct or class that contains a "Launch<NUM_NODES_PER_ELEM,NUM_QUADRATURE_POINTS>()"
   *                       function to launch the kernel.
   * @tparam PARAMS Varaidic parameter pack to pass arguments to Launch function.
   * @param NUM_NODES_PER_ELEM The number of nodes in an element.
   * @param NUM_QUADRATURE_POINTS The number of quadrature points in an element.
   * @param params Variadic parameter list to hold all parameters that are forwarded to the kernel function.
   * @return Depends on the kernel.
   */
  template< typename KERNELWRAPPER, typename ... PARAMS >
  real64
  ElementKernelLaunchSelector( localIndex NUM_NODES_PER_ELEM,
                                localIndex NUM_QUADRATURE_POINTS,
                                PARAMS&&... params );

/**
 * @param WRAPPER The class/struct that contains the Launch() function that launches the kernel
 * @param OVERRIDE An optional argument to add the override specifier to the function definiton. For the base class
 *                 this should be empty. For a derived class, "override" should be entered.
 * @return A valid definition of the virtual ExplicitElementKernelLaunch() function
 *
 * This macro provides the definition for the virtual ExplicitElementKernelLaunch() function that will call the
 * ElementKernelLaunchSelector<WRAPPER>() function.
 *
 */
#define EXPLICIT_ELEMENT_KERNEL_LAUNCH( WRAPPER,OVERRIDE )\
    virtual real64\
    ExplicitElementKernelLaunch(\
        localIndex NUM_NODES_PER_ELEM,\
        localIndex NUM_QUADRATURE_POINTS,\
        set<localIndex> const & elementList,\
        arrayView2d<localIndex const> const & elemsToNodes,\
        arrayView3d< R1Tensor const> const & dNdX,\
        arrayView2d<real64 const> const & detJ,\
        arrayView1d<R1Tensor const> const & u,\
        arrayView1d<R1Tensor const> const & vel,\
        arrayView1d<R1Tensor> const & acc,\
        constitutive::ConstitutiveBase * const constitutiveRelation, \
        arrayView2d<real64> const & meanStress, \
        arrayView2d<R2SymTensor> const & devStress, \
        real64 const dt ) OVERRIDE\
    {\
      return ElementKernelLaunchSelector<WRAPPER>( NUM_NODES_PER_ELEM,\
                                                             NUM_QUADRATURE_POINTS,\
                                                             elementList,\
                                                             elemsToNodes,\
                                                             dNdX,\
                                                             detJ,\
                                                             u,\
                                                             vel,\
                                                             acc,\
                                                             constitutiveRelation,\
                                                             meanStress,\
                                                             devStress,\
                                                             dt );\
    }

  EXPLICIT_ELEMENT_KERNEL_LAUNCH(SolidMechanicsLagrangianFEM::ExplicitElementKernelWrapper,)


  /**
   * @param WRAPPER The class/struct that contains the Launch() function that launches the kernel
   * @param OVERRIDE An optional argument to add the override specifier to the function definiton. For the base class
   *                 this should be empty. For a derived class, "override" should be entered.
   * @return A valid definition of the virtual ExplicitElementKernelLaunch() function
   *
   * This macro provides the definition for the virtual ImplicitElementKernelLaunch() function that will call the
   * ElementKernelLaunchSelector<WRAPPER>() function.
   *
   */
#define IMPLICIT_ELEMENT_KERNEL_LAUNCH( WRAPPER,OVERRIDE )\
  virtual real64\
  ImplicitElementKernelLaunchSelector(\
      localIndex NUM_NODES_PER_ELEM,\
      localIndex NUM_QUADRATURE_POINTS,\
      localIndex const numElems,\
      real64 const dt,\
      arrayView3d<R1Tensor const> const & dNdX,\
      arrayView2d<real64 const > const& detJ,\
      FiniteElementBase const * const fe,\
      constitutive::ConstitutiveBase const * const constitutiveRelation,\
      arrayView1d< integer const > const & elemGhostRank,\
      arrayView2d< localIndex const > const & elemsToNodes,\
      arrayView1d< globalIndex const > const & globalDofNumber,\
      arrayView1d< R1Tensor const > const & disp,\
      arrayView1d< R1Tensor const > const & uhat,\
      arrayView1d< R1Tensor const > const & vtilde,\
      arrayView1d< R1Tensor const > const & uhattilde,\
      arrayView1d< real64 const > const & density,\
      arrayView1d< real64 const > const & fluidPressure,\
      arrayView1d< real64 const > const & deltaFluidPressure,\
      arrayView1d< real64 const > const & biotCoefficient,\
      timeIntegrationOption const tiOption,\
      real64 const stiffnessDamping,\
      real64 const massDamping,\
      real64 const newmarkBeta,\
      real64 const newmarkGamma,\
      Epetra_FECrsMatrix * const matrix,\
      Epetra_FEVector * const rhs ) OVERRIDE\
  {\
    return\
    ElementKernelLaunchSelector<WRAPPER>( NUM_NODES_PER_ELEM,\
                                           NUM_QUADRATURE_POINTS,\
                                           numElems,\
                                           dt,\
                                           dNdX,\
                                           detJ,\
                                           fe,\
                                           constitutiveRelation,\
                                           elemGhostRank,\
                                           elemsToNodes,\
                                           globalDofNumber,\
                                           disp,\
                                           uhat,\
                                           vtilde,\
                                           uhattilde,\
                                           density,\
                                           fluidPressure,\
                                           deltaFluidPressure,\
                                           biotCoefficient,\
                                           tiOption,\
                                           stiffnessDamping,\
                                           massDamping,\
                                           newmarkBeta,\
                                           newmarkGamma,\
                                           matrix,\
                                           rhs );\
    }

  IMPLICIT_ELEMENT_KERNEL_LAUNCH(SolidMechanicsLagrangianFEM::ImplicitElementKernelWrapper,)

  /**
   * @struct Structure to wrap templated functions that support the explicit time integration kernels.
   */
  struct ExplicitElementKernelWrapper
  {
    /**
     * @brief Launch of the element processing kernel for explicit time integration.
     * @tparam NUM_NODES_PER_ELEM The number of nodes/dof per element.
     * @tparam NUM_QUADRATURE_POINTS The number of quadrature points per element.
     * @param elementList The list of elements to be processed
     * @param elemsToNodes The map from the elements to the nodes that form that element.
     * @param dNdX The derivitaves of the shape functions wrt the reference configuration.
     * @param detJ The determinant of the transformation matrix (Jacobian) to the parent element.
     * @param u The nodal array of total displacements.
     * @param vel The nodal array of velocity.
     * @param acc The nodal array of force/acceleration.
     * @param constitutiveRelation An array of pointers to the constitutitve relations that are used on this region.
     * @param meanStress The mean stress at each element quadrature point
     * @param devStress The deviator stress at each element quadrature point.
     * @param dt The timestep
     * @return The achieved timestep.
     */
    template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS >
    static real64
    Launch( set<localIndex> const & elementList,
            arrayView2d<localIndex const> const & elemsToNodes,
            arrayView3d< R1Tensor const> const & dNdX,
            arrayView2d<real64 const> const & detJ,
            arrayView1d<R1Tensor const> const & u,
            arrayView1d<R1Tensor const> const & vel,
            arrayView1d<R1Tensor> const & acc,
            constitutive::ConstitutiveBase * const constitutiveRelation,
            arrayView2d<real64> const & meanStress,
            arrayView2d<R2SymTensor> const & devStress,
            real64 const dt );
  };

  /**
   * @struct Structure to wrap templated functions that support the implicit time integration kernels.
   */
  struct ImplicitElementKernelWrapper
  {
    /**
     * @brief Launch of the element processing kernel for implicit time integration.
     * @tparam NUM_NODES_PER_ELEM The number of nodes/dof per element.
     * @tparam NUM_QUADRATURE_POINTS The number of quadrature points per element.
     * @param numElems The number of elements the kernel will process.
     * @param dt The timestep.
     * @param dNdX The derivitaves of the shape functions wrt the reference configuration.
     * @param detJ The determinant of the transformation matrix (Jacobian) to the parent element.
     * @param fe A pointer to the finite element class used in this kernel.
     * @param constitutiveRelation An array of pointers to the constitutitve relations that are used on this region.
     * @param elemGhostRank An array containing the values of the owning ranks for ghost elements.
     * @param elemsToNodes The map from the elements to the nodes that form that element.
     * @param globalDofNumber The map from localIndex to the globalDOF number.
     * @param disp The array of total displacements.
     * @param uhat The array of incremental displacements (displacement for this step).
     * @param vtilde The array for the velocity predictor.
     * @param uhattilde The array for the incremental displacement predictor.
     * @param density The array containing the density
     * @param fluidPressure Array containing element fluid pressure at the beginning of the step.
     * @param deltaFluidPressure Array containing the change in element fluid pressure over this step.
     * @param biotCoefficient The biotCoefficient used to calculate effective stress.
     * @param tiOption The time integration option used for the integration.
     * @param stiffnessDamping The stiffness damping coefficient for the Newmark method assuming Rayleigh damping.
     * @param massDamping The mass damping coefficient for the Newmark method assuming Rayleigh damping.
     * @param newmarkBeta The value of \beta in the Newmark update.
     * @param newmarkGamma The value of \gamma in the Newmark update.
     * @param globaldRdU  Pointer to the sparse matrix containing the derivatives of the residual wrt displacement.
     * @param globalResidual Pointer to the parallel vector containing the global residual.
     * @return The maximum nodal force contribution from all elements.
     */
    template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS >
    static real64
    Launch( localIndex const numElems,
            real64 const dt,
            arrayView3d<R1Tensor const> const & dNdX,
            arrayView2d<real64 const > const& detJ,
            FiniteElementBase const * const fe,
            constitutive::ConstitutiveBase const * const constitutiveRelation,
            arrayView1d< integer const > const & elemGhostRank,
            arrayView2d< localIndex const > const & elemsToNodes,
            arrayView1d< globalIndex const > const & globalDofNumber,
            arrayView1d< R1Tensor const > const & disp,
            arrayView1d< R1Tensor const > const & uhat,
            arrayView1d< R1Tensor const > const & vtilde,
            arrayView1d< R1Tensor const > const & uhattilde,
            arrayView1d< real64 const > const & density,
            arrayView1d< real64 const > const & fluidPressure,
            arrayView1d< real64 const > const & deltaFluidPressure,
            arrayView1d< real64 const > const & biotCoefficient,
            timeIntegrationOption const tiOption,
            real64 const stiffnessDamping,
            real64 const massDamping,
            real64 const newmarkBeta,
            real64 const newmarkGamma,
            Epetra_FECrsMatrix * const globaldRdU,
            Epetra_FEVector * const globalResidual );
  };

  /**
   * Applies displacement boundary conditions to the system for implicit time integration
   * @param time          The time to use for any lookups associated with this BC
   * @param domain        The DomainPartition.
   * @param blockSystem   The system of equations.
   */
  void ApplyDisplacementBC_implicit( real64 const time,
                                     DomainPartition & domain,
                                     systemSolverInterface::EpetraBlockSystem & blockSystem  );


  void ApplyTractionBC( DomainPartition * const domain,
                        real64 const time,
                        systemSolverInterface::EpetraBlockSystem & blockSystem );

  void ApplyChomboPressure( DomainPartition * const domain,
                            systemSolverInterface::EpetraBlockSystem & blockSystem );

  void SetTimeIntegrationOption( string const & stringVal )
  {
    if( stringVal == "ExplicitDynamic" )
    {
      this->m_timeIntegrationOption = timeIntegrationOption::ExplicitDynamic;
    }
    else if( stringVal == "ImplicitDynamic" )
    {
      this->m_timeIntegrationOption = timeIntegrationOption::ImplicitDynamic;
    }
    else if ( stringVal == "QuasiStatic" )
    {
      this->m_timeIntegrationOption = timeIntegrationOption::QuasiStatic;
    }
    else
    {
      GEOS_ERROR("Invalid time integration option: " << stringVal);
    }
  }

  struct viewKeyStruct
  {
    static constexpr auto vTildeString = "velocityTilde";
    static constexpr auto uhatTildeString = "uhatTilde";
    static constexpr auto cflFactorString = "cflFactor";
    static constexpr auto newmarkGammaString = "newmarkGamma";
    static constexpr auto newmarkBetaString = "newmarkBeta";
    static constexpr auto massDampingString = "massDamping";
    static constexpr auto stiffnessDampingString = "stiffnessDamping";
    static constexpr auto useVelocityEstimateForQSString = "useVelocityForQS";
    static constexpr auto globalDofNumberString = "trilinosIndex";
    static constexpr auto timeIntegrationOptionStringString = "timeIntegrationOption";
    static constexpr auto timeIntegrationOptionString = "timeIntegrationOptionEnum";
    static constexpr auto maxNumResolvesString = "maxNumResolves";
    static constexpr auto strainTheoryString = "strainTheory";
    static constexpr auto solidMaterialNameString = "solidMaterialName";
    static constexpr auto solidMaterialFullIndexString = "solidMaterialFullIndex";


    dataRepository::ViewKey vTilde = { vTildeString };
    dataRepository::ViewKey uhatTilde = { uhatTildeString };
    dataRepository::ViewKey newmarkGamma = { newmarkGammaString };
    dataRepository::ViewKey newmarkBeta = { newmarkBetaString };
    dataRepository::ViewKey massDamping = { massDampingString };
    dataRepository::ViewKey stiffnessDamping = { stiffnessDampingString };
    dataRepository::ViewKey useVelocityEstimateForQS = { useVelocityEstimateForQSString };
    dataRepository::ViewKey globalDofNumber = { globalDofNumberString };
    dataRepository::ViewKey timeIntegrationOption = { timeIntegrationOptionString };
  } solidMechanicsViewKeys;

  struct groupKeyStruct
  {
    dataRepository::GroupKey systemSolverParameters = { "SystemSolverParameters" };
  } solidMechanicsGroupKeys;

protected:
  virtual void PostProcessInput() override final;

  virtual void InitializePostInitialConditions_PreSubGroups( dataRepository::ManagedGroup * const problemManager ) override final;


protected:

  real64 m_newmarkGamma;
  real64 m_newmarkBeta;
  real64 m_massDamping;
  real64 m_stiffnessDamping;
  string m_timeIntegrationOptionString;
  timeIntegrationOption m_timeIntegrationOption;
  integer m_useVelocityEstimateForQS;
  real64 m_maxForce = 0.0;
  integer m_maxNumResolves;
  integer m_strainTheory;
  string m_solidMaterialName;
  localIndex m_solidMaterialFullIndex;

  array1d< array1d < set<localIndex> > > m_elemsAttachedToSendOrReceiveNodes;
  array1d< array1d < set<localIndex> > > m_elemsNotAttachedToSendOrReceiveNodes;
  set<localIndex> m_sendOrRecieveNodes;
  set<localIndex> m_nonSendOrRecieveNodes;
  MPI_iCommData m_icomm;

  SolidMechanicsLagrangianFEM();

};
//**********************************************************************************************************************
//**********************************************************************************************************************
//**********************************************************************************************************************



template< int N >
void Integrate( const R2SymTensor& fieldvar,
                arraySlice1d<R1Tensor const> const & dNdX,
                real64 const& detJ,
                real64 const& detF,
                const R2Tensor& Finv,
                arraySlice1d<R1Tensor> & result)
{
  real64 const integrationFactor = detJ * detF;

  R2Tensor P;
  P.AijBkj( fieldvar,Finv);
  P *= integrationFactor;

  for( int a=0 ; a<N ; ++a )  // loop through all shape functions in element
  {
    result[a].minusAijBj( P, dNdX[a] );
  }
}

//**********************************************************************************************************************
//**********************************************************************************************************************
//**********************************************************************************************************************

template< typename KERNELWRAPPER, typename ... PARAMS>
real64 SolidMechanicsLagrangianFEM::
ElementKernelLaunchSelector( localIndex NUM_NODES_PER_ELEM,
                              localIndex NUM_QUADRATURE_POINTS,
                              PARAMS&& ... params)
{
  real64 rval = 0;

  if( NUM_NODES_PER_ELEM==8 && NUM_QUADRATURE_POINTS==8 )
  {
    rval = KERNELWRAPPER::template Launch<8,8>( std::forward<PARAMS>(params)...);
  }
  else if( NUM_NODES_PER_ELEM==4 && NUM_QUADRATURE_POINTS==1 )
  {
    rval = KERNELWRAPPER::template Launch<4,1>( std::forward<PARAMS>(params)...);
  }

  return rval;
}


template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS >
real64 SolidMechanicsLagrangianFEM::ExplicitElementKernelWrapper::
Launch( set<localIndex> const & elementList,
        arrayView2d<localIndex const> const & elemsToNodes,
        arrayView3d< R1Tensor const> const & dNdX,
        arrayView2d<real64 const> const & detJ,
        arrayView1d<R1Tensor const> const & u,
        arrayView1d<R1Tensor const> const & vel,
        arrayView1d<R1Tensor> const & acc,
        constitutive::ConstitutiveBase * const constitutiveRelation,
        arrayView2d<real64> const & meanStress,
        arrayView2d<R2SymTensor> const & devStress,
        real64 const dt )
{

  constitutive::ConstitutiveBase::UpdateFunctionPointer update = constitutiveRelation->GetStateUpdateFunctionPointer();
  void * data = nullptr;
  constitutiveRelation->SetParamStatePointers( data );
  forall_in_set<elemPolicy>( elementList.values(),
                             elementList.size(),
                             GEOSX_LAMBDA ( localIndex k) mutable
  {
    r1_array v_local( NUM_NODES_PER_ELEM );
    r1_array u_local( NUM_NODES_PER_ELEM );
    r1_array f_local( NUM_NODES_PER_ELEM );

    CopyGlobalToLocal<R1Tensor,NUM_NODES_PER_ELEM>( elemsToNodes[k],
                                                    u, vel,
                                                    u_local, v_local );

    //Compute Quadrature
    for( localIndex q = 0 ; q<NUM_QUADRATURE_POINTS ; ++q)
    {

      R2Tensor dUhatdX, dUdX;
      CalculateGradients<NUM_NODES_PER_ELEM>( dUhatdX, dUdX, v_local, u_local, dNdX[k][q]);
      dUhatdX *= dt;

      R2Tensor F,Ldt, Finv;

      // calculate du/dX
      F = dUhatdX;
      F *= 0.5;
      F += dUdX;
      F.PlusIdentity(1.0);
      Finv.Inverse(F);

      // chain rule: calculate dv/du = dv/dX * dX/du
      Ldt.AijBjk(dUhatdX, Finv);

      // calculate gradient (end of step)
      F = dUhatdX;
      F += dUdX;
      F.PlusIdentity(1.0);
      real64 detF = F.Det();
      Finv.Inverse(F);


      R2Tensor Rot;
      R2SymTensor Dadt;
      HughesWinget(Rot, Dadt, Ldt);

      constitutiveRelation->StateUpdatePoint( Dadt, Rot, k, q, 0);

      R2SymTensor TotalStress;
      TotalStress = devStress[k][q];
      TotalStress.PlusIdentity( meanStress[k][q] );

      Integrate<NUM_NODES_PER_ELEM>( TotalStress, dNdX[k][q], detJ[k][q], detF, Finv, f_local );
    }//quadrature loop


    AddLocalToGlobal( elemsToNodes[k], f_local, acc, NUM_NODES_PER_ELEM );
  });

  return dt;
}


template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS >
real64
SolidMechanicsLagrangianFEM::
ImplicitElementKernelWrapper::Launch( localIndex const numElems,
                             real64 const dt,
                             arrayView3d<R1Tensor const> const & dNdX,
                             arrayView2d<real64 const > const& detJ,
                             FiniteElementBase const * const fe,
                             constitutive::ConstitutiveBase const * const constitutiveRelation,
                             arrayView1d< integer const > const & elemGhostRank,
                             arrayView2d< localIndex const > const & elemsToNodes,
                             arrayView1d< globalIndex const > const & globalDofNumber,
                             arrayView1d< R1Tensor const > const & disp,
                             arrayView1d< R1Tensor const > const & uhat,
                             arrayView1d< R1Tensor const > const & vtilde,
                             arrayView1d< R1Tensor const > const & uhattilde,
                             arrayView1d< real64 const > const & density,
                             arrayView1d< real64 const > const & fluidPressure,
                             arrayView1d< real64 const > const & deltaFluidPressure,
                             arrayView1d< real64 const > const & biotCoefficient,
                             timeIntegrationOption const tiOption,
                             real64 const stiffnessDamping,
                             real64 const massDamping,
                             real64 const newmarkBeta,
                             real64 const newmarkGamma,
                             Epetra_FECrsMatrix * const matrix,
                             Epetra_FEVector * const rhs )
{
  GEOS_ERROR("SolidMechanicsLagrangianFEM::ImplicitElementKernelWrapper::Launch() not implemented");
return 0;
}

} /* namespace geosx */

#endif /* SOLID_MECHANICS_LAGRANGIAN_FEM_HPP_ */
