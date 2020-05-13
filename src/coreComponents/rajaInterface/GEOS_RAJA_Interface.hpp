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

#ifndef GEOSX_RAJAINTERFACE_RAJAINTERFACE_HPP
#define GEOSX_RAJAINTERFACE_RAJAINTERFACE_HPP

// Source includes
#include "common/DataTypes.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

namespace geosx
{

using serialPolicy = RAJA::loop_exec;
using serialReduce = RAJA::seq_reduce;
using serialAtomic = RAJA::seq_atomic;

#if defined(GEOSX_USE_OPENMP)

using parallelHostPolicy = RAJA::omp_parallel_for_exec;
using parallelHostReduce = RAJA::omp_reduce;
using parallelHostAtomic = RAJA::builtin_atomic;

#else

using parallelHostPolicy = serialPolicy;
using parallelHostReduce = serialReduce;
using parallelHostAtomic = serialAtomic;

#endif

#if defined(GEOSX_USE_CUDA)

template< int BLOCK_SIZE = 256 >
using parallelDevicePolicy = RAJA::cuda_exec< BLOCK_SIZE >;
template< int BLOCK_SIZE = 256 >
using parallelDeviceAsync  = RAJA::cuda_exec< BLOCK_SIZE, true >;
using parallelDeviceReduce = RAJA::cuda_reduce;
using parallelDeviceAtomic = RAJA::cuda_atomic;

using parallelDeviceStream = RAJA::resources::Resource;
using parallelDeviceEvent = RAJA::resources::Event;

inline parallelDeviceStream getDeviceStream( )
{
  return RAJA::resources::Cuda();
}

#else

template< int BLOCK_SIZE = 0 >
using parallelDevicePolicy = parallelHostPolicy;
template< int BLOCK_SIZE = 0 >
using parallelDeviceAsync  = parallelHostPolicy;
using parallelDeviceReduce = parallelHostReduce;
using parallelDeviceAtomic = parallelHostAtomic;

struct parallelDeviceEvent
{
  inline bool check( ) { return true; }
  inline void wait( ) { return; }
};
struct parallelDeviceStream 
{ 
  inline void wait_for( parallelDeviceEvent * GEOSX_UNUSED_PARAM( event ) ) { return; }
};


#endif

template< typename POLICY, typename LAMBDA >
RAJA_INLINE void forAll( const localIndex end, LAMBDA && body )
{
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< localIndex >( 0, end ), std::forward< LAMBDA >( body ) );
}


template< typename POLICY, typename RESOURCE, typename LAMBDA >
RAJA_INLINE parallelDeviceEvent forAll( RESOURCE && res, const localIndex end, LAMBDA && body )
{
  return RAJA::forall< POLICY >( std::forward< RESOURCE >( res ),  
				 RAJA::TypedRangeSegment< localIndex >( 0, end ), 
				 std::forward< LAMBDA >( body ) );
}

} // namespace geosx

#endif // GEOSX_RAJAINTERFACE_RAJAINTERFACE_HPP
