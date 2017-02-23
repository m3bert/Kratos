//
//   Project Name:        Kratos
//   Last modified by:    $Author: rrossi $
//   Date:                $Date: 2007-08-21 14:11:10 $
//   Revision:            $Revision: 1.3 $
//
//


// System includes

// External includes
#include <boost/python.hpp>

// Project includes
#include "spaces/ublas_space.h"
#include "includes/define.h"
#include "processes/process.h"
#include "custom_python/add_custom_utilities_to_python.h"
#include "custom_utilities/FSI_utils.h"
#include "custom_utilities/aitken_utils.h"
#include "custom_utilities/partitioned_fsi_utilities.hpp"

namespace Kratos
{

namespace Python
{

void  AddCustomUtilitiesToPython()
{
    using namespace boost::python;
    typedef UblasSpace<double, Matrix, Vector > TSpace;

    class_<FSIUtils>("FSIUtils", init<>())
//		.def("FSIUtils",&FSIUtils::GenerateCouplingElements)
    .def("CheckPressureConvergence",&FSIUtils::CheckPressureConvergence)
    .def("StructuralPressurePrediction",&FSIUtils::StructuralPressurePrediction)
    ;

    class_<AitkenUtils>("AitkenUtils", init<>())
    .def("ComputeAitkenFactor",&AitkenUtils::ComputeAitkenFactor)
    .def("ComputeRelaxedDisplacement",&AitkenUtils::ComputeRelaxedDisplacement)
    ;

    class_<PartitionedFSIUtilities <TSpace,2>, boost::noncopyable >("PartitionedFSIUtilities2D", init< ModelPart&, ModelPart& >())
    // .def("GetFluidInterfaceProblemSize",&PartitionedFSIUtilities<TSpace,2>::GetFluidInterfaceProblemSize)
    // .def("GetStructureInterfaceProblemSize",&PartitionedFSIUtilities<TSpace,2>::GetStructureInterfaceProblemSize)
    .def("GetFluidInterfaceResidualSize",&PartitionedFSIUtilities<TSpace,2>::GetFluidInterfaceResidualSize)
    .def("GetStructureInterfaceResidualSize",&PartitionedFSIUtilities<TSpace,2>::GetStructureInterfaceResidualSize)
    .def("SetFluidInterfaceVelocity",&PartitionedFSIUtilities<TSpace,2>::SetFluidInterfaceVelocity)
    .def("SetFluidInterfaceNodalFlux",&PartitionedFSIUtilities<TSpace,2>::SetFluidInterfaceNodalFlux)
    .def("ComputeFluidInterfaceVelocityResidual",&PartitionedFSIUtilities<TSpace,2>::ComputeFluidInterfaceVelocityResidual)
    .def("ComputeFluidInterfaceMeshVelocityResidualNorm",&PartitionedFSIUtilities<TSpace,2>::ComputeFluidInterfaceMeshVelocityResidualNorm)
    ;

    class_<PartitionedFSIUtilities <TSpace,3>, boost::noncopyable >("PartitionedFSIUtilities3D", init< ModelPart&, ModelPart& >())
    // .def("GetFluidInterfaceProblemSize",&PartitionedFSIUtilities<TSpace,3>::GetFluidInterfaceProblemSize)
    // .def("GetStructureInterfaceProblemSize",&PartitionedFSIUtilities<TSpace,3>::GetStructureInterfaceProblemSize)
    .def("GetFluidInterfaceResidualSize",&PartitionedFSIUtilities<TSpace,3>::GetFluidInterfaceResidualSize)
    .def("GetStructureInterfaceResidualSize",&PartitionedFSIUtilities<TSpace,3>::GetStructureInterfaceResidualSize)
    .def("SetFluidInterfaceVelocity",&PartitionedFSIUtilities<TSpace,3>::SetFluidInterfaceVelocity)
    .def("SetFluidInterfaceNodalFlux",&PartitionedFSIUtilities<TSpace,3>::SetFluidInterfaceNodalFlux)
    .def("ComputeFluidInterfaceVelocityResidual",&PartitionedFSIUtilities<TSpace,3>::ComputeFluidInterfaceVelocityResidual)
    .def("ComputeFluidInterfaceMeshVelocityResidualNorm",&PartitionedFSIUtilities<TSpace,3>::ComputeFluidInterfaceMeshVelocityResidualNorm)
    ;

}

}  // namespace Python.

} // Namespace Kratos
