//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Aditya Ghantasala

#if !defined(DISTANCE_CALCULATION_UTILITY)
#define DISTANCE_CALCULATION_UTILITY

// System includes

// External includes
#include "omp.h"

// Project includes
#include "includes/define.h"
#include "processes/variational_distance_calculation_process.h"
#include "utilities/parallel_levelset_distance_calculator.h"
#include "processes/calculate_signed_distance_to_3d_condition_skin_process.h"
#include "mpi/utilities/gather_modelpart_utility.h"
#include "processes/fast_transfer_between_model_parts_process.h"

#include "mpi/utilities/parallel_fill_communicator.h"
#include "includes/data_communicator.h"
#include "mpi/includes/mpi_communicator.h"

namespace Kratos
{

///@name Kratos Classes
///@{

/// Utility for calculating the Distance on a given modelpart
template <int TDim, class TSparseSpaceType, class TLocalSpaceType>
class KRATOS_API(CHIMERA_APPLICATION) DistanceCalculationUtility
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of DistanceCalculationUtility
    KRATOS_CLASS_POINTER_DEFINITION(DistanceCalculationUtility);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    DistanceCalculationUtility() = delete;

    /// Destructor.
    /// Deleted copy constructor
    DistanceCalculationUtility(const DistanceCalculationUtility &rOther) = delete;

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Calculates distance on the whole of rVolumeModelPart from rSkinModelPart
     * @param rVolumeModelPart The background modelpart where distances are calculated.
     * @param rSkinModelPart The skin modelpart from where the distances are calculated
     */
    static inline void CalculateDistance(ModelPart &rVolumeModelPart, ModelPart &rSkinModelPart)
    {
        typedef CalculateDistanceToSkinProcess<TDim> CalculateDistanceToSkinProcessType;
        const int n_bg_nodes = static_cast<int>(rVolumeModelPart.NumberOfNodes());
#pragma omp parallel for
        for (int i_node = 0; i_node < n_bg_nodes; ++i_node)
        {
            auto it_node = rVolumeModelPart.NodesBegin() + i_node;
            it_node->FastGetSolutionStepValue(DISTANCE, 0) = 0.0;
            it_node->FastGetSolutionStepValue(DISTANCE, 1) = 0.0;
            it_node->SetValue(DISTANCE, 0.0);
        }

        const DataCommunicator &r_comm =
            rVolumeModelPart.GetCommunicator().GetDataCommunicator();
        rSkinModelPart.SetCommunicator(rVolumeModelPart.pGetCommunicator());
        Model &current_model = rVolumeModelPart.GetModel();
        ModelPart &r_gathered_skin_mp = r_comm.IsDistributed() ? current_model.CreateModelPart("GatheredSkin") : rSkinModelPart;

        // If it is distributed, gather on the rank 0 to do the distance calculation
        if (r_comm.IsDistributed())
        {
            DistanceCalculationUtility::GatherModelPartOnAllRanks(rSkinModelPart, r_gathered_skin_mp);
        }
        r_comm.Barrier();

        // This distance computation is always local to each rank.
        // Now the bg modelpart has the distances from the skin.
        const int n_skin_nodes = static_cast<int>(r_gathered_skin_mp.NumberOfNodes());
        if(n_skin_nodes != 0 && n_bg_nodes != 0)
            // If on of them do not have any nodes, then there is no need to continue
            CalculateDistanceToSkinProcessType(rVolumeModelPart, r_gathered_skin_mp).Execute();

        unsigned int max_level = 100;
        double max_distance = 200;
        auto p_distance_smoother = Kratos::make_shared<ParallelDistanceCalculator<TDim>>();
        p_distance_smoother->CalculateDistances(rVolumeModelPart, DISTANCE, NODAL_AREA, max_level, max_distance);

        current_model.DeleteModelPart("GatheredSkin");
    }

    /**
     * @brief Gathers the given modelpart on all the ranks
     * @param rModelPartToGather The modelpart which is to be gathered on all nodes
     * @param rGatheredModelPart The full gathered modelpart from all the ranks.
     */
    static inline void GatherModelPartOnAllRanks(ModelPart &rModelPartToGather, ModelPart &rGatheredModelPart)
    {
        typedef ModelPart::NodesContainerType NodesContainerType;
        typedef ModelPart::ElementsContainerType ElementsContainerType;
        typedef ModelPart::ConditionsContainerType ConditionsContainerType;

        const DataCommunicator &r_comm =
            rModelPartToGather.GetCommunicator().GetDataCommunicator();
        const int mpi_size = r_comm.Size();
        const int mpi_rank = r_comm.Rank();

        rGatheredModelPart.GetNodalSolutionStepVariablesList() =
            rModelPartToGather.GetNodalSolutionStepVariablesList();
        if (r_comm.IsDistributed())
        {
        VariablesList* pVariablesList =
            &rGatheredModelPart.GetNodalSolutionStepVariablesList();
        rGatheredModelPart.SetCommunicator(
            Communicator::Pointer(new MPICommunicator(pVariablesList, r_comm)));
        }
        rGatheredModelPart.SetBufferSize(rModelPartToGather.GetBufferSize());

        // send everything to node with id "gather_rank"
        // transfer nodes
        std::vector<NodesContainerType> SendNodes(mpi_size);
        std::vector<NodesContainerType> RecvNodes(mpi_size);

        for (int dest_rank = 0; dest_rank < mpi_size; ++dest_rank)
        {
            SendNodes[dest_rank].reserve(rModelPartToGather.Nodes().size());
            if (r_comm.IsDistributed())
            {
                for (NodesContainerType::iterator it = rModelPartToGather.NodesBegin();
                     it != rModelPartToGather.NodesEnd(); ++it)
                {
                    // only send the nodes owned by this partition
                    if (it->FastGetSolutionStepValue(PARTITION_INDEX) == mpi_rank)
                        SendNodes[dest_rank].push_back(*it.base());
                }
            }
            else
            {
                for (NodesContainerType::iterator it = rModelPartToGather.NodesBegin();
                     it != rModelPartToGather.NodesEnd(); ++it)
                {
                    SendNodes[dest_rank].push_back(*it.base());
                }
            }
        }

        rModelPartToGather.GetCommunicator().TransferObjects(SendNodes, RecvNodes);
        for (unsigned int i = 0; i < RecvNodes.size(); i++)
        {
            for (NodesContainerType::iterator it = RecvNodes[i].begin();
                 it != RecvNodes[i].end(); ++it)
                if (rGatheredModelPart.Nodes().find(it->Id()) ==
                    rGatheredModelPart.Nodes().end())
                    rGatheredModelPart.Nodes().push_back(*it.base());
        }
        int temp = rGatheredModelPart.Nodes().size();
        KRATOS_ERROR_IF(temp != int(rGatheredModelPart.Nodes().size()))
            << "the rGatheredModelPart has repeated nodes";
        SendNodes.clear();
        RecvNodes.clear();
        for (NodesContainerType::iterator it = rModelPartToGather.GetMesh(0).NodesBegin();
             it != rModelPartToGather.GetMesh(0).NodesEnd(); ++it)
        {
            rGatheredModelPart.Nodes().push_back(*it.base());
        }
        rGatheredModelPart.Nodes().Unique();

        // transfer elements
        std::vector<ElementsContainerType> SendElements(mpi_size);
        std::vector<ElementsContainerType> RecvElements(mpi_size);
        for (int dest_rank = 0; dest_rank < mpi_size; ++dest_rank)
        {
            SendElements[dest_rank].reserve(rModelPartToGather.Elements().size());
            for (ElementsContainerType::iterator it = rModelPartToGather.ElementsBegin();
                 it != rModelPartToGather.ElementsEnd(); ++it)
            {
                SendElements[dest_rank].push_back(*it.base());
            }
        }
        rModelPartToGather.GetCommunicator().TransferObjects(SendElements, RecvElements);

        for (unsigned int i = 0; i < RecvElements.size(); i++)
        {
            for (ElementsContainerType::iterator it = RecvElements[i].begin();
                 it != RecvElements[i].end(); ++it)
            {
                // replace the nodes copied with the element by nodes
                // in the model part
                Element::GeometryType &rGeom = it->GetGeometry();
                unsigned int NumNodes = rGeom.PointsNumber();
                for (unsigned int iNode = 0; iNode < NumNodes; iNode++)
                {
                    NodesContainerType::iterator itNode =
                        rGatheredModelPart.Nodes().find(rGeom(iNode)->Id());
                    if (itNode != rGatheredModelPart.Nodes().end())
                        rGeom(iNode) = *itNode.base();
                }
                rGatheredModelPart.Elements().push_back(*it.base());
            }
        }
        SendElements.clear();
        RecvElements.clear();
        for (ElementsContainerType::iterator it =
                 rModelPartToGather.GetMesh(0).ElementsBegin();
             it != rModelPartToGather.GetMesh(0).ElementsEnd(); ++it)
        {
            rGatheredModelPart.Elements().push_back(*it.base());
        }
        // rGatheredModelPart.Elements().Unique();

        // transfer conditions
        std::vector<ConditionsContainerType> SendConditions(mpi_size);
        std::vector<ConditionsContainerType> RecvConditions(mpi_size);
        for (int dest_rank = 0; dest_rank < mpi_size; ++dest_rank)
        {
            SendConditions[dest_rank].reserve(rModelPartToGather.Conditions().size());
            for (ConditionsContainerType::iterator it = rModelPartToGather.ConditionsBegin();
                 it != rModelPartToGather.ConditionsEnd(); ++it)
            {
                SendConditions[dest_rank].push_back(*it.base());
            }
        }
        rModelPartToGather.GetCommunicator().TransferObjects(SendConditions, RecvConditions);
        for (unsigned int i = 0; i < RecvConditions.size(); i++)
        {
            for (ConditionsContainerType::iterator it = RecvConditions[i].begin();
                 it != RecvConditions[i].end(); ++it)
            {
                // replace the nodes copied with the condition by nodes
                // in the model part
                Condition::GeometryType &rGeom = it->GetGeometry();
                unsigned int NumNodes = rGeom.PointsNumber();
                for (unsigned int iNode = 0; iNode < NumNodes; iNode++)
                {
                    NodesContainerType::iterator itNode =
                        rGatheredModelPart.Nodes().find(rGeom(iNode)->Id());
                    if (itNode != rGatheredModelPart.Nodes().end())
                        rGeom(iNode) = *itNode.base();
                }
                rGatheredModelPart.Conditions().push_back(*it.base());
            }
        }
        SendConditions.clear();
        RecvConditions.clear();

        for (ConditionsContainerType::iterator it =
                 rModelPartToGather.GetMesh(0).ConditionsBegin();
             it != rModelPartToGather.GetMesh(0).ConditionsEnd(); ++it)
        {
            rGatheredModelPart.Conditions().push_back(*it.base());
        }
        // rGatheredModelPart.Conditions().Unique();

        if (r_comm.IsDistributed())
        {
            ParallelFillCommunicator(rGatheredModelPart).Execute();
        }
    }

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    ///@}
    ///@name Friends
    ///@{

    ///@}

private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    ///@}

}; // Class DistanceCalculationUtility

///@}

///@name Type Definitions
///@{

///@}
///@name Input and output
///@{

///@}

} // namespace Kratos.

#endif // DISTANCE_CALCULATION_UTILITY  defined
