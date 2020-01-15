#include "hole_cutting_utility.h"

#ifdef KRATOS_USING_MPI
#include "mpi/utilities/parallel_fill_communicator.h"
#endif
#include "includes/data_communicator.h"
#include "input_output/vtk_output.h"

namespace Kratos
{

template <int TDim>
void ChimeraHoleCuttingUtility::CreateHoleAfterDistance(
    ModelPart &rModelPart, ModelPart &rHoleModelPart,
    ModelPart &rHoleBoundaryModelPart, const double Distance)
{
    KRATOS_TRY;
    ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<TDim>(rModelPart,
                                                               rHoleModelPart,
                                                               ChimeraHoleCuttingUtility::Domain::MAIN_BACKGROUND,
                                                               Distance,
                                                               ChimeraHoleCuttingUtility::SideToExtract::INSIDE);

    ChimeraHoleCuttingUtility::ExtractBoundaryMesh<TDim>(rHoleModelPart, rHoleBoundaryModelPart);
    KRATOS_CATCH("");
}

template <int TDim>
void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements(
    ModelPart &rModelPart, ModelPart &rRemovedModelPart,
    const ChimeraHoleCuttingUtility::Domain DomainType, const double OverLapDistance,
    const ChimeraHoleCuttingUtility::SideToExtract Side)
{
    KRATOS_TRY;
    std::vector<IndexType> vector_of_node_ids;
    std::vector<IndexType> vector_of_elem_ids;
    int count = 0;

    const auto &r_local_mesh = rModelPart.GetCommunicator().LocalMesh();
    for (auto &i_element : r_local_mesh.Elements())
    {
        double nodal_distance = 0.0;
        IndexType numPointsOutside = 0;
        IndexType j = 0;
        Geometry<Node<3>> &geom = i_element.GetGeometry();

        for (j = 0; j < geom.size(); j++)
        {
            nodal_distance =
                i_element.GetGeometry()[j].FastGetSolutionStepValue(CHIMERA_DISTANCE);

            nodal_distance = nodal_distance * DomainType;
            if (nodal_distance < -1 * OverLapDistance)
            {
                numPointsOutside++;
            }
        }

        /* Any node goes out of the domain means the element need to be INACTIVE ,
       otherwise the modified patch boundary wont find any nodes on background
     */
        if (numPointsOutside > 0)
        {
            i_element.Set(ACTIVE, false);
            IndexType num_nodes_per_elem = i_element.GetGeometry().PointsNumber();
            if (Side == ChimeraHoleCuttingUtility::SideToExtract::INSIDE)
                vector_of_elem_ids.push_back(i_element.Id());
            for (j = 0; j < num_nodes_per_elem; j++)
            {
                i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_X, 0) =
                    0.0;
                i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_Y, 0) =
                    0.0;
                if (TDim > 2)
                    i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_Z, 0) =
                        0.0;
                i_element.GetGeometry()[j].FastGetSolutionStepValue(PRESSURE, 0) = 0.0;
                i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_X, 1) =
                    0.0;
                i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_Y, 1) =
                    0.0;
                if (TDim > 2)
                    i_element.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY_Z, 1) =
                        0.0;
                i_element.GetGeometry()[j].FastGetSolutionStepValue(PRESSURE, 1) = 0.0;
                if (Side == ChimeraHoleCuttingUtility::SideToExtract::INSIDE)
                    vector_of_node_ids.push_back(i_element.GetGeometry()[j].Id());
            }
        }
        else
        {
            if (Side == ChimeraHoleCuttingUtility::SideToExtract::OUTSIDE)
            {
                count++;
                IndexType num_nodes_per_elem =
                    i_element.GetGeometry().PointsNumber(); // Size()
                vector_of_elem_ids.push_back(i_element.Id());
                for (j = 0; j < num_nodes_per_elem; j++)
                    vector_of_node_ids.push_back(i_element.GetGeometry()[j].Id());
            }
        }
    }

    rRemovedModelPart.AddElements(vector_of_elem_ids);
    // sorting and making unique list of node ids
    std::set<IndexType> s(vector_of_node_ids.begin(), vector_of_node_ids.end());
    vector_of_node_ids.assign(s.begin(), s.end());
    rRemovedModelPart.AddNodes(vector_of_node_ids);

#ifdef KRATOS_USING_MPI
        ParallelFillCommunicator(rRemovedModelPart).Execute();
#endif

    KRATOS_CATCH("");
}

template <int TDim>
void ChimeraHoleCuttingUtility::ExtractBoundaryMesh(
    ModelPart &rVolumeModelPart, ModelPart &rExtractedBoundaryModelPart,
    const ChimeraHoleCuttingUtility::SideToExtract GetInternal)
{
    KRATOS_TRY;

    const auto &r_comm = rVolumeModelPart.GetCommunicator();
    const bool &is_distributed = r_comm.IsDistributed();

    // IndexType n_nodes = rVolumeModelPart.ElementsBegin()->GetGeometry().size();
    // KRATOS_ERROR_IF(!(n_nodes != 3 || n_nodes != 4))
    //     << "Hole cutting process is only supported for tetrahedral and "
    //        "triangular elements"
    //     << std::endl;

    // Create map to ask for number of faces for the given set of node ids
    // representing on face in the model part
    hashmap n_faces_map;
    hashmap faces_elem_id_map;
    const int num_elements =
        static_cast<int>(rVolumeModelPart.NumberOfElements());
    const auto elements_begin = rVolumeModelPart.ElementsBegin();
    // Fill map that counts number of faces for given set of nodes
#pragma omp parallel for
    for (int i_e = 0; i_e < num_elements; ++i_e)
    {
        auto i_element = elements_begin + i_e;
        Element::GeometryType::GeometriesArrayType faces;
        faces = i_element->GetGeometry().GenerateBoundariesEntities();

        for (IndexType i_face = 0; i_face < faces.size(); i_face++)
        {
            // Create vector that stores all node is of current i_face
            vector<IndexType> ids(faces[i_face].size());

            // Store node ids
            for (IndexType i = 0; i < faces[i_face].size(); i++)
                ids[i] = faces[i_face][i].Id();

            //*** THE ARRAY OF IDS MUST BE ORDERED!!! ***
            std::sort(ids.begin(), ids.end());

// Fill the map
#pragma omp critical
            n_faces_map[ids] += 1;

            faces_elem_id_map[ids] = i_element->Id();
        }
    }
    // Create a map to get nodes of skin face in original order for given set of
    // node ids representing that face The given set of node ids may have a
    // different node order
    hashmap_vec ordered_skin_face_nodes_map;

    // Fill map that gives original node order for set of nodes
#pragma omp parallel for
    for (int i_e = 0; i_e < num_elements; ++i_e)
    {
        auto i_element = elements_begin + i_e;
        Element::GeometryType::GeometriesArrayType faces;
        faces = i_element->GetGeometry().GenerateBoundariesEntities();

        for (IndexType i_face = 0; i_face < faces.size(); i_face++)
        {
            // Create vector that stores all node is of current i_face
            vector<IndexType> ids(faces[i_face].size());
            vector<IndexType> unsorted_ids(faces[i_face].size());

            // Store node ids
            for (IndexType i = 0; i < faces[i_face].size(); i++)
            {
                ids[i] = faces[i_face][i].Id();
                unsorted_ids[i] = faces[i_face][i].Id();
            }

            //*** THE ARRAY OF IDS MUST BE ORDERED!!! ***
            std::sort(ids.begin(), ids.end());
#pragma omp critical
            {
                if (n_faces_map[ids] == 1)
                    ordered_skin_face_nodes_map[ids] = unsorted_ids;
            }
        }
    }
    // First assign to skin model part all nodes from original model_part,
    // unnecessary nodes will be removed later
    IndexType id_condition = 1;

    // Add skin faces as triangles to skin-model-part (loop over all node sets)
    for (typename hashmap::const_iterator it = n_faces_map.begin();
         it != n_faces_map.end(); it++)
    {
        // If given node set represents face that is not overlapping with a face of
        // another element, add it as skin element
        if (it->second == 1)
        {
            // If skin edge is a triangle store triangle in with its original
            // orientation in new skin model part
            if (it->first.size() == 2)
            {
                // Getting original order is important to properly reproduce skin edge
                // including its normal orientation
                vector<IndexType> original_nodes_order =
                    ordered_skin_face_nodes_map[it->first];

                Node<3>::Pointer pnode1 =
                    rVolumeModelPart.Nodes()(original_nodes_order[0]);
                Node<3>::Pointer pnode2 =
                    rVolumeModelPart.Nodes()(original_nodes_order[1]);

                Properties::Pointer properties =
                    rExtractedBoundaryModelPart.rProperties()(0);
                Condition const &rReferenceLineCondition =
                    KratosComponents<Condition>::Get(
                        "LineCondition2D2N"); // Condition2D

                // Skin edges are added as conditions
                Line2D2<Node<3>> line1(pnode1, pnode2);
                Condition::Pointer p_condition1 =
                    rReferenceLineCondition.Create(id_condition++, line1, properties);
                rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);
            }
            // If skin face is a triangle store triangle in with its original
            // orientation in new skin model part
            if (it->first.size() == 3)
            {
                // Getting original order is important to properly reproduce skin face
                // including its normal orientation
                vector<IndexType> original_nodes_order =
                    ordered_skin_face_nodes_map[it->first];

                Node<3>::Pointer pnode1 =
                    rVolumeModelPart.Nodes()(original_nodes_order[0]);
                Node<3>::Pointer pnode2 =
                    rVolumeModelPart.Nodes()(original_nodes_order[1]);
                Node<3>::Pointer pnode3 =
                    rVolumeModelPart.Nodes()(original_nodes_order[2]);

                Properties::Pointer properties =
                    rExtractedBoundaryModelPart.rProperties()(0);
                Condition const &rReferenceTriangleCondition =
                    KratosComponents<Condition>::Get(
                        "SurfaceCondition3D3N"); // Condition3D

                // Skin faces are added as conditions
                Triangle3D3<Node<3>> triangle1(pnode1, pnode2, pnode3);
                Condition::Pointer p_condition1 = rReferenceTriangleCondition.Create(
                    id_condition++, triangle1, properties);
                rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);
            }
            // If skin face is a quadrilateral then divide in two triangles and store
            // them with their original orientation in new skin model part
            if (it->first.size() == 4)
            {
                // Getting original order is important to properly reproduce skin
                // including its normal orientation
                vector<IndexType> original_nodes_order =
                    ordered_skin_face_nodes_map[it->first];

                Node<3>::Pointer pnode1 =
                    rVolumeModelPart.Nodes()(original_nodes_order[0]);
                Node<3>::Pointer pnode2 =
                    rVolumeModelPart.Nodes()(original_nodes_order[1]);
                Node<3>::Pointer pnode3 =
                    rVolumeModelPart.Nodes()(original_nodes_order[2]);
                Node<3>::Pointer pnode4 =
                    rVolumeModelPart.Nodes()(original_nodes_order[3]);

                Properties::Pointer properties =
                    rExtractedBoundaryModelPart.rProperties()(0);
                Condition const &rReferenceTriangleCondition =
                    KratosComponents<Condition>::Get(
                        "SurfaceCondition3D3N"); // Condition3D

                // Add triangle one as condition
                Triangle3D3<Node<3>> triangle1(pnode1, pnode2, pnode3);
                Condition::Pointer p_condition1 = rReferenceTriangleCondition.Create(
                    id_condition++, triangle1, properties);
                rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);

                // Add triangle two as condition
                Triangle3D3<Node<3>> triangle2(pnode1, pnode3, pnode4);
                Condition::Pointer p_condition2 = rReferenceTriangleCondition.Create(
                    id_condition++, triangle2, properties);
                rExtractedBoundaryModelPart.Conditions().push_back(p_condition2);
            }
        }
    }

    std::vector<IndexType> vector_of_node_ids;
    vector_of_node_ids.reserve(rVolumeModelPart.NumberOfNodes()/3);
    for(const auto& cond : rExtractedBoundaryModelPart.Conditions()){
        const auto& geom = cond.GetGeometry();
        for(const auto& node : geom){
            vector_of_node_ids.push_back(node.Id());
        }
    }

    // sorting and making unique list of node ids
    std::set<IndexType> sort_set(vector_of_node_ids.begin(),
                                 vector_of_node_ids.end());
    vector_of_node_ids.assign(sort_set.begin(), sort_set.end());
    rExtractedBoundaryModelPart.AddNodes(vector_of_node_ids);

    // for multipatch
    const int num_nodes =
        static_cast<int>(rExtractedBoundaryModelPart.NumberOfNodes());
    const auto nodes_begin = rExtractedBoundaryModelPart.NodesBegin();

#pragma omp parallel for
    for (int i_n = 0; i_n < num_nodes; ++i_n)
    {
        auto i_node = nodes_begin + i_n;
        i_node->Set(TO_ERASE, false);
    }

    const int num_conditions =
        static_cast<int>(rExtractedBoundaryModelPart.NumberOfConditions());
    const auto conditions_begin = rExtractedBoundaryModelPart.ConditionsBegin();

#pragma omp parallel for
    for (int i_c = 0; i_c < num_conditions; ++i_c)
    {
        auto i_condition = conditions_begin + i_c;
        i_condition->Set(TO_ERASE, false);
    }

    for (auto &i_condition : rExtractedBoundaryModelPart.Conditions())
    {
        auto &geo = i_condition.GetGeometry();
        bool is_internal = true;
        for (const auto &node : geo)
            is_internal = is_internal && node.Is(CHIMERA_INTERNAL_BOUNDARY);
        if (is_internal)
        {
            if (GetInternal == ChimeraHoleCuttingUtility::SideToExtract::OUTSIDE)
            {
                i_condition.Set(TO_ERASE);
                for (auto &node : geo)
                    node.Set(TO_ERASE);
            }
        }
        else
        {
            if (GetInternal == ChimeraHoleCuttingUtility::SideToExtract::INSIDE)
            {
                i_condition.Set(TO_ERASE);
                for (auto &node : geo)
                    node.Set(TO_ERASE);
            }
        }
    }

    rExtractedBoundaryModelPart.RemoveConditions(TO_ERASE);
    rExtractedBoundaryModelPart.RemoveNodes(TO_ERASE);

    if(is_distributed)
        CheckInterfaceConditionsInMPI(rVolumeModelPart, rExtractedBoundaryModelPart, faces_elem_id_map);

    Parameters vtk_parameters(R"(
            {
                "output_control_type"                : "step",
                "output_frequency"                   : 1,
                "file_format"                        : "binary",
                "output_precision"                   : 3,
                "output_sub_model_parts"             : false,
                "folder_name"                        : "test_vtk_output",
                "save_output_files_in_folder"        : false,
                "nodal_solution_step_data_variables" : ["PARTITION_INDEX"],
                "nodal_data_value_variables"         : [],
                "element_flags"                      : [],
                "nodal_flags"                        : [],
                "element_data_value_variables"       : [],
                "condition_data_value_variables"     : [],
                "write_ids"                          :true
            }
            )");

    VtkOutput vtk_output(rExtractedBoundaryModelPart, vtk_parameters);
    vtk_output.PrintOutput();

    KRATOS_CATCH("");
}

void ChimeraHoleCuttingUtility::CheckInterfaceConditionsInMPI(ModelPart& rVolumeModelPart, ModelPart& rExtractedBoundaryModelPart, hashmap& rFaceElemMap)
{

    const auto &r_comm = rVolumeModelPart.GetCommunicator();
    //const bool &is_distributed = r_comm.IsDistributed();
    const int my_rank = r_comm.MyPID();
    const auto &r_interface_mesh = r_comm.InterfaceMesh();
    // const auto &r_local_mesh = r_comm.LocalMesh();
    // const auto &r_ghost_mesh = r_comm.GhostMesh();
    std::vector<int> cond_ids_to_remove;
    cond_ids_to_remove.reserve(100);

    rExtractedBoundaryModelPart.Nodes().clear();

    for(auto& i_cond : rExtractedBoundaryModelPart.Conditions()){
        bool cond_on_interface = true;
        bool cond_local = true;
        int cond_local_nodes = 0;
        for(auto& i_node : i_cond.GetGeometry()){
            cond_on_interface = cond_on_interface && r_interface_mesh.HasNode(i_node.Id());
            bool is_my_node = (i_node.GetSolutionStepValue(PARTITION_INDEX) == my_rank);
            cond_local = cond_local && is_my_node;
            if(is_my_node) ++cond_local_nodes;
        }

        if(cond_on_interface && !cond_local){
        //if(cond_on_interface && cond_local_nodes < (int)(i_cond.GetGeometry().size()/2) ){
            // Create vector that stores all node is of current i_face
            vector<IndexType> ids(i_cond.GetGeometry().size());
            // Store node ids
            int i=0;
            for(auto& i_node : i_cond.GetGeometry())
                ids[i++] = i_node.Id();

            //*** THE ARRAY OF IDS MUST BE ORDERED!!! ***
            std::sort(ids.begin(), ids.end());

            auto& r_face_elem = rVolumeModelPart.Elements()( rFaceElemMap[ids] );
            bool all_nodes_on_interface = true;
            for(auto& node : r_face_elem->GetGeometry())
                all_nodes_on_interface = all_nodes_on_interface && r_interface_mesh.HasNode(node.Id());

            if(!all_nodes_on_interface)
                cond_ids_to_remove.push_back(i_cond.Id());
        }

    }

    for(const int& cond_id : cond_ids_to_remove){
        rExtractedBoundaryModelPart.RemoveCondition(cond_id);
    }

    std::vector<IndexType> vector_of_node_ids;
    vector_of_node_ids.reserve(rVolumeModelPart.NumberOfNodes()/3);
    for(const auto& cond : rExtractedBoundaryModelPart.Conditions()){
        const auto& geom = cond.GetGeometry();
        for(const auto& node : geom){
            vector_of_node_ids.push_back(node.Id());
        }
    }

    // sorting and making unique list of node ids
    std::set<IndexType> sort_set(vector_of_node_ids.begin(),
                                 vector_of_node_ids.end());
    vector_of_node_ids.assign(sort_set.begin(), sort_set.end());
    rExtractedBoundaryModelPart.AddNodes(vector_of_node_ids);

#ifdef KRATOS_USING_MPI
    ParallelFillCommunicator(rExtractedBoundaryModelPart).Execute();
#endif

}


//
// Specializeing the functions for diff templates
//
template void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<2>(ModelPart &rModelPart,
                                                                      ModelPart &rRemovedModelPart,
                                                                      const ChimeraHoleCuttingUtility::Domain DomainType,
                                                                      const double OverLapDistance,
                                                                      const ChimeraHoleCuttingUtility::SideToExtract Side);

template void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<3>(ModelPart &rModelPart,
                                                                      ModelPart &rRemovedModelPart,
                                                                      const ChimeraHoleCuttingUtility::Domain DomainType,
                                                                      const double OverLapDistance,
                                                                      const ChimeraHoleCuttingUtility::SideToExtract Side);

template void ChimeraHoleCuttingUtility::ExtractBoundaryMesh<2>(ModelPart &rVolumeModelPart,
                                                                ModelPart &rExtractedBoundaryModelPart,
                                                                const ChimeraHoleCuttingUtility::SideToExtract GetInternal);
template void ChimeraHoleCuttingUtility::ExtractBoundaryMesh<3>(ModelPart &rVolumeModelPart,
                                                                ModelPart &rExtractedBoundaryModelPart,
                                                                const ChimeraHoleCuttingUtility::SideToExtract GetInternal);

template void ChimeraHoleCuttingUtility::CreateHoleAfterDistance<2>(ModelPart &rModelPart,
                                                                    ModelPart &rHoleModelPart,
                                                                    ModelPart &rHoleBoundaryModelPart,
                                                                    const double Distance);
template void ChimeraHoleCuttingUtility::CreateHoleAfterDistance<3>(ModelPart &rModelPart,
                                                                    ModelPart &rHoleModelPart,
                                                                    ModelPart &rHoleBoundaryModelPart,
                                                                    const double Distance);

} // namespace Kratos
