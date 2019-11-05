//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
// ==============================================================================
//  ChimeraApplication
//
//  License:         BSD License
//                   license: ChimeraApplication/license.txt
//
//  Authors:        Aditya Ghantasala, https://github.com/adityaghantasala
// 					Navaneeth K Narayanan
//					Rishith Ellath Meethal
// ==============================================================================
//
#if !defined(KRATOS_APPLY_CHIMERA_H_INCLUDED)
#define KRATOS_APPLY_CHIMERA_H_INCLUDED

// System includes
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "omp.h"

// External includes

// Project includes
#include "includes/define.h"
#include "includes/process_info.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "containers/model.h"
#include "includes/variables.h"
#include "includes/linear_master_slave_constraint.h"
#include "processes/calculate_distance_to_skin_process.h"
#include "input_output/vtk_output.h"
#include "utilities/binbased_fast_point_locator.h"
#include "factories/standard_linear_solver_factory.h"
#include "utilities/builtin_timer.h"
#include "utilities/variable_utils.h"

// Application includes
#include "chimera_application_variables.h"
#include "custom_utilities/hole_cutting_utility.h"
#include "custom_utilities/distance_calcuation_utility.h"

namespace Kratos
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

template <int TDim, class TSparseSpaceType, class TLocalSpaceType>
class KRATOS_API(CHIMERA_APPLICATION) ApplyChimera : public Process
{
public:
    ///@name Type Definitions
    ///@{

    ///@}
    ///@name Pointer Definitions
    typedef ProcessInfo::Pointer ProcessInfoPointerType;
    typedef Kratos::VariableComponent<Kratos::VectorComponentAdaptor<Kratos::array_1d<double, 3>>> VariableComponentType;
    typedef std::size_t IndexType;
    typedef Kratos::Variable<double> VariableType;
    typedef std::vector<IndexType> ConstraintIdsVectorType;
    typedef typename ModelPart::MasterSlaveConstraintType MasterSlaveConstraintType;
    typedef typename ModelPart::MasterSlaveConstraintContainerType MasterSlaveConstraintContainerType;
    typedef std::vector<MasterSlaveConstraintContainerType> MasterSlaveContainerVectorType;
    typedef BinBasedFastPointLocator<TDim> PointLocatorType;
    typedef typename PointLocatorType::Pointer PointLocatorPointerType;

    KRATOS_CLASS_POINTER_DEFINITION(ApplyChimera);

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Constructor
     * @param rMainModelPart The reference to the modelpart which will be used for computations later on.
     * @param iParameters The settings parameters.
     */
    explicit ApplyChimera(ModelPart &rMainModelPart, Parameters iParameters) : mrMainModelPart(rMainModelPart), mParameters(iParameters)
    {
        // This is only for example
        Parameters example_parameters(R"(
            {
               	"chimera_parts"   :   [
									[{
										"model_part_name":"PLEASE_SPECIFY",
                                        "search_model_part_name":"PLEASE_SPECIFY",
                                        "boundary_model_part_name":"PLEASE_SPECIFY",
										"overlap_distance":0.0
									}],
									[{
										"model_part_name":"PLEASE_SPECIFY",
                                        "search_model_part_name":"PLEASE_SPECIFY",
                                        "boundary_model_part_name":"PLEASE_SPECIFY",
										"overlap_distance":0.0
									}],
									[{
										"model_part_name":"PLEASE_SPECIFY",
                                        "search_model_part_name":"PLEASE_SPECIFY",
                                        "boundary_model_part_name":"PLEASE_SPECIFY",
										"overlap_distance":0.0
									}]
								]
            })");
        mNumberOfLevels = mParameters.size();
        KRATOS_ERROR_IF(mNumberOfLevels < 2) << "Chimera requires atleast two levels !. Put Background in one level and the patch in second one." << std::endl;

        ProcessInfoPointerType info = mrMainModelPart.pGetProcessInfo();
        mEchoLevel = 0;
        mReformulateEveryStep = false;
        mIsFormulated = false;
    }

    /// Destructor.
    virtual ~ApplyChimera()
    {
    }

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    void SetEchoLevel(int EchoLevel)
    {
        mEchoLevel = EchoLevel;
    }

    void SetReformulateEveryStep(bool Reformulate)
    {
        mReformulateEveryStep = Reformulate;
    }

    virtual void ExecuteInitializeSolutionStep() override
    {
        KRATOS_TRY;
        // Actual execution of the functionality of this class
        if (mReformulateEveryStep || !mIsFormulated)
        {
            DoChimeraLoop();
            mIsFormulated = true;
        }
        KRATOS_CATCH("");
    }

    virtual void ExecuteFinalizeSolutionStep() override
    {
        VariableUtils().SetFlag(VISITED, false, mrMainModelPart.Nodes());
        VariableUtils().SetFlag(VISITED, false, mrMainModelPart.Elements());
        VariableUtils().SetNonHistoricalVariable(SPLIT_ELEMENT, false, mrMainModelPart.Elements());

        if (mReformulateEveryStep)
        {
            mrMainModelPart.RemoveMasterSlaveConstraintsFromAllLevels(TO_ERASE);
        }
        mIsFormulated = false;
    }

    virtual std::string Info() const override
    {
        return "ApplyChimera";
    }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream &rOStream) const override
    {
        rOStream << "ApplyChimera" << std::endl;
    }

    /// Print object's data.
    virtual void PrintData(std::ostream &rOStream) const override
    {
        rOStream << "ApplyChimera" << std::endl;
    }

    ///@}
    ///@name Friends
    ///@{

    ///@}

protected:
    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{
    ModelPart &mrMainModelPart;
    IndexType mNumberOfLevels;
    Parameters mParameters;
    std::unordered_map<IndexType, ConstraintIdsVectorType> mNodeIdToConstraintIdsMap;
    int mEchoLevel;
    bool mReformulateEveryStep;
    std::map<std::string, PointLocatorPointerType> mPointLocatorsMap;
    bool mIsFormulated;
    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Does a loop on the background and patch combinations possible and uses FormulateChimera method.
     */
    virtual void DoChimeraLoop()
    {
        const int num_elements = static_cast<int>(mrMainModelPart.NumberOfElements());
        const auto elem_begin = mrMainModelPart.ElementsBegin();

        Parameters parameters_for_validation(R"(
                                {
                                    "model_part_name":"PLEASE_SPECIFY",
                                    "search_model_part_name":"PLEASE_SPECIFY",
                                    "boundary_model_part_name":"PLEASE_SPECIFY",
                                    "overlap_distance":0.0
                                }
        )");

#pragma omp parallel for
        for (int i_be = 0; i_be < num_elements; ++i_be)
        {
            auto i_elem = elem_begin + i_be;
            if (!i_elem->Is(VISITED)) //for multipatch
                i_elem->Set(ACTIVE, true);
        }

        const int num_nodes = static_cast<int>(mrMainModelPart.NumberOfNodes());
        const auto nodes_begin = mrMainModelPart.ElementsBegin();

#pragma omp parallel for
        for (int i_bn = 0; i_bn < num_nodes; ++i_bn)
        {
            auto i_node = nodes_begin + i_bn;
            i_node->Set(VISITED, false);
        }
        BuiltinTimer do_chimera_loop_time;

        int i_current_level = 0;
        for (auto &current_level : mParameters)
        {
            ChimeraHoleCuttingUtility::Domain domain_type = ChimeraHoleCuttingUtility::Domain::MAIN_BACKGROUND;
            for (auto &background_patch_param : current_level)
            { // Gives the current background patch
                background_patch_param.ValidateAndAssignDefaults(parameters_for_validation);
                Model &current_model = mrMainModelPart.GetModel();
                ModelPart &r_background_model_part = current_model.GetModelPart(background_patch_param["model_part_name"].GetString());
                if (!r_background_model_part.HasSubModelPart("chimera_boundary_mp"))
                {
                    auto &r_boundary_model_part = r_background_model_part.CreateSubModelPart("chimera_boundary_mp");
                    BuiltinTimer extraction_time;
                    if (i_current_level == 0)
                    {
                        ChimeraHoleCuttingUtility().ExtractBoundaryMesh<TDim>(r_background_model_part, r_boundary_model_part);
                    }
                    else
                    {
                        ChimeraHoleCuttingUtility().ExtractBoundaryMesh<TDim>(r_background_model_part, r_boundary_model_part, ChimeraHoleCuttingUtility::SideToExtract::INSIDE);
                    }
                    KRATOS_INFO_IF("Extraction of boundary mesh took : ", mEchoLevel > 0) << extraction_time.ElapsedSeconds() << " seconds" << std::endl;
                }
                // compute the outerboundary of the background to save
                for (IndexType i_slave_level = i_current_level + 1; i_slave_level < mNumberOfLevels; ++i_slave_level)
                {
                    for (auto &slave_patch_param : mParameters[i_slave_level]) // Loop over all other slave patches
                    {
                        KRATOS_INFO_IF("Formulating Chimera for the combination :: ", mEchoLevel > 0) << "Background" << background_patch_param << "\n Patch::" << slave_patch_param << std::endl;
                        slave_patch_param.ValidateAndAssignDefaults(parameters_for_validation);
                        if (i_current_level == 0) // a check to identify computational Domain boundary
                            domain_type = ChimeraHoleCuttingUtility::Domain::OTHER;
                        FormulateChimera(background_patch_param, slave_patch_param, domain_type);
                        mPointLocatorsMap.erase(background_patch_param["model_part_name"].GetString());
                    }
                }
            }
            ++i_current_level;
        }

        mNodeIdToConstraintIdsMap.clear();
        mPointLocatorsMap.clear();

        KRATOS_INFO_IF("Chrimera Initialization took : ", mEchoLevel > 0)
            << do_chimera_loop_time.ElapsedSeconds() << " seconds" << std::endl;

        KRATOS_INFO_IF("Total number of constraints created so far", mEchoLevel > 0) << mrMainModelPart.NumberOfMasterSlaveConstraints() << std::endl;
    }

    /**
     * @brief Formulates the Chimera conditions with a given set of background and patch combination.
     * @param BackgroundParam Parameters/Settings for the background
     * @param PatchParameters Parameters/Settings for the Patch
     * @param DomainType Flag specifying if the background is the main bg or not
     */
    virtual void FormulateChimera(const Parameters BackgroundParam, const Parameters PatchParameters, ChimeraHoleCuttingUtility::Domain DomainType)
    {
        Model &current_model = mrMainModelPart.GetModel();
        const auto& r_comm = mrMainModelPart.GetCommunicator().GetDataCommunicator();
        ModelPart &r_background_model_part = current_model.GetModelPart(BackgroundParam["model_part_name"].GetString());
        ModelPart &r_background_boundary_model_part = r_background_model_part.GetSubModelPart("chimera_boundary_mp");
        ModelPart &r_patch_model_part = current_model.GetModelPart(PatchParameters["model_part_name"].GetString());
        const std::string bg_search_mp_name = BackgroundParam["search_model_part_name"].GetString();
        auto &r_background_search_model_part = current_model.HasModelPart(bg_search_mp_name) ? current_model.GetModelPart(bg_search_mp_name) : r_background_model_part;

        const double overlap_bg = BackgroundParam["overlap_distance"].GetDouble();
        const double overlap_pt = PatchParameters["overlap_distance"].GetDouble();
        const double over_lap_distance = (overlap_bg > overlap_pt) ? overlap_bg : overlap_pt;

        BuiltinTimer search_creation_time;
        PointLocatorPointerType p_point_locator_on_background = GetPointLocator(r_background_search_model_part);
        PointLocatorPointerType p_pointer_locator_on_patch = GetPointLocator(r_patch_model_part);
        auto elapsed_search_creation_time = search_creation_time.ElapsedSeconds();
        r_comm.Max(elapsed_search_creation_time, 0);
        KRATOS_INFO_IF("Creation of search structures took : ", mEchoLevel > 0) << elapsed_search_creation_time << " seconds" << std::endl;
        KRATOS_ERROR_IF(over_lap_distance < 1e-12) << "Overlap distance should be a positive and non-zero number." << std::endl;

        ModelPart &r_hole_model_part = current_model.CreateModelPart("HoleModelpart");
        ModelPart &r_hole_boundary_model_part = current_model.CreateModelPart("HoleBoundaryModelPart");

        auto &r_modified_patch_boundary_model_part = ExtractPatchBoundary(PatchParameters, r_background_boundary_model_part, DomainType);

        BuiltinTimer bg_distance_calc_time;
        DistanceCalculationUtility<TDim, TSparseSpaceType, TLocalSpaceType>::CalculateDistance(r_background_model_part,
                                                                                               r_modified_patch_boundary_model_part);
        auto bg_distance_calc_time_elapsed = bg_distance_calc_time.ElapsedSeconds();
        r_comm.Max(bg_distance_calc_time_elapsed, 0);
        KRATOS_INFO_IF("Distance calculation on background took : ", mEchoLevel > 0) << bg_distance_calc_time_elapsed << " seconds" << std::endl;

        BuiltinTimer hole_creation_time;
        ChimeraHoleCuttingUtility().CreateHoleAfterDistance<TDim>(r_background_model_part, r_hole_model_part, r_hole_boundary_model_part, over_lap_distance);
        auto hole_creation_time_elapsed = hole_creation_time.ElapsedSeconds();
        r_comm.Max(hole_creation_time_elapsed, 0);
        KRATOS_INFO_IF("Hole creation took : ", mEchoLevel > 0) << hole_creation_time_elapsed << " seconds" << std::endl;


        WriteModelPart(r_hole_model_part);
        // WriteModelPart(r_modified_patch_boundary_model_part);
        // WriteModelPart(r_background_boundary_model_part);

        const int n_elements = static_cast<int>(r_hole_model_part.NumberOfElements());
#pragma omp parallel for
        for (int i_elem = 0; i_elem < n_elements; ++i_elem)
        {
            ModelPart::ElementsContainerType::iterator it_elem = r_hole_model_part.ElementsBegin() + i_elem;
            it_elem->Set(VISITED, true); //for multipatch
        }

        BuiltinTimer mpc_time;
        ApplyContinuityWithMpcs(r_modified_patch_boundary_model_part, *p_point_locator_on_background);
        ApplyContinuityWithMpcs(r_hole_boundary_model_part, *p_pointer_locator_on_patch);
        auto mpc_time_elapsed = mpc_time.ElapsedSeconds();
        r_comm.Max(mpc_time_elapsed, 0);
        KRATOS_INFO_IF("Creation of MPC for chimera took : ", mEchoLevel > 0) << mpc_time_elapsed << " seconds" << std::endl;

        current_model.DeleteModelPart("HoleModelpart");
        current_model.DeleteModelPart("HoleBoundaryModelPart");
        current_model.DeleteModelPart("ModifiedPatchBoundary");

        r_comm.Barrier();
        exit(-1);
        KRATOS_INFO("End of Formulate Chimera") << std::endl;
    }

    /**
     * @brief Creates a vector of unique constraint ids based on how many required and how many are already present in the mrModelPart.
     * @param rIdVector The vector which is populated with unique constraint ids.
     * @param NumberOfConstraintsRequired The number of further constraints required. used for calculation of unique ids.
     */
    virtual void CreateConstraintIds(std::vector<int> &rIdVector, const IndexType NumberOfConstraintsRequired)
    {
        IndexType max_constraint_id = 0;
        // Get current maximum constraint ID
        if (mrMainModelPart.MasterSlaveConstraints().size() != 0)
        {
            mrMainModelPart.MasterSlaveConstraints().Sort();
            ModelPart::MasterSlaveConstraintContainerType::iterator it = mrMainModelPart.MasterSlaveConstraintsEnd() - 1;
            max_constraint_id = (*it).Id();
            ++max_constraint_id;
        }

        // Now create a vector size NumberOfConstraintsRequired
        rIdVector.resize(NumberOfConstraintsRequired * (TDim + 1));
        std::iota(std::begin(rIdVector), std::end(rIdVector), max_constraint_id); // Fill with consecutive integers
    }

    /**
     * @brief Applies the continuity between the boundary modelpart and the background.
     * @param rBoundaryModelPart The boundary modelpart for which the continuity is to be enforced.
     * @param pBinLocator The bin based locator formulated on the background. This is used to locate nodes of rBoundaryModelPart on background.
     */
    virtual void ApplyContinuityWithMpcs(ModelPart &rBoundaryModelPart, PointLocatorType &pBinLocator)
    {
    }

    /**
     * @brief Applies the master-slave constraint between the given master and slave nodes with corresponding variable.
     * @param rMasterSlaveContainer The container to which the constraint to be added (useful to so OpenMP loop)
     * @param rCloneConstraint The prototype of constraint which is to be added.
     * @param ConstraintId The ID of the constraint to be added.
     * @param rMasterNode The Master node of the constraint.
     * @param rMasterVariable The variable for the master node.
     * @param rSlaveNode The Slave node of the constraint.
     * @param rSlaveVariable The variable for the slave node.
     * @param Weight The weight of the Master node.
     * @param Constant The constant of the master slave relation.
     */
    template <typename TVariableType>
    void AddMasterSlaveRelation(MasterSlaveConstraintContainerType &rMasterSlaveContainer,
                                const LinearMasterSlaveConstraint &rCloneConstraint,
                                unsigned int ConstraintId,
                                Node<3> &rMasterNode,
                                TVariableType &rMasterVariable,
                                Node<3> &rSlaveNode,
                                TVariableType &rSlaveVariable,
                                const double Weight,
                                const double Constant = 0.0)
    {
        rSlaveNode.Set(SLAVE);
        ModelPart::MasterSlaveConstraintType::Pointer p_new_constraint = rCloneConstraint.Create(ConstraintId, rMasterNode, rMasterVariable, rSlaveNode, rSlaveVariable, Weight, Constant);
        p_new_constraint->Set(TO_ERASE);
        mNodeIdToConstraintIdsMap[rSlaveNode.Id()].push_back(ConstraintId);
        //TODO: Check if an insert(does a sort) is required here or just use a push_back and then
        //      sort once at the end.
        // rMasterSlaveContainer.insert(rMasterSlaveContainer.begin(), p_new_constraint);
        rMasterSlaveContainer.push_back(p_new_constraint);
    }

    /**
     * @brief Applies the master-slave constraint to enforce the continuity between a given geometry/element and a boundary node
     * @param rGeometry The geometry of the element
     * @param rBoundaryNode The boundary node for which the connections are to be made.
     * @param rShapeFuncWeights The shape function weights for the node in the rGeometry.
     * @param StartIndex The start Index of the constraints which are to be added.
     * @param rConstraintIdVector The vector of the constraints Ids which is accessed with StartIndex.
     * @param rMsContainer The Constraint container to which the contraints are added.
     */
    template <typename TVariableType>
    void ApplyContinuityWithElement(Geometry<Node<3>> &rGeometry,
                                    Node<3> &rBoundaryNode,
                                    Vector &rShapeFuncWeights,
                                    TVariableType &rVariable,
                                    unsigned int StartIndex,
                                    std::vector<int> &rConstraintIdVector,
                                    MasterSlaveConstraintContainerType &rMsContainer)
    {
        const auto &r_clone_constraint = (LinearMasterSlaveConstraint)KratosComponents<MasterSlaveConstraint>::Get("LinearMasterSlaveConstraint");
        // Initialise the boundary nodes dofs to 0 at ever time steps
        rBoundaryNode.FastGetSolutionStepValue(rVariable, 0) = 0.0;
        for (std::size_t i = 0; i < rGeometry.size(); i++)
        {
            //Interpolation of rVariable
            rBoundaryNode.FastGetSolutionStepValue(rVariable, 0) += rGeometry[i].GetDof(rVariable).GetSolutionStepValue(0) * rShapeFuncWeights[i];
            AddMasterSlaveRelation(rMsContainer, r_clone_constraint, rConstraintIdVector[StartIndex++], rGeometry[i], rVariable, rBoundaryNode, rVariable, rShapeFuncWeights[i]);
        } // end of loop over host element nodes

        // Setting the buffer 1 same buffer 0
        rBoundaryNode.FastGetSolutionStepValue(rVariable, 1) = rBoundaryNode.FastGetSolutionStepValue(rVariable, 0);
    }

    /**
     * @brief This function reserves the necessary memory for the contraints in each thread
     * @param rModelPart The modelpart to which the constraints are to be added.
     * @param rContainerVector The container vector which has constraints to transfer
     */
    void ReserveMemoryForConstraintContainers(ModelPart &rModelPart, MasterSlaveContainerVectorType &rContainerVector)
    {
#pragma omp parallel
        {
            const IndexType num_constraints_per_thread = (rModelPart.NumberOfNodes() * 4) / omp_get_num_threads();
#pragma omp single
            {
                rContainerVector.resize(omp_get_num_threads());
                for (auto &container : rContainerVector)
                    container.reserve(num_constraints_per_thread);
            }
        }
    }

    /**
     * @brief Given a node, this funcition finds and deletes all the existing constraints for that node
     * @param rBoundaryNode The boundary node for which the connections are to be made.
     */
    int RemoveExistingConstraintsForNode(ModelPart::NodeType &rBoundaryNode)
    {
        ConstraintIdsVectorType constrainIds_for_the_node;
        int removed_counter = 0;
        constrainIds_for_the_node = mNodeIdToConstraintIdsMap[rBoundaryNode.Id()];
        for (auto const &constraint_id : constrainIds_for_the_node)
        {
#pragma omp critical
            {
                mrMainModelPart.RemoveMasterSlaveConstraintFromAllLevels(constraint_id);
                removed_counter++;
            }
        }
        constrainIds_for_the_node.clear();
        return removed_counter;
    }

    /**
     * @brief The function transfers all the constraints in the container vector to the modelpart.
     *          IMPORTANT: The constraints are directly added to the constraints container of the
     *                     modelpart. So the parent and child modelparts have no info about these
     *                     constraints.
     * @param rModelPart The modelpart to which the constraints are to be added.
     * @param rContainerVector The container vector which has constraints to transfer
     */
    void AddConstraintsToModelpart(ModelPart &rModelPart, MasterSlaveContainerVectorType &rContainerVector)
    {
        IndexType n_total_constraints = 0;
        for (auto &container : rContainerVector)
        {
            const int n_constraints = static_cast<int>(container.size());
            n_total_constraints += n_constraints;
        }

        auto &constraints = rModelPart.MasterSlaveConstraints();
        constraints.reserve(n_total_constraints);
        auto &constraints_data = constraints.GetContainer();
        for (auto &container : rContainerVector)
        {
            constraints_data.insert(constraints_data.end(), container.ptr_begin(), container.ptr_end());
        }
        constraints.Sort();
    }

    ///@}
    ///@name Protected  Access
    ///@{

    ///@}
    ///@name Protected Inquiry
    ///@{

    ///@}
    ///@name Protected LifeCycle
    ///@{

    ///@}

private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Extracts the patch boundary modelpart
     * @param rBackgroundBoundaryModelpart background boundary to remove out of domain patch
     * @param PatchParameters Parameters/Settings for the Patch
     * @param DomainType Flag specifying if the background is the main bg or not
     */
    ModelPart &ExtractPatchBoundary(const Parameters PatchParameters, ModelPart &rBackgroundBoundaryModelpart, const ChimeraHoleCuttingUtility::Domain DomainType)
    {
        Model &current_model = rBackgroundBoundaryModelpart.GetModel();
        const auto& r_comm = rBackgroundBoundaryModelpart.GetCommunicator().GetDataCommunicator();
        const std::string patch_boundary_mp_name = PatchParameters["boundary_model_part_name"].GetString();

        if (!current_model.HasModelPart(patch_boundary_mp_name))
        {
            ModelPart &r_patch_model_part = current_model.GetModelPart(PatchParameters["model_part_name"].GetString());
            ModelPart &r_modified_patch_model_part = current_model.CreateModelPart("ModifiedPatch");
            ModelPart &r_modified_patch_boundary_model_part = current_model.CreateModelPart("ModifiedPatchBoundary");
            BuiltinTimer distance_calc_time_patch;
            DistanceCalculationUtility<TDim, TSparseSpaceType, TLocalSpaceType>::CalculateDistance(r_patch_model_part,
                                                                                                   rBackgroundBoundaryModelpart);
            auto dist_calc_time_elapsed = distance_calc_time_patch.ElapsedSeconds();
            r_comm.Max(dist_calc_time_elapsed,0);
            KRATOS_INFO_IF("Distance calculation on patch took : ", mEchoLevel > 0) << dist_calc_time_elapsed << " seconds" << std::endl;
            //TODO: Below is brutforce. Check if the boundary of bg is actually cutting the patch.
            BuiltinTimer rem_out_domain_time;
            ChimeraHoleCuttingUtility().RemoveOutOfDomainElements<TDim>(r_patch_model_part, r_modified_patch_model_part, DomainType, ChimeraHoleCuttingUtility::SideToExtract::INSIDE);
            auto rem_out_domain_time_elapsed = rem_out_domain_time.ElapsedSeconds();
            r_comm.Max(rem_out_domain_time_elapsed,0);
            KRATOS_INFO_IF("Removing out of domain patch took : ", mEchoLevel > 0) << rem_out_domain_time_elapsed << " seconds" << std::endl;

            BuiltinTimer patch_boundary_extraction_time;
            ChimeraHoleCuttingUtility().ExtractBoundaryMesh<TDim>(r_modified_patch_model_part, r_modified_patch_boundary_model_part);
            auto patch_boundary_extraction_time_elapsed = patch_boundary_extraction_time.ElapsedSeconds();
            r_comm.Max(patch_boundary_extraction_time_elapsed, 0);
            KRATOS_INFO_IF("Extraction of patch boundary took : ", mEchoLevel > 0) << patch_boundary_extraction_time_elapsed << " seconds" << std::endl;
            current_model.DeleteModelPart("ModifiedPatch");
            return r_modified_patch_boundary_model_part;
        }
        else
        {
            return current_model.GetModelPart(patch_boundary_mp_name);
        }
    }

    /**
     * @brief Creates or returns an existing point locator on a given ModelPart
     * @param rModelPart The modelpart on which a point locator is to be obtained.
     */
    PointLocatorPointerType GetPointLocator(ModelPart &rModelPart)
    {
        if (mPointLocatorsMap.count(rModelPart.Name()) == 0)
        {
            PointLocatorPointerType p_point_locator = Kratos::make_shared<PointLocatorType>(rModelPart);
            p_point_locator->UpdateSearchDatabase();
            mPointLocatorsMap[rModelPart.Name()] = p_point_locator;
            return p_point_locator;
        }
        else
        {
            auto p_point_locator = mPointLocatorsMap.at(rModelPart.Name());
            if (mReformulateEveryStep)
                p_point_locator->UpdateSearchDatabase();
            return p_point_locator;
        }
    }


    /**
     * @brief Utility function to print out various intermediate modelparts
     * @param rModelPart ModelPart to be printed.
     */
    void WriteModelPart(ModelPart &rModelPart)
    {

        if(rModelPart.NumberOfNodes()<=0)
            return;
        Parameters vtk_parameters(R"(
                {
                    "model_part_name"                    : "HoleModelpart",
                    "output_control_type"                : "step",
                    "output_frequency"                   : 1,
                    "file_format"                        : "ascii",
                    "output_precision"                   : 3,
                    "output_sub_model_parts"             : false,
                    "folder_name"                        : "test_vtk_output",
                    "save_output_files_in_folder"        : true,
                    "nodal_solution_step_data_variables" : ["VELOCITY","PRESSURE","DISTANCE"],
                    "nodal_data_value_variables"         : [],
                    "element_flags"                      : ["ACTIVE"],
                    "nodal_flags"                        : ["VISITED"],
                    "element_data_value_variables"       : [],
                    "condition_data_value_variables"     : []
                }
                )");

        VtkOutput vtk_output(rModelPart, vtk_parameters);
        vtk_output.PrintOutput();
    }


    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    ApplyChimera &operator=(ApplyChimera const &rOther);

    ///@}
}; // Class ApplyChimera

} // namespace Kratos.

#endif //  KRATOS_APPLY_CHIMERA_H_INCLUDED defined
