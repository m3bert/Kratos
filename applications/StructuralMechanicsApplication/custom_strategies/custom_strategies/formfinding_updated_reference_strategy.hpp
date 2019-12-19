// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Anna Rehr
//

#if !defined(KRATOS_FORMFINDING_UPDATED_REFERENCE_STRATEGY )
#define  KRATOS_FORMFINDING_UPDATED_REFERENCE_STRATEGY

// Project includes
#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"
#include "includes/gid_io.h"
#include "custom_utilities/project_vector_on_surface_utility.h"


namespace Kratos
{
    ///@}
    ///@name Kratos Classes
    ///@{

/**
 * @class FormfindingUpdatedReferenceStrategy
 *
 * @ingroup StrucutralMechanicsApplication
 *
 * @brief inherited class from ResidualBasedNewtonRaphsonStrategy for formfinding
 *
 * @details additions for formfinding: update the reference configuration for each element, initialize the elements for formfinding,
 * adaption line search for formfinding, print formfinding output (different nonlinear iterations)
 *
 * @author Anna Rehr
 */

    template<class TSparseSpace,
    class TDenseSpace, // = DenseSpace<double>,
    class TLinearSolver //= LinearSolver<TSparseSpace,TDenseSpace>
    >
    class FormfindingUpdatedReferenceStrategy
        : public ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
    {
    public:
        ///@name Type Definitions
        ///@{
        typedef ConvergenceCriteria<TSparseSpace, TDenseSpace> TConvergenceCriteriaType;

        // Counted pointer of ClassName
        KRATOS_CLASS_POINTER_DEFINITION(FormfindingUpdatedReferenceStrategy);

        typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
        typedef typename BaseType::TBuilderAndSolverType TBuilderAndSolverType;
        typedef typename BaseType::TSchemeType TSchemeType;
        typedef GidIO<> IterationIOType;
        typedef IterationIOType::Pointer IterationIOPointerType;
        typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
        typedef typename BaseType::TSystemVectorType TSystemVectorType;

        ///@}
        ///@name Life Cycle

        ///@{

        /**
        * Constructor.
        */

        FormfindingUpdatedReferenceStrategy(
            ModelPart& model_part,
            typename TSchemeType::Pointer pScheme,
            typename TLinearSolver::Pointer pNewLinearSolver,
            typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
            int MaxIterations = 30,
            bool CalculateReactions = false,
            bool ReformDofSetAtEachStep = false,
            bool MoveMeshFlag = false,
            bool PrintIterations = false,
            bool IncludeLineSearch = false
            )
            : ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part, pScheme,
                pNewLinearSolver,
                pNewConvergenceCriteria,
                MaxIterations,
                CalculateReactions,
                ReformDofSetAtEachStep,
                MoveMeshFlag),
                mPrintIterations(PrintIterations),
                mIncludeLineSearch(IncludeLineSearch)
        {
            if (PrintIterations)
            {
                mPrintIterations = true;
                InitializeIterationIO();
            }
        }

        // constructor with Builder and Solver
        FormfindingUpdatedReferenceStrategy(
            ModelPart& model_part,
            typename TSchemeType::Pointer pScheme,
            typename TLinearSolver::Pointer pNewLinearSolver,
            typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
            typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
            int MaxIterations = 30,
            bool CalculateReactions = false,
            bool ReformDofSetAtEachStep = false,
            bool MoveMeshFlag = false,
            bool PrintIterations = false,
            bool IncludeLineSearch = false
            )
            : ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part, pScheme,
                pNewLinearSolver,pNewConvergenceCriteria,pNewBuilderAndSolver,MaxIterations,CalculateReactions,ReformDofSetAtEachStep,
                MoveMeshFlag), mPrintIterations(PrintIterations), mIncludeLineSearch(IncludeLineSearch)
        {
            if (PrintIterations)
            {
                mPrintIterations = true;
                InitializeIterationIO();
            }
        }

        /**
        * Destructor.
        */

        ~FormfindingUpdatedReferenceStrategy() override
        {
        }

        /**
        * Initialization. In addition to the base class initialization, the elements are initialized for formfinding
        */

        void Initialize() override
        {
            KRATOS_TRY;
            // set elemental values for formfinding
            for(auto& elem : BaseType::GetModelPart().Elements())
                elem.SetValue(IS_FORMFINDING, true);
            BaseType::Initialize();
            KRATOS_CATCH("");
        }


        bool SolveSolutionStep() override
        {
            if (mPrintIterations)
            {
                KRATOS_ERROR_IF_NOT(mpIterationIO) << " IterationIO is uninitialized!" << std::endl;
                mpIterationIO->InitializeResults(0.0, BaseType::GetModelPart().GetMesh());
            }

            BaseType::SolveSolutionStep();

            if (mPrintIterations){
                mpIterationIO->FinalizeResults();
                KRATOS_WATCH(mPrintIterations)
            }


            return true;
        }

          ///@}


    protected:
        ///@name Protected Operators
        ///@{

void UpdateDatabase(
        TSystemMatrixType& A,
        TSystemVectorType& Dx,
        TSystemVectorType& b,
        const bool MoveMesh
    ) override
    {
        BaseType::UpdateDatabase(A,Dx, b, MoveMesh);
        ModelPart& r_model_part = BaseType::GetModelPart();
        for(auto& r_node : r_model_part.Nodes()){
            const array_1d<double, 3>& disp = r_node.FastGetSolutionStepValue(DISPLACEMENT);
            const array_1d<double, 3>& step_disp = r_node.GetValue(VELOCITY);
            // if(r_node.Id()==122){
            //     KRATOS_WATCH(disp)
            //     KRATOS_WATCH(r_node.Y0())
            //     KRATOS_WATCH(step_disp)
            // }
            r_node.GetValue(VELOCITY) += r_node.FastGetSolutionStepValue(DISPLACEMENT);
            // Updating reference
            r_node.X0() += disp[0];
            r_node.Y0() += disp[1];
            r_node.Z0() += disp[2];

            // if(r_node.Id()==122){
            //     KRATOS_WATCH(disp)
            //     KRATOS_WATCH(r_node.Y0())
            //     KRATOS_WATCH(step_disp)
            // }

            r_node.FastGetSolutionStepValue(DISPLACEMENT) = ZeroVector(3);

            // if(r_node.Id()==122){
            //     KRATOS_WATCH(disp)
            //     KRATOS_WATCH(r_node.Y0())
            //     KRATOS_WATCH(step_disp)
            // }
        }

        Parameters settings( R"({
            "model_part_name"  : "Structure",
            "echo_level"       : 1,
            "projection_type"  : "radial",
            "global_direction" : [0,0,1],
            "variable_name"    : "LOCAL_PRESTRESS_AXIS_1",
            "method_specific_settings" : { }
        } )" );

        ProjectVectorOnSurfaceUtility::Execute(r_model_part, settings);

        // r_model_part.GetProcessInfo()[STEP] += 1;
        // r_model_part.GetProcessInfo()[TIME] += 0.1;
    }

    private:
        ///@name Member Variables
        ///@{
        bool mPrintIterations;
        bool mIncludeLineSearch;
        IterationIOPointerType mpIterationIO;
        ///@}
        ///@name Private Operators
        ///@{

        /**
        * Copy constructor.
        */

        FormfindingUpdatedReferenceStrategy(const FormfindingUpdatedReferenceStrategy& Other)
        {
        };

        void EchoInfo(const unsigned int IterationNumber) override
        {
            BaseType::EchoInfo(IterationNumber);

            if (mPrintIterations)
            {
                KRATOS_ERROR_IF_NOT(mpIterationIO) << " IterationIO is uninitialized!" << std::endl;
                mpIterationIO->WriteNodalResults(DISPLACEMENT, BaseType::GetModelPart().Nodes(), IterationNumber, 0);
            }
        }

        void InitializeIterationIO()
        {
            mpIterationIO = Kratos::make_unique<IterationIOType>(
                "Formfinding_Iterations",
                GiD_PostAscii, // GiD_PostAscii // for debugging GiD_PostBinary
                MultiFileFlag::SingleFile,
                WriteDeformedMeshFlag::WriteUndeformed,
                WriteConditionsFlag::WriteConditions);

            mpIterationIO->InitializeMesh(0.0);
            mpIterationIO->WriteMesh(BaseType::GetModelPart().GetMesh());
            mpIterationIO->WriteNodeMesh(BaseType::GetModelPart().GetMesh());
            mpIterationIO->FinalizeMesh();
        }
        ///@}
    }; /* Class FormfindingUpdatedReferenceStrategy */
       ///@}
} /* namespace Kratos. */

#endif /* KRATOS_FORMFINDING_UPDATED_REFERENCE_STRATEGY defined */
