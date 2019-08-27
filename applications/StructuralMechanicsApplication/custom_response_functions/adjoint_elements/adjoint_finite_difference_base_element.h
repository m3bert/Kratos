// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Armin Geiser, https://github.com/armingeiser
//

#if !defined(ADJOINT_FINITE_DIFFERENCE_BASE_ELEMENT_H_INCLUDED )
#define  ADJOINT_FINITE_DIFFERENCE_BASE_ELEMENT_H_INCLUDED


// System includes

// External includes

// Project includes
#include "includes/element.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/openmp_utils.h"

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

/** \brief AdjointFiniteDifferencingBaseElement
 *
 * This element a wrapper for a primal element to calculate element derivatives using
 * finite differencing  (adjoint semi analytic approach). It is designed to be used in adjoint
 * sensitivity analysis
 */
template <typename TPrimalElement>
class AdjointFiniteDifferencingBaseElement : public Element
{
public:

    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AdjointFiniteDifferencingBaseElement);
    ///@}

    ///@name Classes
    ///@{
    ///@}

    ///@name Life Cycle
    ///@{

    AdjointFiniteDifferencingBaseElement(IndexType NewId = 0,
                        bool HasRotationDofs = false)
    : Element(NewId),
      mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGetGeometry())),
      mHasRotationDofs(HasRotationDofs)
    {
    }

    AdjointFiniteDifferencingBaseElement(IndexType NewId,
                        GeometryType::Pointer pGeometry,
                        bool HasRotationDofs = false)
    : Element(NewId, pGeometry),
      mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGeometry)),
      mHasRotationDofs(HasRotationDofs)
    {
    }

    AdjointFiniteDifferencingBaseElement(IndexType NewId,
                        GeometryType::Pointer pGeometry,
                        PropertiesType::Pointer pProperties,
                        bool HasRotationDofs = false)
    : Element(NewId, pGeometry, pProperties),
      mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGeometry, pProperties)),
      mHasRotationDofs(HasRotationDofs)
    {
    }

    ///@}

    ///@}
    ///@name Informations
    ///@{

    ///@name Operations
    ///@{

    Element::Pointer Create(IndexType NewId,
                              NodesArrayType const& ThisNodes,
                              PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive<AdjointFiniteDifferencingBaseElement<TPrimalElement>>(
            NewId, GetGeometry().Create(ThisNodes), pProperties);
    }

    Element::Pointer Create(IndexType NewId,
                              GeometryType::Pointer pGeometry,
                              PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive<AdjointFiniteDifferencingBaseElement<TPrimalElement>>(
            NewId, pGeometry, pProperties);
    }

    void EquationIdVector(EquationIdVectorType& rResult, ProcessInfo& rCurrentProcessInfo) override;

    void GetDofList(DofsVectorType& ElementalDofList, ProcessInfo& CurrentProcessInfo) override;


    IntegrationMethod GetIntegrationMethod() const override
    {
        return mpPrimalElement->GetIntegrationMethod();
    }

    void GetValuesVector(Vector& values, int Step = 0) override;

    void Initialize() override
    {
        mpPrimalElement->Initialize();
    }

    void ResetConstitutiveLaw() override
    {
        mpPrimalElement->ResetConstitutiveLaw();
    }

    void CleanMemory() override
    {
        mpPrimalElement->CleanMemory();
    }

    void InitializeSolutionStep(ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->InitializeSolutionStep(rCurrentProcessInfo);
    }

    void InitializeNonLinearIteration(ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->InitializeNonLinearIteration(rCurrentProcessInfo);
    }

    void FinalizeNonLinearIteration(ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->FinalizeNonLinearIteration(rCurrentProcessInfo);
    }

    void FinalizeSolutionStep(ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->FinalizeSolutionStep(rCurrentProcessInfo);
    }

    void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
                                      VectorType& rRightHandSideVector,
                                      ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateLocalSystem(rLeftHandSideMatrix,
                                              rRightHandSideVector,
                                              rCurrentProcessInfo);
    }

    void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
                                       ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateLeftHandSide(rLeftHandSideMatrix,
                                               rCurrentProcessInfo);
    }

    void CalculateLeftHandSide(std::vector< MatrixType >& rLeftHandSideMatrices,
					const std::vector< Variable< MatrixType > >& rLHSVariables,
					ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateLeftHandSide(rLeftHandSideMatrices,
                                               rLHSVariables,
                                               rCurrentProcessInfo);
    }

    void CalculateRightHandSide(VectorType& rRightHandSideVector,
                                        ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateRightHandSide(rRightHandSideVector,
                                                rCurrentProcessInfo);
    }

    void CalculateRightHandSide(std::vector< VectorType >& rRightHandSideVectors,
					const std::vector< Variable< VectorType > >& rRHSVariables,
					ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateRightHandSide(rRightHandSideVectors,
                                                rRHSVariables,
                                                rCurrentProcessInfo);
    }

    void CalculateFirstDerivativesContributions(MatrixType& rLeftHandSideMatrix,
							VectorType& rRightHandSideVector,
							ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateFirstDerivativesContributions(rLeftHandSideMatrix,
							rRightHandSideVector,
							rCurrentProcessInfo);
    }

    void CalculateFirstDerivativesLHS(MatrixType& rLeftHandSideMatrix,
					        ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateFirstDerivativesLHS(rLeftHandSideMatrix,
					        rCurrentProcessInfo);
    }

    void CalculateFirstDerivativesRHS(VectorType& rRightHandSideVector,
					      ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateFirstDerivativesRHS(rRightHandSideVector,
					        rCurrentProcessInfo);
    }

    void CalculateSecondDerivativesContributions(MatrixType& rLeftHandSideMatrix,
							 VectorType& rRightHandSideVector,
							 ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateSecondDerivativesContributions(rLeftHandSideMatrix,
							    rRightHandSideVector,
							    rCurrentProcessInfo);
    }

    void CalculateSecondDerivativesLHS(MatrixType& rLeftHandSideMatrix,
					       ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateSecondDerivativesLHS(rLeftHandSideMatrix,
					        rCurrentProcessInfo);
    }

    void CalculateSecondDerivativesRHS(VectorType& rRightHandSideVector,
					       ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateSecondDerivativesRHS(rRightHandSideVector,
					        rCurrentProcessInfo);
    }

    void CalculateMassMatrix(MatrixType& rMassMatrix, ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateMassMatrix(rMassMatrix,rCurrentProcessInfo);
    }

    void CalculateDampingMatrix(MatrixType& rDampingMatrix, ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->CalculateDampingMatrix(rDampingMatrix, rCurrentProcessInfo);
    }

    void AddExplicitContribution(ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->AddExplicitContribution(rCurrentProcessInfo);
    }

    void AddExplicitContribution(const VectorType& rRHSVector,
                                const Variable<VectorType>& rRHSVariable,
                                Variable<double >& rDestinationVariable,
                                const ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->AddExplicitContribution(rRHSVector,
                                    rRHSVariable,
                                    rDestinationVariable,
                                    rCurrentProcessInfo);
    }

    void AddExplicitContribution(const VectorType& rRHSVector,
                                const Variable<VectorType>& rRHSVariable,
                                Variable<array_1d<double,3> >& rDestinationVariable,
                                const ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->AddExplicitContribution(rRHSVector,
                                rRHSVariable,
                                rDestinationVariable,
                                rCurrentProcessInfo);
    }

    void AddExplicitContribution(const MatrixType& rLHSMatrix,
                                const Variable<MatrixType>& rLHSVariable,
                                Variable<Matrix>& rDestinationVariable,
                                const ProcessInfo& rCurrentProcessInfo) override
    {
        mpPrimalElement->AddExplicitContribution(rLHSMatrix,
                                rLHSVariable,
                                rDestinationVariable,
                                rCurrentProcessInfo);
    }

    void Calculate(const Variable<double >& rVariable,
			   double& Output,
			   const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
    }

    void Calculate(const Variable< array_1d<double,3> >& rVariable,
			   array_1d<double,3>& Output,
			   const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
    }

    void Calculate(const Variable<Vector >& rVariable,
			   Vector& Output,
			   const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
    }

    void Calculate(const Variable<Matrix >& rVariable,
			   Matrix& Output,
			   const ProcessInfo& rCurrentProcessInfo) override;

    // Results calculation on integration points
    void CalculateOnIntegrationPoints(const Variable<bool>& rVariable,
					      std::vector<bool>& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
    }

    void CalculateOnIntegrationPoints(const Variable<double>& rVariable,
					      std::vector<double>& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateOnIntegrationPoints(const Variable<array_1d<double, 3 > >& rVariable,
					      std::vector< array_1d<double, 3 > >& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
    }

    void CalculateOnIntegrationPoints(const Variable<array_1d<double, 6 > >& rVariable,
					      std::vector< array_1d<double, 6 > >& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
    }

    void CalculateOnIntegrationPoints(const Variable<Vector >& rVariable,
					      std::vector< Vector >& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
    }

    void CalculateOnIntegrationPoints(const Variable<Matrix >& rVariable,
					      std::vector< Matrix >& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
    }

    void GetValueOnIntegrationPoints(const Variable<double>& rVariable,
					     std::vector<double>& rValues,
					     const ProcessInfo& rCurrentProcessInfo) override
    {
        this->CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }

    void GetValueOnIntegrationPoints(const Variable<array_1d<double, 3 > >& rVariable,
					      std::vector< array_1d<double, 3 > >& rOutput,
					      const ProcessInfo& rCurrentProcessInfo) override
    {
        this->CalculateOnIntegrationPoints(rVariable, rOutput, rCurrentProcessInfo);
    }

    void GetValueOnIntegrationPoints(const Variable<Vector>& rVariable,
					     std::vector<Vector>& rValues,
					     const ProcessInfo& rCurrentProcessInfo) override
    {
        this->CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }


    int Check(const ProcessInfo& rCurrentProcessInfo) override;

    // Sensitivity functions

    /**
     * Calculates the pseudo-load contribution of the element w.r.t. all properties which are available at the element.
     * This is done by finite differencing of the RHS of the primal element when perturbing a property value.
     * This operation is thread-save, because the property pointer is exchanged by a local one before pertubation.
     */
    void CalculateSensitivityMatrix(const Variable<double>& rDesignVariable, Matrix& rOutput,
                                            const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * Calculates the pseudo-load of the design variable SHAPE_SENSITIVITY (coordinates of nodes) contribution of the element.
     * This is done by finite differencing of the RHS of the primal element when perturbing a nodal coordinate.
     * This operation is currently NOT thread-save!
     */
    void CalculateSensitivityMatrix(const Variable<array_1d<double,3>>& rDesignVariable, Matrix& rOutput,
                                            const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * Calculates the stress-displacement derivative of the given rStressVariable.
     * For the linear case this happens by calculating the stress using a unit-displacement for each dof.
     */
    virtual void CalculateStressDisplacementDerivative(const Variable<Vector>& rStressVariable,
                                    Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);
    /**
     * Calculates the stress-design variable derivative of the given rStressVariable.
     * this is done by finite differencing of the Calculate function of the primal element
     */
    void CalculateStressDesignVariableDerivative(const Variable<double>& rDesignVariable, const Variable<Vector>& rStressVariable,
                                        Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);
    /**
     * Calculates the stress-design variable derivative of the given rStressVariable.
     * this is done by finite differencing of the Calculate function of the primal element
     */
    void CalculateStressDesignVariableDerivative(const Variable<array_1d<double,3>>& rDesignVariable,
                                            const Variable<Vector>& rStressVariable,
                                             Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);
    /**
     * Gets the pointer to the primal element.
     */
    Element::Pointer pGetPrimalElement()
    {
        return mpPrimalElement;
    }

    /*TPrimalElement& GetPrimalElement()
    {
        return mPrimalElement;
    }*/
    ///@}

    ///@name Public specialized Access - Temporary
    ///@{
    ///@}

protected:

    ///@name Protected Lyfe Cycle
    ///@{


    ///@}
    ///@name Operations
    ///@{

    template <typename TDataType>
    void CalculateAdjointFieldOnIntegrationPoints(const Variable<TDataType>& rVariable, std::vector< TDataType >& rOutput, const ProcessInfo& rCurrentProcessInfo, bool ParticularOnly =false)
    {
        KRATOS_TRY;

        KRATOS_WARNING_IF("CalculateAdjointFieldOnIntegrationPoints", OpenMPUtils::IsInParallel() != 0)
                << "The call of this non omp-parallelized function within a parallel section should be avoided for efficiency reasons!" << std::endl;

        const SizeType num_nodes = mpPrimalElement->GetGeometry().PointsNumber();
        const SizeType dimension = mpPrimalElement->GetGeometry().WorkingSpaceDimension();
        const SizeType num_dofs_per_node = (mHasRotationDofs) ?  2 * dimension : dimension;
        const SizeType num_dofs = num_nodes * num_dofs_per_node;
        Vector initial_state_variables;
        initial_state_variables.resize(num_dofs);

        Vector particular_solution = ZeroVector(num_dofs);
        if(this->Has(ADJOINT_PARTICULAR_DISPLACEMENT))
        {
            particular_solution = this->GetValue(ADJOINT_PARTICULAR_DISPLACEMENT);
            //KRATOS_WATCH(particular_solution)
        }


        // Build vector of variables containing the DOF-variables of the primal problem
        std::vector<VariableComponent<VectorComponentAdaptor<array_1d<double, 3>>>> primal_solution_variable_list;
        primal_solution_variable_list.reserve(num_dofs_per_node);
        primal_solution_variable_list.push_back(DISPLACEMENT_X);
        primal_solution_variable_list.push_back(DISPLACEMENT_Y);
        primal_solution_variable_list.push_back(DISPLACEMENT_Z);

        if(mHasRotationDofs)
        {
            primal_solution_variable_list.push_back(ROTATION_X);
            primal_solution_variable_list.push_back(ROTATION_Y);
            primal_solution_variable_list.push_back(ROTATION_Z);
        }

        // Build vector of variables containing the DOF-variables of the adjoint problem
        std::vector<VariableComponent<VectorComponentAdaptor<array_1d<double, 3>>>> adjoint_solution_variable_list;
        adjoint_solution_variable_list.reserve(num_dofs_per_node);
        adjoint_solution_variable_list.push_back(ADJOINT_DISPLACEMENT_X);
        adjoint_solution_variable_list.push_back(ADJOINT_DISPLACEMENT_Y);
        adjoint_solution_variable_list.push_back(ADJOINT_DISPLACEMENT_Z);

        if(mHasRotationDofs)
        {
            adjoint_solution_variable_list.push_back(ADJOINT_ROTATION_X);
            adjoint_solution_variable_list.push_back(ADJOINT_ROTATION_Y);
            adjoint_solution_variable_list.push_back(ADJOINT_ROTATION_Z);
        }

        for (IndexType i = 0; i < num_nodes; ++i)
        {
            const IndexType index = i * num_dofs_per_node;
            for(IndexType j = 0; j < primal_solution_variable_list.size(); ++j)
            {
                initial_state_variables[index + j] = mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(primal_solution_variable_list[j]);
                if (!ParticularOnly)
                {
                    mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(primal_solution_variable_list[j]) =
                            this->GetGeometry()[i].FastGetSolutionStepValue(adjoint_solution_variable_list[j]) +
                            particular_solution[index + j];
                }
                else
                {
                    mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(primal_solution_variable_list[j]) =
                        0.0 + particular_solution[index + j];
                }
            }
        }

        mpPrimalElement->CalculateOnIntegrationPoints(rVariable, rOutput, rCurrentProcessInfo);

        for (IndexType i = 0; i < num_nodes; ++i)
        {
            const IndexType index = i * num_dofs_per_node;
            for(IndexType j = 0; j < primal_solution_variable_list.size(); ++j)
                mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(primal_solution_variable_list[j]) = initial_state_variables[index + j];
        }

        KRATOS_CATCH("")
    }

    ///@}
    ///@name Member Variables
    ///@{

    /**
     * pointer to the primal element
     */
    Element::Pointer mpPrimalElement;
    bool mHasRotationDofs = false;
    //TPrimalElement mPrimalElement;

    ///@}

private:

    ///@name Private Classes
    ///@{
    ///@}

    ///@name Private Operations
    ///@{

    /**
     * Get the perturbation size for a scalar variable
     */
    double GetPerturbationSize(const Variable<double>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;

    /**
     * Get the perturbation size for a vector variable
     */
    double GetPerturbationSize(const Variable<array_1d<double,3>>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;

    /**
     * Get the perturbation size modification factor for a scalar variable.
     * The computed factor reflects the current value of the property (design variable).
     * Note: This approach is only based on experience.
     * This can be overwritten by derived classes.
     */
    virtual double GetPerturbationSizeModificationFactor(const Variable<double>& rVariable) const;

    /**
     * Get the perturbation size modification factor for a vector variable.
     * The computed factor reflects the size of the element.
     * Note: This approach is only based on experience.
     * This can be overwritten by derived classes.
     */
    virtual double GetPerturbationSizeModificationFactor(const Variable<array_1d<double,3>>& rDesignVariable) const;

    ///@}

    ///@name Static Member Variables
    ///@{
    ///@}

    ///@name Member Variables
    ///@{
    ///@}

    ///@name Serialization
    ///@{

    friend class Serializer;

    void save(Serializer& rSerializer) const override;

    void load(Serializer& rSerializer) override;

    ///@}

    ///@name Private  Access
    ///@{
    ///@}

    ///@name Private Inquiry
    ///@{
    ///@}

    ///@name Un accessible methods
    ///@{
    ///@}

};

}
#endif // ADJOINT_FINITE_DIFFERENCE_BASE_ELEMENT_H_INCLUDED
