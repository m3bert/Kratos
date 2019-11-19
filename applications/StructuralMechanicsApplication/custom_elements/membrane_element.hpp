// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Klaus B. Sautter
//

#if !defined(MEMBRANE_ELEMENT_3D_H_INCLUDED )
#define  MEMBRANE_ELEMENT_3D_H_INCLUDED

// System includes

// External includes

// Project includes
#include "includes/element.h"
#include "custom_utilities/structural_mechanics_math_utilities.hpp"

namespace Kratos
{

  class MembraneElement
    : public Element
  {
  public:

    // Counted pointer of MembraneElement
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(MembraneElement);

    // Constructor using an array of nodes
    MembraneElement(IndexType NewId, GeometryType::Pointer pGeometry);

    // Constructor using an array of nodes with properties
    MembraneElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

    // Destructor
    ~MembraneElement() override;


    // Name Operations

    /**
     * @brief Creates a new element
     * @param NewId The Id of the new created element
     * @param pGeom The pointer to the geometry of the element
     * @param pProperties The pointer to property
     * @return The pointer to the created element
     */
    Element::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties
        ) const override;

    /**
     * @brief Creates a new element
     * @param NewId The Id of the new created element
     * @param ThisNodes The array containing nodes
     * @param pProperties The pointer to property
     * @return The pointer to the created element
     */
    Element::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        PropertiesType::Pointer pProperties
        ) const override;

    void EquationIdVector(
      EquationIdVectorType& rResult,
      ProcessInfo& rCurrentProcessInfo) override;

    void GetDofList(
      DofsVectorType& ElementalDofList,
      ProcessInfo& rCurrentProcessInfo) override;

    void Initialize() override;

    void CalculateLeftHandSide(
    MatrixType& rLeftHandSideMatrix,
    ProcessInfo& rCurrentProcessInfo) override;

    void CalculateRightHandSide(
      VectorType& rRightHandSideVector,
      ProcessInfo& rCurrentProcessInfo) override;

    void CalculateLocalSystem(
      MatrixType& rLeftHandSideMatrix,
      VectorType& rRightHandSideVector,
      ProcessInfo& rCurrentProcessInfo) override;

    void CovariantBaseVectors(array_1d<Vector,2>& rBaseVectors,
     const Matrix& rShapeFunctionGradientValues,const std::string Configuration);

    void CovariantMetric(Matrix& rMetric,const array_1d<Vector,2>& rBaseVectorCovariant);

    void ContraVariantBaseVectors(array_1d<Vector,2>& rBaseVectors,const Matrix& rContraVariantMetric,
      const array_1d<Vector,2> rCovariantBaseVectors);

    void ContravariantMetric(Matrix& rMetric,const Matrix& rCovariantMetric);

    void DeriveCurrentCovariantBaseVectors(array_1d<Vector,2>& rBaseVectors,
     const Matrix& rShapeFunctionGradientValues, const SizeType DofR);

    void Derivative2CurrentCovariantMetric(Matrix& rMetric,
      const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const SizeType DofS);

    void JacobiDeterminante(double& rDetJacobi, const array_1d<Vector,2>& rReferenceBaseVectors);

    void Derivative2StrainGreenLagrange(Vector& rStrain,
      const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const SizeType DofS,
      const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void DerivativeStrainGreenLagrange(Vector& rStrain, const Matrix& rShapeFunctionGradientValues, const SizeType DofR,
      const array_1d<Vector,2> rCurrentCovariantBaseVectors, const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void StrainGreenLagrange(Vector& rStrain, const Matrix& rReferenceCoVariantMetric,const Matrix& rCurrentCoVariantMetric,
      const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void MaterialTangentModulus(Matrix& rTangentModulus,const Matrix& rReferenceContraVariantMetric);

    void StressPk2(Vector& rStress,
      const Matrix& rReferenceContraVariantMetric,const Matrix& rReferenceCoVariantMetric,const Matrix& rCurrentCoVariantMetric,
      const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void AddPreStressPk2(Vector& rStress);

    void VoigtNotation(const Matrix& rMetric, Vector& rOutputVector, const std::string StrainStressCheck);

    void DerivativeCurrentCovariantMetric(Matrix& rMetric,
      const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const array_1d<Vector,2> rCurrentCovariantBaseVectors);

    void InternalForces(Vector& rInternalForces,const IntegrationMethod ThisMethod);

    void TotalStiffnessMatrix(Matrix& rStiffnessMatrix,const IntegrationMethod ThisMethod);

    void InitialStressStiffnessMatrixEntryIJ(double& rEntryIJ,
      const Vector& rStressVector,const double& rDetJ, const double& rWeight,
      const SizeType& rPositionI, const SizeType& rPositionJ, const Matrix& rShapeFunctionGradientValues,
      const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void MaterialStiffnessMatrixEntryIJ(double& rEntryIJ,
      const Matrix& rMaterialTangentModulus,const double& rDetJ, const double& rWeight,
      const SizeType& rPositionI, const SizeType& rPositionJ, const Matrix& rShapeFunctionGradientValues,
      const array_1d<Vector,2> rCurrentCovariantBaseVectors, const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void TransformStrains(Matrix& rStrainsMatrix,const Matrix& rReferenceStrainsMatrix,
      const array_1d<Vector,2> rLocalContraVariantBaseVectorsReference);

    void GetValuesVector(
      Vector& rValues,
      int Step = 0) override;

    void GetFirstDerivativesVector(
      Vector& rValues,
      int Step = 0) override;

    void GetSecondDerivativesVector(
      Vector& rValues,
      int Step = 0) override;

    int Check(const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateOnIntegrationPoints(
      const Variable<array_1d<double, 3>>& rVariable,
      std::vector<array_1d<double, 3>>& rOutput,
      const ProcessInfo& rCurrentProcessInfo) override;

    void GetValueOnIntegrationPoints(
      const Variable<array_1d<double, 3>>& rVariable,
      std::vector<array_1d<double, 3>>& rOutput,
      const ProcessInfo& rCurrentProcessInfo) override;

    void TransformBaseVectors(array_1d<Vector,2>& rBaseVectors,
     const array_1d<Vector,2>& rLocalBaseVectors,const std::string Configuration);

protected:
  ConstitutiveLaw::Pointer mpConstitutiveLaw = nullptr;

private:

  ///@}
  ///@name Serialization
  ///@{

  friend class Serializer;

  // A private default constructor necessary for serialization
  MembraneElement() {}

  void save(Serializer& rSerializer) const override;

  void load(Serializer& rSerializer) override;

  ///@}

  };	// class MembraneElement.

}	// namespace Kratos.

#endif // KRATOS_MEMBRANE_ELEMENT_3D_H_INCLUDED  defined
