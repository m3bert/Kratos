//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Suneth Warnakulasuriya
//

#ifndef KRATOS_RANS_EVM_VMS_MONOLITHIC_WALL_CONDITION_H
#define KRATOS_RANS_EVM_VMS_MONOLITHIC_WALL_CONDITION_H

// System includes
#include <iostream>
#include <string>

// External includes

// Project includes
#include "includes/checks.h"
#include "includes/condition.h"
#include "includes/define.h"
#include "includes/process_info.h"
#include "includes/serializer.h"

// Application includes
#include "custom_conditions/monolithic_wall_condition.h"
#include "custom_utilities/rans_calculation_utilities.h"
#include "includes/cfd_variables.h"
#include "rans_modelling_application_variables.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

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

/**
 * @brief
 *
 * @tparam TDim       Dimensionality of the condition (2D or 3D)
 * @tparam TNumNodes  Number of nodes in the condition
 */
template <unsigned int TDim, unsigned int TNumNodes = TDim>
class RansEvmVmsMonolithicWallCondition : public MonolithicWallCondition<TDim, TNumNodes>
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of RansEvmVmsMonolithicWallCondition
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(RansEvmVmsMonolithicWallCondition);

    using BaseType = MonolithicWallCondition<TDim, TNumNodes>;
    using NodeType = Node<3>;
    using PropertiesType = Properties;
    using GeometryType = Geometry<NodeType>;
    using NodesArrayType = Geometry<NodeType>::PointsArrayType;
    using VectorType = Vector;
    using MatrixType = Matrix;
    using IndexType = std::size_t;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    /** Admits an Id as a parameter.
      @param NewId Index for the new condition
      */
    explicit RansEvmVmsMonolithicWallCondition(IndexType NewId = 0)
        : BaseType(NewId)
    {
    }

    /// Constructor using an array of nodes
    /**
     @param NewId Index of the new condition
     @param ThisNodes An array containing the nodes of the new condition
     */
    RansEvmVmsMonolithicWallCondition(IndexType NewId, const NodesArrayType& ThisNodes)
        : BaseType(NewId, ThisNodes)
    {
    }

    /// Constructor using Geometry
    /**
     @param NewId Index of the new condition
     @param pGeometry Pointer to a geometry object
     */
    RansEvmVmsMonolithicWallCondition(IndexType NewId, GeometryType::Pointer pGeometry)
        : BaseType(NewId, pGeometry)
    {
    }

    /// Constructor using Properties
    /**
     @param NewId Index of the new element
     @param pGeometry Pointer to a geometry object
     @param pProperties Pointer to the element's properties
     */
    RansEvmVmsMonolithicWallCondition(IndexType NewId,
                                      GeometryType::Pointer pGeometry,
                                      PropertiesType::Pointer pProperties)
        : BaseType(NewId, pGeometry, pProperties)
    {
    }

    /// Copy constructor.
    RansEvmVmsMonolithicWallCondition(RansEvmVmsMonolithicWallCondition const& rOther)
        : BaseType(rOther)
    {
    }

    /// Destructor.
    ~RansEvmVmsMonolithicWallCondition() override
    {
    }

    ///@}
    ///@name Operators
    ///@{

    /// Assignment operator
    RansEvmVmsMonolithicWallCondition& operator=(RansEvmVmsMonolithicWallCondition const& rOther)
    {
        Condition::operator=(rOther);

        return *this;
    }

    ///@}
    ///@name Operations
    ///@{

    /// Create a new RansEvmVmsMonolithicWallCondition object.
    /**
      @param NewId Index of the new condition
      @param ThisNodes An array containing the nodes of the new condition
      @param pProperties Pointer to the element's properties
      */
    Condition::Pointer Create(IndexType NewId,
                              NodesArrayType const& ThisNodes,
                              PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive<RansEvmVmsMonolithicWallCondition>(
            NewId, this->GetGeometry().Create(ThisNodes), pProperties);
    }

    Condition::Pointer Create(IndexType NewId,
                              GeometryType::Pointer pGeom,
                              PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive<RansEvmVmsMonolithicWallCondition>(
            NewId, pGeom, pProperties);
    }

    /**
     * Clones the selected element variables, creating a new one
     * @param NewId the ID of the new element
     * @param ThisNodes the nodes of the new element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */

    Condition::Pointer Clone(IndexType NewId, NodesArrayType const& rThisNodes) const override
    {
        Condition::Pointer pNewCondition = Create(
            NewId, this->GetGeometry().Create(rThisNodes), this->pGetProperties());

        pNewCondition->SetData(this->GetData());
        pNewCondition->SetFlags(this->GetFlags());

        return pNewCondition;
    }

    int Check(const ProcessInfo& rCurrentProcessInfo) override
    {
        KRATOS_TRY;

        int Check = BaseType::Check(rCurrentProcessInfo);

        if (Check != 0)
        {
            return Check;
        }

        KRATOS_CHECK_VARIABLE_KEY(DISTANCE);
        KRATOS_CHECK_VARIABLE_KEY(TURBULENCE_RANS_C_MU);
        KRATOS_CHECK_VARIABLE_KEY(WALL_VON_KARMAN);
        KRATOS_CHECK_VARIABLE_KEY(WALL_SMOOTHNESS_BETA);
        KRATOS_CHECK_VARIABLE_KEY(TURBULENT_KINETIC_ENERGY);
        KRATOS_CHECK_VARIABLE_KEY(RANS_Y_PLUS);
        KRATOS_CHECK_VARIABLE_KEY(DENSITY);
        KRATOS_CHECK_VARIABLE_KEY(IS_CO_SOLVING_PROCESS_ACTIVE);
        KRATOS_CHECK_VARIABLE_KEY(VELOCITY);

        const GeometryType& r_geometry = this->GetGeometry();

        for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
        {
            const NodeType& r_node = r_geometry[i_node];

            KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(DISTANCE, r_node);
            KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(TURBULENT_KINETIC_ENERGY, r_node);
            KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(RANS_Y_PLUS, r_node);
            KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(DENSITY, r_node);
            KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(VELOCITY, r_node);
        }

        return 0;

        KRATOS_CATCH("");
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

    /// Turn back information as a string.
    std::string Info() const override
    {
        std::stringstream buffer;
        buffer << "RansEvmVmsMonolithicWallCondition" << TDim << "D";
        return buffer.str();
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "RansEvmVmsMonolithicWallCondition";
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
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

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Applies wall law based on logarithmic wall law
     *
     * This applies wall law based on the following equation for the period of
     * time where RANS model is not activated for $y+ \leq 11.06$
     * \[
     *     u^+ = y^+
     * \]
     * for $y^+ > 11.06$
     * \[
     *     u^+ = \frac{1}{\kappa}ln\left(y^+\right\) + \beta
     * \]
     *
     * @param rLocalMatrix         Left hand side matrix
     * @param rLocalVector         Right hand side vector
     * @param rCurrentProcessInfo  Current process info from model part
     */
    virtual void ApplyLogarithmicWallLaw(MatrixType& rLocalMatrix,
                                         VectorType& rLocalVector,
                                         ProcessInfo& rCurrentProcessInfo)
    {
        GeometryType& rGeometry = this->GetGeometry();
        const size_t BlockSize = TDim + 1;
        const double NodalFactor = 1.0 / double(TDim);

        double area = NodalFactor * rGeometry.DomainSize();
        // DomainSize() is the way to ask the geometry's length/area/volume (whatever is relevant for its dimension) without asking for the number of spatial dimensions first

        for (size_t itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
        {
            const NodeType& rConstNode = rGeometry[itNode];
            const double y = rConstNode.FastGetSolutionStepValue(DISTANCE); // wall distance to use in stress calculation
            if (y > 0.0 && rConstNode.Is(SLIP))
            {
                array_1d<double, 3> Vel =
                    rGeometry[itNode].FastGetSolutionStepValue(VELOCITY);
                const array_1d<double, 3>& VelMesh =
                    rGeometry[itNode].FastGetSolutionStepValue(MESH_VELOCITY);
                Vel -= VelMesh;
                const double Ikappa = 1.0 / 0.41; // inverse of Von Karman's kappa
                const double B = 5.2;
                const double limit_yplus = 10.9931899; // limit between linear and log regions

                const double rho = rGeometry[itNode].FastGetSolutionStepValue(DENSITY);
                const double nu = rGeometry[itNode].FastGetSolutionStepValue(VISCOSITY);

                double wall_vel = 0.0;
                for (size_t d = 0; d < TDim; d++)
                {
                    wall_vel += Vel[d] * Vel[d];
                }
                wall_vel = sqrt(wall_vel);

                if (wall_vel > 1e-12) // do not bother if velocity is zero
                {
                    // linear region
                    double utau = sqrt(wall_vel * nu / y);
                    double yplus = y * utau / nu;

                    // log region
                    if (yplus > limit_yplus)
                    {
                        // wall_vel / utau = 1/kappa * log(yplus) + B
                        // this requires solving a nonlinear problem:
                        // f(utau) = utau*(1/kappa * log(y*utau/nu) + B) - wall_vel = 0
                        // note that f'(utau) = 1/kappa * log(y*utau/nu) + B + 1/kappa

                        IndexType iter = 0;
                        double dx = 1e10;
                        const double tol = 1e-6;
                        double uplus = Ikappa * log(yplus) + B;

                        while (iter < 100 && fabs(dx) > tol * utau)
                        {
                            // Newton-Raphson iteration
                            double f = utau * uplus - wall_vel;
                            double df = uplus + Ikappa;
                            dx = f / df;

                            // Update variables
                            utau -= dx;
                            yplus = y * utau / nu;
                            uplus = Ikappa * log(yplus) + B;
                            ++iter;
                        }
                        if (iter == 100)
                        {
                            std::cout
                                << "Warning: wall condition Newton-Raphson did "
                                   "not converge. Residual is "
                                << dx << std::endl;
                        }
                    }
                    const double Tmp = area * utau * utau * rho / wall_vel;
                    for (size_t d = 0; d < TDim; d++)
                    {
                        size_t k = itNode * BlockSize + d;
                        rLocalVector[k] -= Vel[d] * Tmp;
                        rLocalMatrix(k, k) += Tmp;
                    }
                }
            }
        }
    }

    /**
     * @brief Applies rans based wall law
     *
     * This method calculate left hand side matrix and right hand side vector for following equation
     *
     * \[
     *      u_\tau = max\left(C_\mu^0.25 \sqrt{k}, \frac{||\underline{u}||}{\frac{1}{\kappa}ln(y^+)+\beta}\right)
     * \]
     *
     * integration point value = \rho \frac{u_\tau^2}{||\underline{u}||}\underline{u}
     *
     * @param rLocalMatrix         Left hand side matrix
     * @param rLocalVector         Right hand side vector
     * @param rCurrentProcessInfo  Current process info from model part
     */
    virtual void ApplyRansBasedWallLaw(MatrixType& rLocalMatrix,
                                       VectorType& rLocalVector,
                                       ProcessInfo& rCurrentProcessInfo)

    {
        RansCalculationUtilities rans_calculation_utilities;

        GeometryType& r_geometry = this->GetGeometry();

        const GeometryType::IntegrationPointsArrayType& integration_points =
            r_geometry.IntegrationPoints(GeometryData::GI_GAUSS_2);
        const std::size_t number_of_gauss_points = integration_points.size();
        MatrixType shape_functions =
            r_geometry.ShapeFunctionsValues(GeometryData::GI_GAUSS_2);

        array_1d<double, 3> normal;
        this->CalculateNormal(normal); // this already contains the area
        double A = norm_2(normal);

        // CAUTION: "Jacobian" is 2.0*A for triangles but 0.5*A for lines
        double J = (TDim == 2) ? 0.5 * A : 2.0 * A;

        const size_t block_size = TDim + 1;

        const double c_mu_25 = std::pow(rCurrentProcessInfo[TURBULENCE_RANS_C_MU], 0.25);
        const double inv_von_karman = 1.0 / rCurrentProcessInfo[WALL_VON_KARMAN];
        const double beta = rCurrentProcessInfo[WALL_SMOOTHNESS_BETA];

        const double eps = std::numeric_limits<double>::epsilon();

        for (size_t g = 0; g < number_of_gauss_points; ++g)
        {
            const Vector& gauss_shape_functions = row(shape_functions, g);
            const double weight = J * integration_points[g].Weight();

            const array_1d<double, 3>& r_wall_velocity =
                rans_calculation_utilities.EvaluateInPoint(
                    r_geometry, VELOCITY, gauss_shape_functions);
            const double wall_velocity_magnitude = norm_2(r_wall_velocity);

            const double tke = rans_calculation_utilities.EvaluateInPoint(
                r_geometry, TURBULENT_KINETIC_ENERGY, gauss_shape_functions);
            const double y_plus = rans_calculation_utilities.EvaluateInPoint(
                r_geometry, RANS_Y_PLUS, gauss_shape_functions);
            const double rho = rans_calculation_utilities.EvaluateInPoint(
                r_geometry, DENSITY, gauss_shape_functions);

            if (wall_velocity_magnitude > eps)
            {
                const double u_tau = std::max(
                    c_mu_25 * std::sqrt(std::max(tke, 0.0)),
                    wall_velocity_magnitude / (inv_von_karman * std::log(y_plus) + beta));
                const double value = rho * std::pow(u_tau, 2) * weight / wall_velocity_magnitude;

                for (size_t a = 0; a < r_geometry.PointsNumber(); ++a)
                {
                    for (size_t dim = 0; dim < TDim; ++dim)
                    {
                        for (size_t b = 0; b < r_geometry.PointsNumber(); ++b)
                        {
                            rLocalMatrix(a * block_size + dim, b * block_size + dim) +=
                                gauss_shape_functions[a] * gauss_shape_functions[b] * value;
                        }
                        rLocalVector[a * block_size + dim] -=
                            gauss_shape_functions[a] * value * r_wall_velocity[dim];
                    }
                }
            }
        }
    }

    void ApplyWallLaw(MatrixType& rLocalMatrix,
                      VectorType& rLocalVector,
                      ProcessInfo& rCurrentProcessInfo) override
    {
        if (!this->Is(SLIP))
            return;

        if (rCurrentProcessInfo[IS_CO_SOLVING_PROCESS_ACTIVE])
        {
            this->ApplyRansBasedWallLaw(rLocalMatrix, rLocalVector, rCurrentProcessInfo);
        }
        else
        {
            this->ApplyLogarithmicWallLaw(rLocalMatrix, rLocalVector, rCurrentProcessInfo);
        }
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
    ///@name Serialization
    ///@{

    friend class Serializer;

    void save(Serializer& rSerializer) const override
    {
        KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Condition);
    }

    void load(Serializer& rSerializer) override
    {
        KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Condition);
    }

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

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

}; // Class RansEvmVmsMonolithicWallCondition

///@}

///@name Type Definitions
///@{

///@}
///@name Input and output
///@{

/// input stream function
template <unsigned int TDim, unsigned int TNumNodes>
inline std::istream& operator>>(std::istream& rIStream,
                                RansEvmVmsMonolithicWallCondition<TDim, TNumNodes>& rThis)
{
    return rIStream;
}

/// output stream function
template <unsigned int TDim, unsigned int TNumNodes>
inline std::ostream& operator<<(std::ostream& rOStream,
                                const RansEvmVmsMonolithicWallCondition<TDim, TNumNodes>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

///@}

///@} addtogroup block

} // namespace Kratos.

#endif // KRATOS_RANS_EVM_VMS_MONOLITHIC_WALL_CONDITION_H