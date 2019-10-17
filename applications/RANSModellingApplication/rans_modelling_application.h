//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Suneth Warnakulasuriya (https://github.com/sunethwarna)
//

#if !defined(KRATOS_RANS_MODELLING_APPLICATION_H_INCLUDED)
#define KRATOS_RANS_MODELLING_APPLICATION_H_INCLUDED

// System includes

// External includes

// Project includes
#include "includes/kratos_application.h"

// Element includes
#include "custom_elements/evm_k_epsilon/rans_evm_epsilon_element.h"
#include "custom_elements/evm_k_epsilon/rans_evm_k_element.h"
#include "custom_elements/evm_k_epsilon/rans_evm_low_re_epsilon_element.h"
#include "custom_elements/evm_k_epsilon/rans_evm_low_re_k_element.h"

// Condition includes
#include "custom_conditions/evm_k_epsilon/rans_evm_epsilon_wall_condition.h"
#include "custom_conditions/evm_k_epsilon/rans_evm_vms_monolithic_wall_condition.h"

// Adjoint element includes
#include "custom_elements/evm_k_epsilon/rans_evm_epsilon_adjoint.h"
#include "custom_elements/evm_k_epsilon/rans_evm_k_adjoint.h"
#include "custom_elements/evm_k_epsilon/rans_evm_k_epsilon_vms_adjoint.h"
#include "custom_elements/evm_k_epsilon/rans_evm_monolithic_k_epsilon_vms_adjoint.h"

// Adjoint condition includes
#include "custom_conditions/evm_k_epsilon/rans_evm_epsilon_adjoint_wall_condition.h"
#include "custom_conditions/evm_k_epsilon/rans_evm_vms_monolithic_adjoint_wall_condition.h"

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

/// Short class definition.
/** Detail class definition.
 */
class KratosRANSModellingApplication : public KratosApplication
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of KratosRANSModellingApplication
    KRATOS_CLASS_POINTER_DEFINITION(KratosRANSModellingApplication);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    KratosRANSModellingApplication();

    /// Destructor.
    ~KratosRANSModellingApplication() override
    {
    }

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    void Register() override;

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
        return "KratosRANSModellingApplication";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << Info();
        PrintData(rOStream);
    }

    ///// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
        KRATOS_WATCH("in my application");
        KRATOS_WATCH(KratosComponents<VariableData>::GetComponents().size());

        rOStream << "Variables:" << std::endl;
        KratosComponents<VariableData>().PrintData(rOStream);
        rOStream << std::endl;
        rOStream << "Elements:" << std::endl;
        KratosComponents<Element>().PrintData(rOStream);
        rOStream << std::endl;
        rOStream << "Conditions:" << std::endl;
        KratosComponents<Condition>().PrintData(rOStream);
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

    // static const ApplicationCondition  msApplicationCondition;

    ///@}
    ///@name Member Variables
    ///@{

    /// k-epsilon turbulence model elements
    const RansEvmLowReKElement<2, 3> mRansEvmLowReK2D;
    const RansEvmLowReKElement<3, 4> mRansEvmLowReK3D;

    const RansEvmLowReEpsilonElement<2, 3> mRansEvmLowReEpsilon2D;
    const RansEvmLowReEpsilonElement<3, 4> mRansEvmLowReEpsilon3D;

    const RansEvmKElement<2, 3> mRansEvmK2D;
    const RansEvmKElement<3, 4> mRansEvmK3D;

    const RansEvmEpsilonElement<2, 3> mRansEvmEpsilon2D;
    const RansEvmEpsilonElement<3, 4> mRansEvmEpsilon3D;

    /// k-epsilon turbulence model conditions
    const RansEvmEpsilonWallCondition<2> mRansEvmEpsilonWallCondition2D2N;
    const RansEvmEpsilonWallCondition<3> mRansEvmEpsilonWallCondition3D3N;

    const RansEvmVmsMonolithicWallCondition<2> mRansEvmVmsMonolithicWallCondition2D2N;
    const RansEvmVmsMonolithicWallCondition<3> mRansEvmVmsMonolithicWallCondition3D3N;

    // k-epsilon adjoint elements
    const RansEvmEpsilonAdjoint<2, 3> mRansEvmEpsilonAdjoint2D3N;
    const RansEvmEpsilonAdjoint<3, 4> mRansEvmEpsilonAdjoint3D4N;

    const RansEvmKAdjoint<2, 3> mRansEvmKAdjoint2D3N;
    const RansEvmKAdjoint<3, 4> mRansEvmKAdjoint3D4N;

    const RansEvmKEpsilonVMSAdjoint<2> mRansEvmKEpsilonVMSAdjoint2D3N;
    const RansEvmKEpsilonVMSAdjoint<3> mRansEvmKEpsilonVMSAdjoint3D4N;

    const RansEvmMonolithicKEpsilonVMSAdjoint<2> mRansEvmMonolithicKEpsilonVMSAdjoint2D3N;
    const RansEvmMonolithicKEpsilonVMSAdjoint<3> mRansEvmMonolithicKEpsilonVMSAdjoint3D4N;

    // k-epsilon adjoint conditions
    const RansEvmEpsilonAdjointWallCondition<2> mRansEvmEpsilonAdjointWallCondition2D2N;
    const RansEvmEpsilonAdjointWallCondition<3> mRansEvmEpsilonAdjointWallCondition3D3N;

    const RansEvmVmsMonolithicAdjointWallCondition<2> mRansEvmVmsMonolithicAdjointWallCondition2D2N;
    const RansEvmVmsMonolithicAdjointWallCondition<3> mRansEvmVmsMonolithicAdjointWallCondition3D3N;
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

    /// Assignment operator.
    KratosRANSModellingApplication& operator=(KratosRANSModellingApplication const& rOther);

    /// Copy constructor.
    KratosRANSModellingApplication(KratosRANSModellingApplication const& rOther);

    ///@}

}; // Class KratosRANSModellingApplication

///@}

///@name Type Definitions
///@{

///@}
///@name Input and output
///@{

///@}

} // namespace Kratos.

#endif // KRATOS_RANS_MODELLING_APPLICATION_H_INCLUDED  defined