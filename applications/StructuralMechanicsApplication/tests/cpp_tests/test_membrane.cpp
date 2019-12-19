// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Inigo López
//

// System includes

// External includes

// Project includes
#include "containers/model.h"
#include "testing/testing.h"

#include "custom_elements/membrane_element.hpp"

namespace Kratos
{
namespace Testing
{
    KRATOS_TEST_CASE_IN_SUITE(MembraneElement3D3N, KratosStructuralMechanicsFastSuite)
    {
        Model current_model;
        auto &r_model_part = current_model.CreateModelPart("ModelPart",1);

        r_model_part.AddNodalSolutionStepVariable(DISPLACEMENT);

        // Set the element properties
        auto p_elem_prop = r_model_part.CreateNewProperties(0);
        p_elem_prop->SetValue(YOUNG_MODULUS, 2.0e+06);
        p_elem_prop->SetValue(POISSON_RATIO, 0.3);
        p_elem_prop->SetValue(THICKNESS, 0.01);
        const auto &r_clone_cl = KratosComponents<ConstitutiveLaw>::Get("LinearElasticPlaneStress2DLaw");
        p_elem_prop->SetValue(CONSTITUTIVE_LAW, r_clone_cl.Clone());

        // Create the test element
        auto p_node_1 = r_model_part.CreateNewNode(1, 0.0 , 0.0 , 0.0);
        auto p_node_3 = r_model_part.CreateNewNode(2, 1.0 , 0.0 , 0.0);
        auto p_node_2 = r_model_part.CreateNewNode(3, 0.0 , 1.0 , 0.0);
        std::vector<ModelPart::IndexType> element_nodes {1,2,3};
        auto p_element = r_model_part.CreateNewElement("MembraneElement3D3N", 1, element_nodes, p_elem_prop);

        // Set a fake displacement and volumetric strain field to compute the residual
        array_1d<double, 3> aux_disp = ZeroVector(3);
        p_node_1->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;
        p_node_2->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;
        aux_disp[1] = 0.1;
        p_node_3->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;

        // Compute RHS and LHS
        Vector RHS = ZeroVector(9);
        Matrix LHS = ZeroMatrix(9,9);

        p_element->Initialize(); // Initialize the element to initialize the constitutive law
        p_element->CalculateLocalSystem(LHS, RHS, r_model_part.GetProcessInfo());

        // // Check RHS and LHS results
        // const double tolerance = 1.0e-5;
        // const std::vector<double> expected_RHS({51153.8,51153.8,-9388.35,-12692.3,-38461.5,18114.3,-38461.5,-12692.3,3966.35});
        // const std::vector<double> expected_LHS_row_0({778846, 9615.38, -317308, -394231, -384615, -317308, -384615, 375000, -317308});
        // KRATOS_CHECK_VECTOR_RELATIVE_NEAR(RHS, expected_RHS, tolerance)
        // KRATOS_CHECK_VECTOR_RELATIVE_NEAR(row(LHS,0), expected_LHS_row_0, tolerance)
    }

    KRATOS_TEST_CASE_IN_SUITE(MembraneElement3D4N, KratosStructuralMechanicsFastSuite)
    {
        Model current_model;
        auto &r_model_part = current_model.CreateModelPart("ModelPart",1);

        r_model_part.AddNodalSolutionStepVariable(DISPLACEMENT);

        // Set the element properties
        auto p_elem_prop = r_model_part.CreateNewProperties(0);
        p_elem_prop->SetValue(YOUNG_MODULUS, 2.0e+06);
        p_elem_prop->SetValue(POISSON_RATIO, 0.3);
        p_elem_prop->SetValue(THICKNESS, 0.01);
        const auto &r_clone_cl = KratosComponents<ConstitutiveLaw>::Get("LinearElasticPlaneStress2DLaw");
        p_elem_prop->SetValue(CONSTITUTIVE_LAW, r_clone_cl.Clone());

        // Create the test element
        auto p_node_1 = r_model_part.CreateNewNode(1, 0.0 , 0.0 , 0.0);
        auto p_node_3 = r_model_part.CreateNewNode(2, 1.0 , 0.0 , 0.0);
        auto p_node_2 = r_model_part.CreateNewNode(3, 1.0 , 1.0 , 0.0);
        auto p_node_4 = r_model_part.CreateNewNode(4, 1.0 , 1.0 , 0.0);
        std::vector<ModelPart::IndexType> element_nodes {1,2,3,4};
        auto p_element = r_model_part.CreateNewElement("MembraneElement3D4N", 1, element_nodes, p_elem_prop);

        // Set a fake displacement and volumetric strain field to compute the residual
        array_1d<double, 3> aux_disp = ZeroVector(3);
        p_node_1->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;
        p_node_2->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;
        p_node_4->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;
        aux_disp[1] = 0.1;
        p_node_3->FastGetSolutionStepValue(DISPLACEMENT) = aux_disp;

        // Compute RHS and LHS
        Vector RHS = ZeroVector(12);
        Matrix LHS = ZeroMatrix(12,12);

        p_element->Initialize(); // Initialize the element to initialize the constitutive law
        p_element->CalculateLocalSystem(LHS, RHS, r_model_part.GetProcessInfo());

        // // Check RHS and LHS results
        // const double tolerance = 1.0e-5;
        // const std::vector<double> expected_RHS({51153.8,51153.8,-9388.35,-12692.3,-38461.5,18114.3,-38461.5,-12692.3,3966.35});
        // const std::vector<double> expected_LHS_row_0({778846, 9615.38, -317308, -394231, -384615, -317308, -384615, 375000, -317308});
        // KRATOS_CHECK_VECTOR_RELATIVE_NEAR(RHS, expected_RHS, tolerance)
        // KRATOS_CHECK_VECTOR_RELATIVE_NEAR(row(LHS,0), expected_LHS_row_0, tolerance)
    }
}
}