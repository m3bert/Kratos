//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Andreas Apostolatos
//                   Tobias Teschemacher
//

// System includes
#include <limits>

// External includes

// Project includes
#include "testing/testing.h"
#include "geometries/nurbs_curve_geometry.h"
#include "geometries/nurbs_shape_function_utilities/projection_nurbs_geometry_utilities.h"

#include "tests/cpp_tests/geometries/test_geometry.h"

namespace Kratos {
namespace Testing {

typedef Node<3> NodeType;

// /// Factory functions
//namespace {
    NurbsCurveGeometry<2, Point> GenerateReferenceCurveForProjection2d()
    {
        NurbsCurveGeometry<2, Point>::PointsArrayType points;

        points.push_back(Point::Pointer(new Point(-9, -2, 0)));
        points.push_back(Point::Pointer(new Point(-5, -3, 0)));
        points.push_back(Point::Pointer(new Point(-3, -5, 0)));
        points.push_back(Point::Pointer(new Point(-2, -4, 0)));
        points.push_back(Point::Pointer(new Point(2, 4, 0)));
        points.push_back(Point::Pointer(new Point(5, 3, 0)));
        points.push_back(Point::Pointer(new Point(9, -2, 0)));

        Vector knot_vector = ZeroVector(9);
        knot_vector[0] = -1.0;
        knot_vector[1] = -1.0;
        knot_vector[2] = -1.0;
        knot_vector[3] = -0.5;
        knot_vector[4] = 0.0;
        knot_vector[5] = 0.75;
        knot_vector[6] = 1.0;
        knot_vector[7] = 1.0;
        knot_vector[8] = 1.0;

        int p = 3;

        Vector weights = ZeroVector(7);
        weights[0] = 1.0;
        weights[1] = 2.0;
        weights[2] = 3.4;
        weights[3] = 1.0;
        weights[4] = 5.7;
        weights[5] = 4.3;
        weights[6] = 1.0;

        auto curve = NurbsCurveGeometry<2, Point>(points, p, knot_vector, weights);

        return curve;
    }

    ///// Tests
    KRATOS_TEST_CASE_IN_SUITE(NurbsGeometryProjection2d, KratosCoreNurbsGeometriesFastSuite) {
        auto curve = GenerateReferenceCurveForProjection2d();

        array_1d<double, 3> point(0.0);
        point[0] = -2.0;
        point[1] = 1.0;
        point[2] = 0.0;

        array_1d<double, 3> result(0.0);

        // Try projection with initial guess at u = 0.0
        const double initial_guess_parameter1 = 0.0;

        bool isConverged1;

        curve.PointLocalCoordinates(isConverged1, initial_guess_parameter1, result, point);

        KRATOS_CHECK_EQUAL(isConverged1, true);

        KRATOS_CHECK_NEAR(result[0], 0.099395977882318, TOLERANCE);

        std::vector<double> projected_point1 = {-0.129744540301921, -0.044240249340891, 0.0};

        KRATOS_CHECK_VECTOR_NEAR(point, projected_point1, TOLERANCE);

        // Try projection with initial guess at u = -1.0
        const double initial_guess_parameter2 = curve.DomainInterval().MinParameter();
        point[0] = -2.0;
        point[1] = 1.0;
        point[2] = 0.0;

        bool isConverged2;

        curve.PointLocalCoordinates(isConverged2, initial_guess_parameter2,result, point);

        KRATOS_CHECK_EQUAL(isConverged2, true);

        KRATOS_CHECK_NEAR(result[0], -0.788227217287371, TOLERANCE);

        std::vector<double> projected_point2 = {-4.694701201131293, -3.571229085898834, 0.0};

        KRATOS_CHECK_VECTOR_NEAR(point, projected_point2, TOLERANCE);
    }

} // namespace Testing.
} // namespace Kratos.
