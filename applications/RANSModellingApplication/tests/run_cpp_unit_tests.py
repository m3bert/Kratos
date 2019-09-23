from KratosMultiphysics import *
from KratosMultiphysics.RANSModellingApplication import *

def run():
    Tester.SetVerbosity(Tester.Verbosity.FAILED_TESTS_OUTPUTS)
    Tester.RunTestSuite("KratosRANSTestSuite")

if __name__ == '__main__':
    run()