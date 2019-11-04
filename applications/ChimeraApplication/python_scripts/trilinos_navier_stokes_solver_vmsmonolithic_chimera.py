from __future__ import print_function, absolute_import, division  # makes KratosMultiphysics backward compatible with python 2.6 and 2.7

# Importing the Kratos Library
import KratosMultiphysics
import KratosMultiphysics.ChimeraApplication as KratosChimera
# Import applications
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
import KratosMultiphysics.TrilinosApplication as KratosTrilinos     # MPI solvers
from KratosMultiphysics.FluidDynamicsApplication import TrilinosExtension as TrilinosFluid

# Import base class file
from KratosMultiphysics.FluidDynamicsApplication.trilinos_navier_stokes_solver_vmsmonolithic import TrilinosNavierStokesSolverMonolithic

def CreateSolver(main_model_part, custom_settings):
    return TrilinosNavierStokesSolverMonolithic(main_model_part, custom_settings)

class TrilinosNavierStokesSolverMonolithic(TrilinosNavierStokesSolverMonolithic):
    def __init__(self, model, custom_settings):
        super(TrilinosNavierStokesSolverMonolithic,self).__init__(model,custom_settings)

        KratosMultiphysics.Logger.PrintInfo("TrilinosNavierStokesSolverMonolithic", "Construction of NavierStokesSolverMonolithic finished.")

    def AddVariables(self):
        super(TrilinosNavierStokesSolverMonolithic,self).AddVariables()
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.FLAG_VARIABLE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)

        KratosMultiphysics.Logger.PrintInfo("TrilinosNavierStokesSolverMonolithic", "Fluid solver variables added correctly.")

    def Initialize(self):

        ## Construct the communicator
        self.EpetraCommunicator = KratosTrilinos.CreateCommunicator()
        if hasattr(self, "_turbulence_model_solver"):
            self._turbulence_model_solver.SetCommunicator(self.EpetraCommunicator)

        ## Get the computing model part
        self.computing_model_part = self.GetComputingModelPart()

        ## If needed, create the estimate time step utility
        if (self.settings["time_stepping"]["automatic_time_step"].GetBool()):
            self.EstimateDeltaTimeUtility = self._GetAutomaticTimeSteppingUtility()

        ## Creating the Trilinos convergence criteria
        if (self.settings["time_scheme"].GetString() == "bossak"):
            self.conv_criteria = KratosTrilinos.TrilinosUPCriteria(self.settings["relative_velocity_tolerance"].GetDouble(),
                                                                   self.settings["absolute_velocity_tolerance"].GetDouble(),
                                                                   self.settings["relative_pressure_tolerance"].GetDouble(),
                                                                   self.settings["absolute_pressure_tolerance"].GetDouble())
        elif (self.settings["time_scheme"].GetString() == "steady"):
            self.conv_criteria = KratosTrilinos.TrilinosResidualCriteria(self.settings["relative_velocity_tolerance"].GetDouble(),
                                                                             self.settings["absolute_velocity_tolerance"].GetDouble())

        (self.conv_criteria).SetEchoLevel(self.settings["echo_level"].GetInt())

        ## Creating the Trilinos time scheme
        if (self.element_integrates_in_time):
            # "Fake" scheme for those cases in where the element manages the time integration
            # It is required to perform the nodal update once the current time step is solved
            self.time_scheme = KratosTrilinos.TrilinosResidualBasedIncrementalUpdateStaticSchemeSlip(
                self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE],
                self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE]+1)
            # In case the BDF2 scheme is used inside the element, set the time discretization utility to compute the BDF coefficients
            if (self.settings["time_scheme"].GetString() == "bdf2"):
                time_order = self.settings["time_order"].GetInt()
                if time_order == 2:
                    self.time_discretization = KratosMultiphysics.TimeDiscretization.BDF(time_order)
                else:
                    raise Exception("Only \"time_order\" equal to 2 is supported. Provided \"time_order\": " + str(time_order))
            else:
                err_msg = "Requested elemental time scheme " + self.settings["time_scheme"].GetString() + " is not available.\n"
                err_msg += "Available options are: \"bdf2\""
                raise Exception(err_msg)
        else:
            if not hasattr(self, "_turbulence_model_solver"):
                if self.settings["time_scheme"].GetString() == "bossak":
                    # TODO: Can we remove this periodic check, Is the PATCH_INDEX used in this scheme?
                    if self.settings["consider_periodic_conditions"].GetBool() == True:
                        self.time_scheme = TrilinosFluid.TrilinosPredictorCorrectorVelocityBossakSchemeTurbulent(
                            self.settings["alpha"].GetDouble(),
                            self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE],
                            KratosCFD.PATCH_INDEX)
                    else:
                        self.time_scheme = TrilinosFluid.TrilinosPredictorCorrectorVelocityBossakSchemeTurbulent(
                            self.settings["alpha"].GetDouble(),
                            self.settings["move_mesh_strategy"].GetInt(),
                            self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE])
                elif self.settings["time_scheme"].GetString() == "steady":
                    self.time_scheme = TrilinosFluid.TrilinosResidualBasedSimpleSteadyScheme(
                            self.settings["velocity_relaxation"].GetDouble(),
                            self.settings["pressure_relaxation"].GetDouble(),
                            self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE])
            else:
                self._turbulence_model_solver.Initialize()
                if self.settings["time_scheme"].GetString() == "bossak":
                    self.time_scheme = TrilinosFluid.TrilinosPredictorCorrectorVelocityBossakSchemeTurbulent(
                            self.settings["alpha"].GetDouble(),
                            self.settings["move_mesh_strategy"].GetInt(),
                            self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE],
                            self.settings["turbulence_model_solver_settings"]["velocity_pressure_relaxation_factor"].GetDouble(),
                            self._turbulence_model_solver.GetTurbulenceSolvingProcess())
                # Time scheme for steady state fluid solver
                elif self.settings["time_scheme"].GetString() == "steady":
                    self.time_scheme = TrilinosFluid.TrilinosResidualBasedSimpleSteadyScheme(
                            self.settings["velocity_relaxation"].GetDouble(),
                            self.settings["pressure_relaxation"].GetDouble(),
                            self.computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE],
                            self._turbulence_model_solver.GetTurbulenceSolvingProcess())

        ## Set the guess_row_size (guess about the number of zero entries) for the Trilinos builder and solver
        if self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] == 3:
            guess_row_size = 20*4
        elif self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] == 2:
            guess_row_size = 10*3

        ## Construct the Trilinos builder and solver
        if self.settings["consider_periodic_conditions"].GetBool() == True:
            KratosMultiphysics.Logger.PrintInfo("NavierStokesSolverMonolithicForChimera Periodic conditions are not implemented in this case .")
            raise NotImplementedError
        else:
            # TODO: This should be trilinos version
            self.builder_and_solver = KratosTrilinos.TrilinosBlockBuilderAndSolver(self.EpetraCommunicator,
                                                                                   guess_row_size,
                                                                                   self.trilinos_linear_solver)
        ## Construct the Trilinos Newton-Raphson strategy
        self.solver = KratosTrilinos.TrilinosNewtonRaphsonStrategy(self.main_model_part,
                                                                   self.time_scheme,
                                                                   self.trilinos_linear_solver,
                                                                   self.conv_criteria,
                                                                   self.builder_and_solver,
                                                                   self.settings["maximum_iterations"].GetInt(),
                                                                   self.settings["compute_reactions"].GetBool(),
                                                                   self.settings["reform_dofs_at_each_step"].GetBool(),
                                                                   self.settings["move_mesh_flag"].GetBool())

        (self.solver).SetEchoLevel(self.settings["echo_level"].GetInt())

        self.formulation.SetProcessInfo(self.computing_model_part)

        (self.solver).Initialize()

        KratosMultiphysics.Logger.Print("Monolithic MPI solver initialization finished.")
