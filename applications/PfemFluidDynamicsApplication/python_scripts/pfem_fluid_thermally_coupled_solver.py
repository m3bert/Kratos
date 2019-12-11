from __future__ import print_function, absolute_import, division  # makes KratosMultiphysics backward compatible with python 2.6 and 2.7
import sys

# Importing the Kratos Library
import KratosMultiphysics

# Import applications
import KratosMultiphysics.PfemFluidDynamicsApplication as KratosPfemFluid
import KratosMultiphysics.ConvectionDiffusionApplication as KratosConvDiff

# Importing the base class
from KratosMultiphysics.python_solver import PythonSolver

def CreateSolver(main_model_part, custom_settings):
    return CoupledPfemFluidThermalSolver(main_model_part, custom_settings)

class CoupledPfemFluidThermalSolver(PythonSolver):

    def __init__(self, model, custom_settings):

        self._validate_settings_in_baseclass = True
        
        super(CoupledPfemFluidThermalSolver, self).__init__(model, custom_settings)

        # Get domain size
        self.domain_size = self.settings["domain_size"].GetInt()
        
        from KratosMultiphysics.PfemFluidDynamicsApplication import python_solvers_wrapper_pfem_fluid
        self.fluid_solver = python_solvers_wrapper_pfem_fluid.CreateSolverByParameters(self.model,self.settings["fluid_solver_settings"],"OpenMP")
        
        from KratosMultiphysics.ConvectionDiffusionApplication import python_solvers_wrapper_convection_diffusion
        self.thermal_solver = python_solvers_wrapper_convection_diffusion.CreateSolverByParameters(self.model,self.settings["thermal_solver_settings"],"OpenMP")

    @classmethod
    def GetDefaultSettings(cls):
        this_defaults = KratosMultiphysics.Parameters("""{
            "solver_type": "coupled_pfem_fluid_thermal_solver",
            "model_part_name": "PfemFluidModelPart",
            "time_stepping"               : {
                    "automatic_time_step" : false,
                    "time_step"           : 0.001
                },
                "domain_size": 2,
            "echo_level"                         : 1,
            "fluid_solver_settings":{
                "physics_type"   : "fluid",
                "model_import_settings":{
                    "input_type": "mdpa",
                    "input_filename": "unknown_name"
                },
                "buffer_size": 3,
                "echo_level": 1,
                "reform_dofs_at_each_step": false,
                "clear_storage": false,
                "compute_reactions": true,
                "move_mesh_flag": true,
                "dofs"                : [],
                "stabilization_factor": 1.0,
                "line_search": false,
                "compute_contact_forces": false,
                "block_builder": false,
                "component_wise": false,
                "predictor_corrector": true,
                "time_order": 2,
                "maximum_velocity_iterations": 1,
                "maximum_pressure_iterations": 7,
                "velocity_tolerance": 1e-5,
                "pressure_tolerance": 1e-5,
                "pressure_linear_solver_settings":  {
                    "solver_type"                    : "amgcl",
                    "max_iteration"                  : 5000,
                    "tolerance"                      : 1e-9,
                    "provide_coordinates"            : false,
                    "scaling"                        : false,
                    "smoother_type"                  : "damped_jacobi",
                    "krylov_type"                    : "cg",
                    "coarsening_type"                : "aggregation",
                    "verbosity"                      : 0
                },
                "velocity_linear_solver_settings": {
                    "solver_type"                    : "bicgstab",
                    "max_iteration"                  : 5000,
                    "tolerance"                      : 1e-9,
                    "preconditioner_type"            : "none",
                    "scaling"                        : false
                },
                "solving_strategy_settings":{
                   "time_step_prediction_level": 0,
                   "max_delta_time": 1.0e-5,
                   "fraction_delta_time": 0.9,
                   "rayleigh_damping": false,
                   "rayleigh_alpha": 0.0,
                   "rayleigh_beta" : 0.0
                },
                "bodies_list": [],
                "problem_domain_sub_model_part_list": [],
                "processes_sub_model_part_list": [],
                "constraints_process_list": [],
                "loads_process_list"       : [],
                "output_process_list"      : [],
                "output_configuration"     : {},
                "problem_process_list"     : [],
                "processes"                : {},
                "output_processes"         : {},
                "check_process_list": []
            },
            "thermal_solver_settings": {
                "solver_type": "Transient",
                "analysis_type": "linear",
                "computing_model_part_name": "thermal_computing_domain",
                "model_import_settings": {
                    "input_type": "use_input_model_part"
                },
                "material_import_settings": {
                        "materials_filename": "ThermalMaterials.json"
                },
                "reform_dofs_at_each_step": true,
            },
            "coupling_settings": {}
        }""")

        this_defaults.AddMissingParameters(super(CoupledPfemFluidThermalSolver, cls).GetDefaultSettings())

        return this_defaults
        
    def AddVariables(self):
        # Import the fluid and thermal solver variables. Then merge them to have them in both fluid and thermal solvers.
        self.fluid_solver.AddVariables()
        #from KratosMultiphysics.PfemFluidDynamicsApplication import pfem_variables
        #pfem_variables.AddVariables(self.fluid_solver.main_model_part)
        self.thermal_solver.AddVariables()
        KratosMultiphysics.MergeVariableListsUtility().Merge(self.fluid_solver.main_model_part, self.thermal_solver.main_model_part)
        print("::[Coupled Pfem Fluid Thermal Solver]:: Variables MERGED")

    def ImportModelPart(self):
        # Call the fluid solver to import the model part from the mdpa
        self.fluid_solver._ImportModelPart(self.fluid_solver.main_model_part,self.settings["fluid_solver_settings"]["model_import_settings"]) # import model fluid model part and call pfem_check_and_prepare_model_process_fluid
        self.CloneThermalModelPart()

    def CloneThermalModelPart(self):
        # Save the convection diffusion settings
        convection_diffusion_settings = self.thermal_solver.main_model_part.ProcessInfo.GetValue(KratosMultiphysics.CONVECTION_DIFFUSION_SETTINGS)

        # Here the structural model part is cloned to be thermal model part so that the nodes are shared
        modeler = KratosMultiphysics.ConnectivityPreserveModeler()
        if self.domain_size == 2:
            modeler.GenerateModelPart(self.fluid_solver.main_model_part,
                                      self.thermal_solver.main_model_part,
        #                              "EulerianConvDiff2D",
                                      "ThermalFace2D2N")
        else:
            modeler.GenerateModelPart(self.fluid_solver.main_model_part,
                                      self.thermal_solver.main_model_part,
                                      "EulerianConvDiff3D",
                                      "ThermalFace3D3N")
        #self.UpdateThermalElements()
        # Set the saved convection diffusion settings to the new thermal model part
        self.thermal_solver.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.CONVECTION_DIFFUSION_SETTINGS, convection_diffusion_settings)

    def AddDofs(self):
        self.fluid_solver.AddDofs()
        self.thermal_solver.AddDofs()

    def GetComputingModelPart(self):
        return self.fluid_solver.GetComputingModelPart()

    def ComputeDeltaTime(self):
        return self.fluid_solver._ComputeDeltaTime()

    def GetMinimumBufferSize(self):
        buffer_size_fluid = self.fluid_solver.GetMinimumBufferSize()
        buffer_size_thermal = self.thermal_solver.GetMinimumBufferSize()
        self.thermal_solver.main_model_part.SetBufferSize(max(buffer_size_fluid, buffer_size_thermal)) 
        return max(buffer_size_fluid, buffer_size_thermal)

    def Initialize(self):
        self.fluid_solver.Initialize()
        self.thermal_solver.Initialize()

    def InitializeStrategy(self):
        self.fluid_solver.InitializeStrategy()

    def Clear(self):
        (self.fluid_solver).Clear()
        (self.thermal_solver).Clear()

    def Check(self):
        (self.fluid_solver).Check()
        (self.thermal_solver).Check()

    def SetEchoLevel(self, level):
        (self.fluid_solver).SetEchoLevel(level)
        (self.thermal_solver).SetEchoLevel(level)

    def AdvanceInTime(self, current_time):
        #NOTE: the cloning is done ONLY ONCE since the nodes are shared
        new_time = self.fluid_solver.AdvanceInTime(current_time)
        return new_time

    def InitializeSolutionStep(self):
        self.fluid_solver.InitializeSolutionStep()
        self.thermal_solver.InitializeSolutionStep()

    def Predict(self):
        self.fluid_solver.Predict()
        self.thermal_solver.Predict()

    def SolveSolutionStep(self):
        
        fluid_is_converged = self.fluid_solver.SolveSolutionStep()
        #print("fluid elem number: {}, thermal elem number: {}".format(len(self.fluid_solver.GetComputingModelPart().GetElements()), len(self.thermal_solver.GetComputingModelPart().GetElements())))
        #print("fluid nodes number: {}, thermal nodes number: {}".format(len(self.fluid_solver.main_model_part.GetNodes()), len(self.thermal_solver.main_model_part.GetNodes())))
        self.UpdateMeshVelocity()
        thermal_is_converged = self.thermal_solver.SolveSolutionStep()
        return (fluid_is_converged and thermal_is_converged)

    def FinalizeSolutionStep(self):
        self.fluid_solver.FinalizeSolutionStep()
        self.thermal_solver.FinalizeSolutionStep()
    
    def UpdateMeshVelocity(self):
        """ This method is executed right after solving the pfem problem

        Keyword arguments:
        self -- It signifies an instance of a class.
        """

        # We set the mesh velocity equal to the node one
        if self.domain_size == 2:
            for node in self.fluid_solver.main_model_part.Nodes:
                node.SetSolutionStepValue(KratosMultiphysics.MESH_VELOCITY_X, 0, node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X,0))
                node.SetSolutionStepValue(KratosMultiphysics.MESH_VELOCITY_Y, 0, node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y,0))
        else:
            for node in self.fluid_solver.main_model_part.Nodes:
                node.SetSolutionStepValue(KratosMultiphysics.MESH_VELOCITY_X, 0, node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X,0))
                node.SetSolutionStepValue(KratosMultiphysics.MESH_VELOCITY_Y, 0, node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y,0))
                node.SetSolutionStepValue(KratosMultiphysics.MESH_VELOCITY_Z, 0, node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Z,0))
                        
    def AuxiliarCallsBeforeRemesh(self):
        """ This method is executed right before execute the remesh
        Keyword arguments:
        self -- It signifies an instance of a class.
        """
        #print("*****BEFORE REMESHING PROCESS*****")
        #for elem in self.fluid_solver.main_model_part.Elements:
        #    if len(elem.GetNodes())==2: 
        #        print("Fluid elem Id: {}".format(elem.Id))
        #for elem in self.thermal_solver.main_model_part.Elements:
        #    print("Therm elem Id: {}".format(elem.Id))

        # We clean the computing before remesh
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.GetComputingModelPart().Nodes)
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Conditions)
        KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Elements)
        
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Nodes)
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Conditions)
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Elements)
        #print(1)
        #self.thermal_solver.main_model_part.RemoveNodes(KratosMultiphysics.TO_ERASE)
        #self.thermal_solver.main_model_part.RemoveElements(KratosMultiphysics.TO_ERASE)
        
        #self.thermal_solver.main_model_part.RemoveNodesFromAllLevels(KratosMultiphysics.TO_ERASE)
        self.thermal_solver.main_model_part.RemoveElementsFromAllLevels(KratosMultiphysics.TO_ERASE)
        #self.thermal_solver.main_model_part.RemoveNodesFromAllLevels(KratosMultiphysics.TO_ERASE)
        #self.thermal_solver.GetComputingModelPart().RemoveNodesFromAllLevels(KratosMultiphysics.TO_ERASE)
        #self.thermal_solver.GetComputingModelPart().RemoveConditionsFromAllLevels(KratosMultiphysics.TO_ERASE)
        #self.thermal_solver.GetComputingModelPart().RemoveElementsFromAllLevels(KratosMultiphysics.TO_ERASE)
        
        # We remove the thermal computing domain 
        #self.thermal_solver.main_model_part.RemoveSubModelPart(self.thermal_solver.GetComputingModelPart())
        
        # We create the thermal computing domain
        #self.thermal_solver.main_model_part.CreateSubModelPart("thermal_computing_domain")
        
        # We want to avoid the removal of the fluid nodes in the next remesh process
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, False, self.fluid_solver.main_model_part.Nodes)
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, False, self.fluid_solver.main_model_part.Elements)
        
    def AuxiliarCallsAfterRemesh(self):
        """ This method is executed right after execute the remesh

        Keyword arguments:
        self -- It signifies an instance of a class.
        """
        self.thermal_solver.main_model_part.RemoveNodesFromAllLevels(KratosMultiphysics.TO_ERASE)
        self.UpdateThermalNodes()
        #self.DeleteThermalElements()
        self.UpdateThermalElements()
        print("Number of thermal nodes:", self.thermal_solver.GetComputingModelPart().NumberOfNodes())
        print("Number of thermal elements:", self.thermal_solver.GetComputingModelPart().NumberOfElements())
        
    def PrepareModelPart(self):
        #self.GetMinimumBufferSize()
        self.fluid_solver.PrepareModelPart()
        self.UpdateThermalElements()
        self.thermal_solver.PrepareModelPart()
        print(1)

    def UpdateThermalNodes(self):
        for FluidNode in self.fluid_solver.GetComputingModelPart().Nodes:#main_model_part
            ThereIsNode = False
            for ThermalNode in self.thermal_solver.GetComputingModelPart().Nodes:
                if ThermalNode.Id == FluidNode.Id: 
                    ThereIsNode = True
                    break
            if not ThereIsNode:
                self.thermal_solver.GetComputingModelPart().AddNode(FluidNode,0)
        #KratosMultiphysics.FastTransferBetweenModelPartsProcess(self.thermal_solver.GetComputingModelPart(), self.thermal_solver.main_model_part, KratosMultiphysics.FastTransferBetweenModelPartsProcess.EntityTransfered.NODES).Execute()
    
    def UpdateThermalElements(self):
        #KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.main_model_part.Elements)
        #self.thermal_solver.main_model_part.RemoveElementsFromAllLevels(KratosMultiphysics.TO_ERASE)
        if not self.thermal_solver.main_model_part.HasSubModelPart("thermal_computing_domain"):
            self.thermal_solver.main_model_part.CreateSubModelPart("thermal_computing_domain")
        modeler = KratosMultiphysics.ConnectivityPreserveModeler()
        # for FluidElement in self.fluid_solver.GetComputingModelPart().Elements:
        #     #node_id = FluidNode.Id
        #     ThereIsElement = False
        #     for ThermalElement in self.thermal_solver.GetComputingModelPart().Elements:
        #         if ThermalElement.Id == FluidElement.Id: 
        #             ThereIsElement = True
        #             break
        #     if not ThereIsElement:
        #         #TODO: manage 3D case
        #         modeler.CloneElement(FluidElement,
        #                             self.fluid_solver.GetComputingModelPart(),
        #                             self.thermal_solver.GetComputingModelPart(),
        #                             "EulerianConvDiff2D")


        for FluidElement in self.fluid_solver.GetComputingModelPart().Elements:
            if len(FluidElement.GetNodes())==3:
                modeler.CloneElement(FluidElement,
                    self.fluid_solver.GetComputingModelPart(),
                    self.thermal_solver.GetComputingModelPart(),
                    "EulerianConvDiff2D")
        #KratosMultiphysics.FastTransferBetweenModelPartsProcess(self.thermal_solver.GetComputingModelPart(), self.thermal_solver.main_model_part, KratosMultiphysics.FastTransferBetweenModelPartsProcess.EntityTransfered.ELEMENTS).Execute()
        
    def DeleteThermalElements(self):
        KratosMultiphysics.VariableUtils().SetFlag(KratosMultiphysics.TO_ERASE, True, self.thermal_solver.GetComputingModelPart().Elements)
        self.thermal_solver.GetComputingModelPart().RemoveElementsFromAllLevels(KratosMultiphysics.TO_ERASE)
        
    