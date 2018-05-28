//
//   Project Name:        KratosPfemFluidDynamicsApplication $
//   Created by:          $Author:                   AFranci $
//   Last modified by:    $Co-Author:                        $
//   Date:                $Date:                October 2016 $
//   Revision:            $Revision:                     0.0 $
//
//


#if !defined(KRATOS_SET_INLET_PROCESS_H_INCLUDED )
#define  KRATOS_SET_INLET_PROCESS_H_INCLUDED


// External includes

// System includes

// Project includes 
#include "containers/variables_list_data_value_container.h"
#include "spatial_containers/spatial_containers.h"

#include "includes/model_part.h"
#include "custom_utilities/mesh_error_calculation_utilities.hpp"
#include "custom_utilities/modeler_utilities.hpp"

///VARIABLES used:
//Data:      
//StepData: CONTACT_FORCE, DISPLACEMENT
//Flags:    (checked) 
//          (set)     
//          (modified)  
//          (reset)   
//(set):=(set in this process)

namespace Kratos
{

///@name Kratos Classes
///@{

/// Refine Mesh Elements Process 2D and 3D
/** The process labels the nodes to be refined (TO_REFINE)
    if the ThresholdVariable  is larger than a ReferenceThreshold
*/

class SetInletProcess
    : public Process
{
 public:
  ///@name Type Definitions
  ///@{

  /// Pointer definition of Process
  KRATOS_CLASS_POINTER_DEFINITION( SetInletProcess );

  typedef ModelPart::NodeType                   NodeType;
  typedef ModelPart::ConditionType         ConditionType;
  typedef ModelPart::PropertiesType       PropertiesType;
  typedef ConditionType::GeometryType       GeometryType;

  ///@}
  ///@name Life Cycle
  ///@{

  /// Default constructor.
  SetInletProcess(ModelPart& rModelPart,
                  int EchoLevel) 
      : mrModelPart(rModelPart)
  {
    mEchoLevel = EchoLevel;
  }


  /// Destructor.
  virtual ~SetInletProcess() {}


  ///@}
  ///@name Operators
  ///@{

  /// This operator is provided to call the process as a function and simply calls the Execute method.
  void operator()()
  {
    Execute();
  }


  ///@}
  ///@name Operations
  ///@{

  /// Execute method is used to execute the Process algorithms.
  void Execute() override
  {
    KRATOS_TRY

        if( mEchoLevel > 1 )
          std::cout<<"  SET INLET PROCESS ]; "<<std::endl;

    // const unsigned int dimension = mrModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
    // unsigned int count=0;
    for(ModelPart::NodesContainerType::iterator i_node = mrModelPart.NodesBegin() ; i_node != mrModelPart.NodesEnd() ; i_node++)
    {
      i_node->Set(INLET);
    }

    KRATOS_CATCH(" ")
  }


  /// this function is designed for being called at the beginning of the computations
  /// right after reading the model and the groups
  void ExecuteInitialize() override
  {
    KRATOS_TRY

    KRATOS_CATCH(" ")

  }

  /// this function is designed for being execute once before the solution loop but after all of the
  /// solvers where built
  void ExecuteBeforeSolutionLoop() override
  {

  }

  /// this function will be executed at every time step BEFORE performing the solve phase
  void ExecuteInitializeSolutionStep() override
  {	
  }

  /// this function will be executed at every time step AFTER performing the solve phase
  void ExecuteFinalizeSolutionStep() override
  {
  }

  /// this function will be executed at every time step BEFORE  writing the output
  void ExecuteBeforeOutputStep() override
  {
  }

  /// this function will be executed at every time step AFTER writing the output
  void ExecuteAfterOutputStep() override
  {
  }

  /// this function is designed for being called at the end of the computations
  /// right after reading the model and the groups
  void ExecuteFinalize() override
  {
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
    return "SetInletProcess";
  }

  /// Print information about this object.
  void PrintInfo(std::ostream& rOStream) const override
  {
    rOStream << "SetInletProcess";
  }

  /// Print object's data.s
  void PrintData(std::ostream& rOStream) const override
  {
  }


  ///@}
  ///@name Friends
  ///@{

  ///@}


 private:
  ///@name Static Member Variables
  ///@{

  ///@}
  ///@name Static Member Variables
  ///@{
  ModelPart& mrModelPart;
 
  ModelerUtilities mModelerUtilities;  

  int mEchoLevel;

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
  SetInletProcess& operator=(SetInletProcess const& rOther);


  /// this function is a private function


  /// Copy constructor.
  //Process(Process const& rOther);


  ///@}

}; // Class Process

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
inline std::istream& operator >> (std::istream& rIStream,
                                  SetInletProcess& rThis);

/// output stream function
inline std::ostream& operator << (std::ostream& rOStream,
                                  const SetInletProcess& rThis)
{
  rThis.PrintInfo(rOStream);
  rOStream << std::endl;
  rThis.PrintData(rOStream);

  return rOStream;
}
///@}


}  // namespace Kratos.

#endif // KRATOS_SET_INLET_PROCESS_H_INCLUDED  defined 


