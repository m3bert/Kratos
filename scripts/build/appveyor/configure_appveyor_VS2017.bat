rem You can use your interpreter of choice (bash, sh, zsh, ...)

rem For any question please contact with us in:
rem  - https://github.com/KratosMultiphysics/Kratos

rem Set compiler
set CC=cl.exe
set CXX=cl.exe

rem Set variables
if not defined KRATOS_BUILD_TYPE set KRATOS_BUILD_TYPE=Release
set KRATOS_SOURCE="."
set KRATOS_BUILD=".\build"
set KRATOS_APP_DIR="%KRATOS_SOURCE%\applications"

set KRATOS_BUILD_TYPE=${KRATOS_BUILD_TYPE:="Custom"}
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\StructuralMechanicsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\ContactStructuralMechanicsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\FluidDynamicsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\MeshingApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\MeshMovingApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\DEMApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\SwimmingDEMApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\CSharpWrapperApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\SolidMechanicsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\ConstitutiveModelsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\DelaunayMeshingApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\ContactMechanicsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\PfemApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\PfemFluidDynamocsApplication;"
set KRATOS_APPLICATIONS="%KRATOS_APPLICATIONS%%KRATOS_APP_DIR%\PfemSolidMechanicsApplication;"

rem Clean
del /F /Q "%POESIS_BLD%\%POESIS_BUILD_TYPE%\cmake_install.cmake"
del /F /Q "%POESIS_BLD%\%POESIS_BUILD_TYPE%\CMakeCache.txt"
del /F /Q "%POESIS_BLD%\%POESIS_BUILD_TYPE%\CMakeFiles"

rem Configure
 cmake                                              ^
 -G "Visual Studio 15 2017 Win64"                   ^
 -H"%POESIS_SRC%"                                   ^
 -B"%POESIS_BLD%/%POESIS_BUILD_TYPE%"               ^
 -DBOOST_ROOT="C:\Libraries\boost_1_65_1"           ^
 -DPYTHON_EXECUTABLE="C:\Python36-x64\python.exe"	^
 -DINCLUDE_FEAST=OFF                                ^
 -DUSE_COTIRE=ON                                    ^

rem Buid
cmake --build "%KRATOS_BUILD%/%KRATOS_BUILD_TYPE%" --target all_unity -- -j1