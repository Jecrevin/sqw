cmake_minimum_required(VERSION 3.10)
project(Filon VERSION 1.0.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Position Independent Code is needed for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Define sources
set(SOURCES
    src/Filon.cc
    src/ProgressMonitor.cc
    src/Utils.cc
)

# Define headers (for IDEs)
set(HEADERS
    inc/Filon.hh
    inc/ProgressMonitor.hh
    inc/Utils.hh
    inc/Utils.icc
    inc/constants.hh
)

# Create shared library
add_library(filon SHARED ${SOURCES} ${HEADERS})

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(filon PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include directories
target_include_directories(filon PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)

# Windows specific: Export all symbols for DLL creation
if(WIN32)
    set_target_properties(filon PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS filon
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
