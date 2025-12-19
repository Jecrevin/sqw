cmake_minimum_required(VERSION 3.15)
project(sqw_project)

add_subdirectory(src/filon)

# Install the library to sqw/_filon
install(TARGETS filon DESTINATION sqw/_filon)
