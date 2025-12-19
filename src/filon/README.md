# Filon Library Build Instructions

This project uses CMake for building. It produces a shared library (`.dll`, `.so`, or `.dylib` depending on the platform).

## Prerequisites

- CMake (3.10 or higher)
- C++ Compiler (GCC, Clang, or MinGW-w64)
- OpenMP support in the compiler

## Building on Linux / macOS

1. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

2. Configure the project:
   ```bash
   cmake ..
   ```

3. Build:
   ```bash
   cmake --build .
   ```

## Building on Windows with MinGW-w64

Ensure you have MinGW-w64 installed and added to your PATH.

1. Create a build directory:
   ```cmd
   mkdir build
   cd build
   ```

2. Configure the project using the MinGW Makefiles generator:
   ```cmd
   cmake -G "MinGW Makefiles" ..
   ```

3. Build:
   ```cmd
   cmake --build .
   ```

The compiled shared library (`libfilon.dll` on Windows, `libfilon.so` on Linux, `libfilon.dylib` on macOS) will be located in the build directory.
