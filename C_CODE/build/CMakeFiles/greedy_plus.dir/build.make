# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qing/Beam_slice

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qing/Beam_slice/build

# Include any dependencies generated for this target.
include CMakeFiles/greedy_plus.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/greedy_plus.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/greedy_plus.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/greedy_plus.dir/flags.make

CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o: CMakeFiles/greedy_plus.dir/flags.make
CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o: ../greedy_plus_MR.cpp
CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o: CMakeFiles/greedy_plus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qing/Beam_slice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o -MF CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o.d -o CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o -c /home/qing/Beam_slice/greedy_plus_MR.cpp

CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qing/Beam_slice/greedy_plus_MR.cpp > CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.i

CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qing/Beam_slice/greedy_plus_MR.cpp -o CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.s

CMakeFiles/greedy_plus.dir/max_rate.cpp.o: CMakeFiles/greedy_plus.dir/flags.make
CMakeFiles/greedy_plus.dir/max_rate.cpp.o: ../max_rate.cpp
CMakeFiles/greedy_plus.dir/max_rate.cpp.o: CMakeFiles/greedy_plus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qing/Beam_slice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/greedy_plus.dir/max_rate.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/greedy_plus.dir/max_rate.cpp.o -MF CMakeFiles/greedy_plus.dir/max_rate.cpp.o.d -o CMakeFiles/greedy_plus.dir/max_rate.cpp.o -c /home/qing/Beam_slice/max_rate.cpp

CMakeFiles/greedy_plus.dir/max_rate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greedy_plus.dir/max_rate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qing/Beam_slice/max_rate.cpp > CMakeFiles/greedy_plus.dir/max_rate.cpp.i

CMakeFiles/greedy_plus.dir/max_rate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greedy_plus.dir/max_rate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qing/Beam_slice/max_rate.cpp -o CMakeFiles/greedy_plus.dir/max_rate.cpp.s

CMakeFiles/greedy_plus.dir/func.cpp.o: CMakeFiles/greedy_plus.dir/flags.make
CMakeFiles/greedy_plus.dir/func.cpp.o: ../func.cpp
CMakeFiles/greedy_plus.dir/func.cpp.o: CMakeFiles/greedy_plus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qing/Beam_slice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/greedy_plus.dir/func.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/greedy_plus.dir/func.cpp.o -MF CMakeFiles/greedy_plus.dir/func.cpp.o.d -o CMakeFiles/greedy_plus.dir/func.cpp.o -c /home/qing/Beam_slice/func.cpp

CMakeFiles/greedy_plus.dir/func.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/greedy_plus.dir/func.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qing/Beam_slice/func.cpp > CMakeFiles/greedy_plus.dir/func.cpp.i

CMakeFiles/greedy_plus.dir/func.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/greedy_plus.dir/func.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qing/Beam_slice/func.cpp -o CMakeFiles/greedy_plus.dir/func.cpp.s

# Object files for target greedy_plus
greedy_plus_OBJECTS = \
"CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o" \
"CMakeFiles/greedy_plus.dir/max_rate.cpp.o" \
"CMakeFiles/greedy_plus.dir/func.cpp.o"

# External object files for target greedy_plus
greedy_plus_EXTERNAL_OBJECTS =

greedy_plus: CMakeFiles/greedy_plus.dir/greedy_plus_MR.cpp.o
greedy_plus: CMakeFiles/greedy_plus.dir/max_rate.cpp.o
greedy_plus: CMakeFiles/greedy_plus.dir/func.cpp.o
greedy_plus: CMakeFiles/greedy_plus.dir/build.make
greedy_plus: /usr/lib/x86_64-linux-gnu/libarmadillo.so
greedy_plus: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so
greedy_plus: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
greedy_plus: CMakeFiles/greedy_plus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qing/Beam_slice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable greedy_plus"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/greedy_plus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/greedy_plus.dir/build: greedy_plus
.PHONY : CMakeFiles/greedy_plus.dir/build

CMakeFiles/greedy_plus.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/greedy_plus.dir/cmake_clean.cmake
.PHONY : CMakeFiles/greedy_plus.dir/clean

CMakeFiles/greedy_plus.dir/depend:
	cd /home/qing/Beam_slice/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qing/Beam_slice /home/qing/Beam_slice /home/qing/Beam_slice/build /home/qing/Beam_slice/build /home/qing/Beam_slice/build/CMakeFiles/greedy_plus.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/greedy_plus.dir/depend

