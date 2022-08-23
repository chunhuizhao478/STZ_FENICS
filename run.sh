rm -rf CMakeFiles
rm -rf CMakeCache.txt
rm -rf cmake_install.cmake
rm -rf Makefile
rm -rf vf.h
rm -rf project.h
rm -rf visual_eps.h
rm -rf visual_sts.h
rm -rf scalarfunc.h
rm -rf tfunc2by2.h
rm -rf tfunc3by3.h
rm -rf sts_boundary.h
rm -rf res
rm -rf stz
rm -rf ./data/mid*
rm -rf ./curve/mid*
rm -rf stress_strain.txt
rm -rf stzlocs.txt
rm -rf engcsave.txt
ffc -l dolfin vf.ufl
ffc -l dolfin project.ufl
ffc -l dolfin visual_eps.ufl
ffc -l dolfin visual_sts.ufl
ffc -l dolfin scalarfunc.ufl
ffc -l dolfin tfunc2by2.ufl
ffc -l dolfin tfunc3by3.ufl
ffc -l dolfin sts_boundary.ufl
cmake .
make
mkdir curve
mkdir data

