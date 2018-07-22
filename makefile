include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLINKER=${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx
local_install=.

WMG: winslow_mesh_generator.cpp ./include/winslow_mesh_generator.h ./include/petsc_support_routine.h
	-${CLINKER} -c -fpic -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -I./include -std=c++11 winslow_mesh_generator.cpp
	-${CLINKER} -shared -o ./lib/libWMG.so winslow_mesh_generator.o
	${RM} winslow_mesh_generator.o
	cp ./lib/libWMG.so ${local_install}/lib/
	cp ./include/winslow_mesh_generator.h ${local_install}/include/
	cp ./include/petsc_support_routine.h ${local_install}/include/
