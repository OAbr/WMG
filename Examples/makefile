include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLINKER=${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx
local_install=..

ex1: ex1.cpp ${local_install}/include/winslow_mesh_generator.h ${local_install}/include/petsc_support_routine.h ${local_install}/lib/libWMG.so
	-${CLINKER} -o ex1 -std=c++11 -I${local_install}/include -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include ex1.cpp ${PETSC_KSP_LIB} -L${local_install}/lib -lWMG

MC: MC.cpp ${local_install}/include/winslow_mesh_generator.h ${local_install}/include/petsc_support_routine.h ${local_install}/lib/libWMG.so
	-${CLINKER} -o MC -std=c++11 -I${local_install}/include -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include MC.cpp ${PETSC_KSP_LIB} -L${local_install}/lib -lWMG
