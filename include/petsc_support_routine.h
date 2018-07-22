#ifndef PETSC_SUPPORT_ROUTINE_H
#define PETSC_SUPPORT_ROUTINE_H

#include <petsc.h>

// Function saves data cVec to PETSc vector pVec with indeces
// starting from vecBegin and spacing indexStep between them
template <class TScalar=double, class TIndex=int>
void SetToPetscVector(Vec &pVec, const TScalar * cVec, TIndex vecBegin, TIndex vecSize, TIndex indexStep=1)
{
    PetscInt vBegin=(PetscInt)vecBegin, vSize=(PetscInt)vecSize, iStep=(PetscInt)indexStep;

    PetscInt      *indexArray = new PetscInt[vSize];
    PetscScalar   *valuesArray = new PetscScalar[vSize];

    for(PetscInt i = 0; i < vSize; ++i){
        indexArray[i] = i*iStep+vBegin;
        valuesArray[i] = (PetscScalar)cVec[i];
    }

    CHKERRMY(VecSetValues(pVec, vSize, indexArray, valuesArray, INSERT_VALUES));

    CHKERRMY(VecAssemblyBegin(pVec));
    CHKERRMY(VecAssemblyEnd(pVec));

    delete []indexArray;
    delete []valuesArray;
}


// Function creates PETSc vector pVec and saves cVec data to it with indeces
// starting from vecBegin and spacing indexStep between them
template <class TScalar=double, class TIndex=int>
void SetPetscVector(Vec &pVec, const TScalar * cVec, TIndex vecSize,
                    TIndex vecBegin, TIndex vecEnd, TIndex indexStep=1,
                    MPI_Comm comm=PETSC_COMM_WORLD)
{
    CHKERRMY(VecCreate(comm, &pVec));
    CHKERRMY(VecSetSizes(pVec,PETSC_DECIDE,(PetscInt)vecSize));
    CHKERRMY(VecSetFromOptions(pVec));

    SetToPetscVector<TScalar, TIndex> (pVec, cVec, vecBegin, vecEnd-vecBegin, indexStep);
}

#endif // PETSC_SUPPORT_ROUTINE_H
