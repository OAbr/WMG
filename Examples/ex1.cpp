static char help[] = "Test library Winslow_mesh_generator";
#include <iostream>
#include <petsc.h>
#include <winslow_mesh_generator.h>

using namespace std;

PetscErrorCode testRho(const WinslowMeshGenerator &);
PetscErrorCode VecSetUniform(Vec &vec, PetscInt size,
                             PetscScalar min, PetscScalar max);

const PetscScalar pi = 3.141592653;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    Vec top, bottom, left, right;
    PetscMPIInt rank;
    int sizeX = 11, sizeY = 11, minX = -3, maxX = 3, minY = -3, maxY = 3;

    WinslowMeshGenerator g;

    PetscInitialize(&argc,&args,(char*)0,help);
    CHKERRMY(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    // If WinslowMeshGenerator was created before Petsc was initialized
    // WinslowMeshGenerator::init() should be executed after initializing Petsc
    // to set proper linear solver. If WinslowMeshGenerator was created after
    // initializing Petsc, WinslowMeshGenerator::init() will be executed
    // automaticly.
    g.init();

    // Mesh density function should be set before solving the problem
    g.setMeshDensityFunction(testRho);

    // Setting the initial uniform mesh
    g.setUniformMesh(minX, maxX, sizeX, minY, maxY, sizeY);


    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "X matrix:\n"));
    g.printGlobal<float>(g.globalX);
    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Y matrix:\n"));
    g.printGlobal<float>(g.globalY);

    // If mesh density  was created before the initial mesh was set
    // WinslowMeshGenerator::updateDensity() should be executed after setting
    // the domain. If the initial mesh was set after setting the domain
    // WinslowMeshGenerator::updateDensity() will be executed automaticly.
    g.updateDensity();

    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Rho matrix:\n"));
    g.printGlobal<double>(g.globalRho);

    // create boundaries for Xi
    CHKERRMY(VecSetUniform(bottom, sizeX, 0, 1));
    CHKERRMY(VecSetUniform(top, sizeX, 0, 1));

    CHKERRMY(VecSetUniform(left, sizeY, 0, 0));
    CHKERRMY(VecSetUniform(right, sizeY, 1, 1));

    // set boundaries for Xi
    g.setComputationalBoundary(XI_ID, bottom, top, left, right);
    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Xi matrix after setting the boundaries:\n"));
    g.printGlobal(g.globalXi);

    // delete used vectors
    CHKERRMY(VecDestroy(&bottom));
    CHKERRMY(VecDestroy(&top));
    CHKERRMY(VecDestroy(&left));
    CHKERRMY(VecDestroy(&right));

    // create boundaries for Eta
    CHKERRMY(VecSetUniform(bottom, sizeX, 0, 0));
    CHKERRMY(VecSetUniform(top, sizeX, 1, 1));

    CHKERRMY(VecSetUniform(left, sizeY, 0, 1));
    CHKERRMY(VecSetUniform(right, sizeY, 0, 1));

    // set boundaries for Eta
    g.setComputationalBoundary(ETA_ID, bottom, top, left, right);
    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Eta matrix after setting the boundaries:\n"));
    g.printGlobal(g.globalEta);

    // delete used vectors
    CHKERRMY(VecDestroy(&bottom));
    CHKERRMY(VecDestroy(&top));
    CHKERRMY(VecDestroy(&left));
    CHKERRMY(VecDestroy(&right));

    g.solveComputationalCoords();
    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Xi matrix after solving the system:\n"));
    g.printGlobal(g.globalXi);
    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Eta matrix after solving the system:\n"));
    g.printGlobal(g.globalEta);

    g.destroy();

    PetscFinalize();
    printf("Process %i done!\n", rank);
    return 0;
}
#undef __FUNCT__

#define __FUNCT__ "rhoElement"
double rhoElement(double x, double y)
{
    double R,res=1;

    R = x*x + y*y;
    if (R<1) res += 1+cos(R*pi);
    return res;
}
#undef __FUNCT__

#define __FUNCT__ "testRho"
PetscErrorCode testRho(const WinslowMeshGenerator &g)
{
    PetscErrorCode      ierr;

    ierr = directRhoComputation<>(g, rhoElement);CHKERRQ(ierr);
    return 0;
}
#undef __FUNCT__

#define __FUNCT__ "VecSetUniform"
PetscErrorCode VecSetUniform(Vec &vec, PetscInt size,
                             PetscScalar min, PetscScalar max)
{
    PetscErrorCode  ierr;
    PetscScalar     step;
    PetscInt        vecBegin, vecEnd;
    PetscScalar     *vecData;

    ierr = VecCreate(PETSC_COMM_WORLD, &vec);CHKERRQ(ierr);
    ierr = VecSetSizes(vec,PETSC_DECIDE,size);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vec);CHKERRQ(ierr);

    step = (max-min)/(size-1);
    ierr = VecGetOwnershipRange(vec, &vecBegin, &vecEnd);
    ierr = VecGetArray(vec, &vecData);CHKERRQ(ierr);

    for(PetscInt i = vecBegin; i < vecEnd; ++i)
        vecData[i-vecBegin] = min + step*i;

    ierr = VecRestoreArray(vec, &vecData);CHKERRQ(ierr);

    return 0;
}
