#ifndef WINSLOW_MESH_GENERATOR_H
#define WINSLOW_MESH_GENERATOR_H

#define CHKERRMY(a) try{PetscErrorCode ierr = a; if(ierr) throw(ierr);} catch (PetscErrorCode ierr) { throw PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_REPEAT,""); }

#include <typeinfo>
#include <petsc_support_routine.h>

const PetscErrorCode ERROR_NOT_INITIALIZED = 101;
const PetscErrorCode ERROR_DOMAIN_NOT_SET = 101;
const PetscErrorCode ERROR_DENSITY_NOT_SET = 102;
const PetscErrorCode ERROR_BOUNDARY_NOT_SET = 103;
const PetscErrorCode ERROR_UNKNOWN_BOUNDARY_ID = 104;
const int XI_ID = 0;
const int ETA_ID = 1;

class WinslowMeshGenerator
{
public:
    // Physical domain mesh
    Vec globalX, localX, globalY, localY;

    // Mesh density function's matrix
    Vec globalRho, localRho;

    // Problem coords
    PetscScalar left, right, top, bottom;

    // Number of nodes along X and Y axes
    PetscInt globalSizeX, globalSizeY;

    // Computational domain mesh
    Vec globalXi, localXi, globalEta, localEta;

    // Distributed array data manager
    DM da;

    // Optional user data structure
    void *userCtx;

    /************************/
    /******** METHODS *******/
    /************************/

    // Class constructor
    WinslowMeshGenerator();

    // Class destructor
    ~WinslowMeshGenerator();

    // Initializes mpi info and ksp solver
    void init();

    // Sets uniform physical domain mesh
    void setUniformMesh(PetscScalar left, PetscScalar right, PetscInt sizeX,
                        PetscScalar bottom, PetscScalar top, PetscInt sizeY);

    void setComputationalBoundary(int boundaryID, Vec bottom, Vec top, Vec left,
                                  Vec right);

    void setMeshDensityFunction(PetscErrorCode (*RhoFunc)(const WinslowMeshGenerator&));

    void updateDensity();

    void solveComputationalCoords();

    template <class TOut=double>
    TOut** getPetscVectorToZero(Vec pVec);

    template <class TOut=double>
    void printGlobal(Vec);

    void clear();

    void destroy();

private:
    /************************/
    /******* FIELDS *********/
    /************************/

    // Mesh density function's pointer
    PetscErrorCode (* RhoFunc)(const WinslowMeshGenerator&);

    // Linear system matrix
    Mat A;

    // Right-hand sides of the systems
    Vec bXi, bEta;

    // Solution of the system
    Vec u;

    // Linear system solver
    KSP ksp;

    // Linear system preconditioner
    PC pc;

    // MPI properties
    PetscMPIInt mpiSize, mpiRank;

    bool domainIsSetFlag = false, initFlag = false, densityIsSetFlag = false;
    bool boundaryXiIsSetFlag = false, boundaryEtaIsSetFlag = false;

    /************************/
    /******** METHODS *******/
    /************************/

    // Sets the linear system to solve to get computational coordinates of the initial mesh
    void fillLinearSystem();
};

#undef __FUNCT__
#define __FUNCT__ "getPetscVectorToZero"
// Function creates on zero process a c++ 2D dynamic array
// from distributed petsc vector
template <class TOut>
TOut** WinslowMeshGenerator::getPetscVectorToZero(Vec pVec)
{
    VecScatter  ctx;
    Vec         nVec, lVec;

    if(!domainIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set domain before getting vector data\n"));
        throw(ERROR_DOMAIN_NOT_SET);
    }

    CHKERRMY(DMDACreateNaturalVector(da, &nVec));
    CHKERRMY(DMDAGlobalToNaturalBegin(da, pVec, INSERT_VALUES, nVec));
    CHKERRMY(DMDAGlobalToNaturalEnd(da, pVec, INSERT_VALUES, nVec));

    CHKERRMY(VecScatterCreateToZero(nVec, &ctx, &lVec));

    // Copying solution to the first node
    CHKERRMY(VecScatterBegin(ctx,nVec,lVec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRMY(VecScatterEnd(ctx,nVec,lVec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRMY(VecDestroy(&nVec));
    if(mpiRank == 0)
    {
        TOut **res;
        const PetscScalar     *vecData;

        CHKERRMY(VecGetArrayRead(lVec, &vecData));

        res = new TOut*[globalSizeY];
        for(int i=0; i<globalSizeY; ++i)
           res[i] = new TOut[globalSizeX];

        for(int i=0; i<globalSizeY; ++i){
           for(int j=0; j<globalSizeX; ++j)
               res[i][j] = (TOut)vecData[i+j*globalSizeY];
        }

        CHKERRMY(VecRestoreArrayRead(lVec, &vecData));
        CHKERRMY(VecDestroy(&lVec));
        CHKERRMY(VecScatterDestroy(&ctx));
        return res;
    }
    else {
        CHKERRMY(VecDestroy(&lVec));
        CHKERRMY(VecScatterDestroy(&ctx));
        return NULL;
    }
}

#undef __FUNCT__
#define __FUNCT__ "printGlobal"
template <class TOut>
void WinslowMeshGenerator::printGlobal(Vec Out)
{
    TOut **vecData;

    try{vecData = getPetscVectorToZero<TOut>(Out);} catch(PetscErrorCode err)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Error in getting vector data while executing printGlobal(), Petsc error code %i", err));
        throw(err);
    }
    if(mpiRank == 0)
    {
        if (typeid(TOut) == typeid(float))
            for(PetscInt i=0; i<globalSizeY; ++i)
            {
                for(PetscInt j=0; j<globalSizeX; ++j)
                    printf("%7.4f ", vecData[i][j]);
                printf("\n");
            }
        else if (typeid(TOut) == typeid(double))
            for(PetscInt i=0; i<globalSizeY; ++i)
            {
                for(PetscInt j=0; j<globalSizeX; ++j)
                    printf("%11.6lf ", vecData[i][j]);
                printf("\n");
            }
        else
            for(PetscInt i=0; i<globalSizeY; ++i)
            {
                for(PetscInt j=0; j<globalSizeX; ++j)
                    printf("%11.6lf ", (double)vecData[i][j]);
                printf("\n");
            }
         printf("\n");

         for(PetscInt i=0; i<globalSizeY; ++i)
             delete []vecData[i];
         delete []vecData;
    }
}

#undef __FUNCT__
#define __FUNCT__ "directRhoComputation"
// Function allows to easly set mesh density function for WMG if density
// can be computed directly from coordinates
template <class TIn=double, class TOut=double>
PetscErrorCode directRhoComputation(const WinslowMeshGenerator &g, TOut (* density)(TIn x, TIn y))
{
    PetscErrorCode      ierr;
    PetscInt            localXBegin, localXSize, localYBegin, localYSize;
    const PetscScalar   **xData, **yData;
    PetscScalar         **rhoData;

    ierr = DMGlobalToLocalBegin(g.da, g.globalX, INSERT_VALUES, g.localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(g.da, g.globalX, INSERT_VALUES, g.localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(g.da, g.globalY, INSERT_VALUES, g.localY);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(g.da, g.globalY, INSERT_VALUES, g.localY);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(g.da, g.globalRho, INSERT_VALUES, g.localRho);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(g.da, g.globalRho, INSERT_VALUES, g.localRho);CHKERRQ(ierr);

    ierr = DMDAGetCorners(g.da, &localYBegin, &localXBegin, 0, &localYSize, &localXSize, 0);CHKERRQ(ierr);

    ierr = DMDAVecGetArrayRead(g.da, g.localX, &xData);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(g.da, g.localY, &yData);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(g.da, g.localRho, &rhoData);CHKERRQ(ierr);

    for(int i=localXBegin; i<localXBegin+localXSize; ++i)
        for(int j=localYBegin; j<localYBegin+localYSize; ++j){
            rhoData[i][j] = (PetscScalar)density(xData[i][j],yData[i][j]);
        }

    ierr = DMDAVecRestoreArrayRead(g.da, g.localX, &xData);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(g.da, g.localY, &yData);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(g.da, g.localRho, &rhoData);CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(g.da, g.localRho, INSERT_VALUES, g.globalRho);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(g.da, g.localRho, INSERT_VALUES, g.globalRho);CHKERRQ(ierr);

    return 0;
}

#endif // WINSLOW_MESH_GENERATOR_H
