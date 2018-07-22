#include "winslow_mesh_generator.h"

WinslowMeshGenerator::WinslowMeshGenerator()
{
    PetscBool  isInitialized = PETSC_FALSE;

    // If PETSc is already initialized - can run init()
    CHKERRMY(PetscInitialized(&isInitialized));
    if (isInitialized == PETSC_TRUE)
    {
        try{init();} catch(PetscErrorCode err){
            CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Error in mesh generator constructor while executing init(), Petsc error code %i", err));
            throw(err);
        }
    }
}

WinslowMeshGenerator::~WinslowMeshGenerator()
{   
    try{destroy();} catch(PetscErrorCode err){
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Error in mesh generator destructor while executing destroy(), Petsc error code %i", err));
        throw(err);
    }
}

void WinslowMeshGenerator::init()
{
    // PetscErrorCode ierr;

    CHKERRMY(MPI_Comm_size(PETSC_COMM_WORLD,&this->mpiSize));
    CHKERRMY(MPI_Comm_rank(PETSC_COMM_WORLD,&this->mpiRank));

    // Creates linear solver
    CHKERRMY(KSPCreate(PETSC_COMM_WORLD,&ksp));
    CHKERRMY(KSPGetPC(ksp,&pc));

    // Sets preconditioner to LU by default
    CHKERRMY(PCSetType(pc,PCLU));

    // Sets options from command line to ksp and pc
    CHKERRMY(KSPSetFromOptions(ksp));

    initFlag = true;
}

void WinslowMeshGenerator::setUniformMesh(PetscScalar left, PetscScalar right, PetscInt sizeX,
                                          PetscScalar bottom, PetscScalar top, PetscInt sizeY)
{
    //PetscErrorCode   ierr;
    DM               cda;
    DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
    DMDAStencilType  stype = DMDA_STENCIL_BOX;
    Vec              localCoords;
    PetscInt         localXBegin, localXSize, localYBegin, localYSize;
    PetscScalar      ** xData;
    PetscScalar      ** yData;
    DMDACoor2d       ** coordsData;

    if (bottom > top) std::swap(bottom, top);
    if (left > right) std::swap(left, right);

    this->bottom = bottom; this->top = top; this->left = left; this->right = right;
    this->globalSizeX = sizeX; this->globalSizeY = sizeY;

    CHKERRMY(DMDACreate2d(PETSC_COMM_WORLD, bx, by, stype, globalSizeY, globalSizeX,
                          PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
    CHKERRMY(DMSetFromOptions(da));
    CHKERRMY(DMSetUp(da));
    CHKERRMY(DMDASetUniformCoordinates(da, this->bottom, this->top, this->left, this->right, 0.0, 1.0));

    CHKERRMY(DMGetCoordinatesLocal(da, &localCoords));

    CHKERRMY(DMCreateGlobalVector(da, &globalX));
    CHKERRMY(DMCreateLocalVector(da, &localX));

    CHKERRMY(VecDuplicate(globalX, &globalY));
    CHKERRMY(VecDuplicate(globalX, &globalXi));
    CHKERRMY(VecDuplicate(globalX, &globalEta));
    CHKERRMY(VecDuplicate(globalX, &globalRho));

    CHKERRMY(VecDuplicate(localX, &localY));
    CHKERRMY(VecDuplicate(localX, &localXi));
    CHKERRMY(VecDuplicate(localX, &localEta));
    CHKERRMY(VecDuplicate(localX, &localRho));

    CHKERRMY(DMGetCoordinateDM(da, &cda));
    CHKERRMY(DMDAGetCorners(cda, &localYBegin, &localXBegin, 0, &localYSize, &localXSize, 0));

    CHKERRMY(DMDAVecGetArray(cda, localCoords, &coordsData));
    CHKERRMY(DMDAVecGetArray(da, localX, &xData));
    CHKERRMY(DMDAVecGetArray(da, localY, &yData));

    // Petsc stores data by columns, not by rows
    for(int i=localXBegin; i<localXBegin+localXSize; ++i)
        for(int j=localYBegin; j<localYBegin+localYSize; ++j)
        {
            yData[i][j] = coordsData[i][j].x;
            xData[i][j] = coordsData[i][j].y;
        }

    CHKERRMY(DMDAVecRestoreArray(cda, localCoords, &coordsData));
    CHKERRMY(DMDAVecRestoreArray(da, localX, &xData));
    CHKERRMY(DMDAVecRestoreArray(da, localY, &yData));

    CHKERRMY(DMLocalToGlobalBegin(da, localX, INSERT_VALUES, globalX));
    CHKERRMY(DMLocalToGlobalEnd(da, localX, INSERT_VALUES, globalX));
    CHKERRMY(DMLocalToGlobalBegin(da, localY, INSERT_VALUES, globalY));
    CHKERRMY(DMLocalToGlobalEnd(da, localY, INSERT_VALUES, globalY));

    // create A and b
    // CHKERRMY(MatSetType(A, MATMPIAIJ));
    CHKERRMY(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRMY(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,
                       (globalSizeX-2)*(globalSizeY-2),
                       (globalSizeX-2)*(globalSizeY-2)));
    CHKERRMY(MatSetFromOptions(A));
    CHKERRMY(MatSetUp(A));

    CHKERRMY(MatCreateVecs(A, &u, &bXi));
    CHKERRMY(VecDuplicate(bXi, &bEta));
    CHKERRMY(VecSet(bXi, 0));
    CHKERRMY(VecSet(bEta, 0));

    // set domain flag to true
    domainIsSetFlag = true;
    densityIsSetFlag = false;
    boundaryXiIsSetFlag = false;
    boundaryEtaIsSetFlag = false;
}

void WinslowMeshGenerator::setComputationalBoundary(int boundaryID,
        Vec bottom, Vec top, Vec left, Vec right)
{
//    //PetscErrorCode ierr;
//    PetscInt       vecBegin, vecEnd, vecLocalSize, tmpIndexX, tmpIndexY;
//    PetscScalar    *vecData, tmpVal;
    Vec         nat, M;
    const PetscScalar *vecData;
    PetscInt    vecBegin, vecEnd;

    if(boundaryID == XI_ID) M=globalXi;
    else if(boundaryID == ETA_ID) M=globalEta;
    else {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Unknown boundary ID\n"));
        throw(ERROR_UNKNOWN_BOUNDARY_ID);
    }

    if(!domainIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set domain before setting the boundary conditions\n"));
        throw(ERROR_DOMAIN_NOT_SET);
    }

    CHKERRMY(DMDACreateNaturalVector(da, &nat));

    /*** bottom ***/
    CHKERRMY(VecGetOwnershipRange(bottom, &vecBegin, &vecEnd));
    CHKERRMY(VecGetArrayRead(bottom, &vecData));
    // add try{}catch{}
    SetToPetscVector<PetscScalar, PetscInt> (nat, vecData, vecBegin*globalSizeY, vecEnd-vecBegin, globalSizeY);
    CHKERRMY(VecRestoreArrayRead(bottom, &vecData));

    /*** top ***/
    CHKERRMY(VecGetOwnershipRange(top, &vecBegin, &vecEnd));
    CHKERRMY(VecGetArrayRead(top, &vecData));
    // add try{}catch{}
    SetToPetscVector<PetscScalar, PetscInt> (nat, vecData, (vecBegin+1)*globalSizeY-1, vecEnd-vecBegin, globalSizeY);
    CHKERRMY(VecRestoreArrayRead(top, &vecData));

    /*** left ***/
    CHKERRMY(VecGetOwnershipRange(left, &vecBegin, &vecEnd));
    CHKERRMY(VecGetArrayRead(left, &vecData));
    // add try{}catch{}
    SetToPetscVector<PetscScalar, PetscInt> (nat, vecData, vecBegin, vecEnd-vecBegin);
    CHKERRMY(VecRestoreArrayRead(left, &vecData));

    /*** right ***/
    CHKERRMY(VecGetOwnershipRange(right, &vecBegin, &vecEnd));
    CHKERRMY(VecGetArrayRead(right, &vecData));
    // add try{}catch{}
    SetToPetscVector<PetscScalar, PetscInt> (nat, vecData, (globalSizeX-1)*globalSizeY+vecBegin, vecEnd-vecBegin);
    CHKERRMY(VecRestoreArrayRead(right, &vecData));

    CHKERRMY(VecAssemblyBegin(nat));
    CHKERRMY(VecAssemblyEnd(nat));

    CHKERRMY(DMDANaturalToGlobalBegin(da, nat, INSERT_VALUES, M));
    CHKERRMY(DMDANaturalToGlobalEnd(da, nat, INSERT_VALUES, M));

    //CHKERRMY(VecRestoreArray(nat, natData));
    CHKERRMY(VecDestroy(&nat));

    if(boundaryID == XI_ID) boundaryXiIsSetFlag = true;
    else if(boundaryID == ETA_ID) boundaryEtaIsSetFlag = true;
}

void WinslowMeshGenerator::setMeshDensityFunction(PetscErrorCode (*RhoFunc)(const WinslowMeshGenerator &))
{
    //PetscErrorCode ierr;

    this->RhoFunc = RhoFunc;
    densityIsSetFlag = false;
    if(domainIsSetFlag)
    {
        try{updateDensity();}catch(PetscErrorCode err){
            CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Error in setMeshDensityFunction() while trying to update mesh density, Petsc error code %i", err));
            throw(err);
        }
    }
}

void WinslowMeshGenerator::updateDensity()
{
    if(domainIsSetFlag){
        CHKERRMY(RhoFunc(*this));
        densityIsSetFlag = true;
    }
    else {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set domain before updating mesh density values\n"));
        throw(ERROR_DOMAIN_NOT_SET);
    }
}

void WinslowMeshGenerator::fillLinearSystem()
{
//    //PetscErrorCode   ierr;
    PetscInt            uBegin, uEnd, pos;
    PetscInt            localXBegin, localXSize, localYBegin, localYSize;
    const PetscScalar   **xData, **yData, **rhoData;
    PetscScalar         **xiData, **etaData;
    PetscScalar         k, kCentral, bXiVal, bEtaVal;

    if(!domainIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set domain before filling the linear system\n"));
        throw(ERROR_DOMAIN_NOT_SET);
    }

    if(!densityIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set density function and update density before filling the linear system\n"));
        throw(ERROR_DENSITY_NOT_SET);
    }

    if(!boundaryXiIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set boundary conditions for Xi before filling the linear system\n"));
        throw(ERROR_BOUNDARY_NOT_SET);
    }

    if(!boundaryEtaIsSetFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Set boundary conditions for Eta before filling the linear system\n"));
        throw(ERROR_BOUNDARY_NOT_SET);
    }

    CHKERRMY(VecGetOwnershipRange(u, &uBegin, &uEnd));
    //printf("Process %i: u begin at %i, end at %i\n", mpiRank, uBegin, uEnd);

    CHKERRMY(DMGlobalToLocalBegin(da, globalX, INSERT_VALUES, localX));
    CHKERRMY(DMGlobalToLocalEnd(da, globalX, INSERT_VALUES, localX));
    CHKERRMY(DMGlobalToLocalBegin(da, globalY, INSERT_VALUES, localY));
    CHKERRMY(DMGlobalToLocalEnd(da, globalY, INSERT_VALUES, localY));
    CHKERRMY(DMGlobalToLocalBegin(da, globalRho, INSERT_VALUES, localRho));
    CHKERRMY(DMGlobalToLocalEnd(da, globalRho, INSERT_VALUES, localRho));
    CHKERRMY(DMGlobalToLocalBegin(da, globalXi, INSERT_VALUES, localXi));
    CHKERRMY(DMGlobalToLocalEnd(da, globalXi, INSERT_VALUES, localXi));
    CHKERRMY(DMGlobalToLocalBegin(da, globalEta, INSERT_VALUES, localEta));
    CHKERRMY(DMGlobalToLocalEnd(da, globalEta, INSERT_VALUES, localEta));

    CHKERRMY(DMDAGetCorners(da, &localYBegin, &localXBegin, 0, &localYSize, &localXSize, 0));
    CHKERRMY(DMDAVecGetArrayRead(da, localX, &xData));
    CHKERRMY(DMDAVecGetArrayRead(da, localY, &yData));
    CHKERRMY(DMDAVecGetArrayRead(da, localRho, &rhoData));
    CHKERRMY(DMDAVecGetArray(da, localXi, &xiData));
    CHKERRMY(DMDAVecGetArray(da, localEta, &etaData));

    if (localXBegin == 0) {++localXBegin; --localXSize;}
    if ((localXBegin+localXSize) == globalSizeX) --localXSize;
    if (localYBegin == 0) {++localYBegin; --localYSize;}
    if ((localYBegin+localYSize) == globalSizeY) --localYSize;

    for(PetscInt i = localXBegin; i < localXBegin+localXSize ; ++i)
        for(PetscInt j = localYBegin; j < localYBegin+localYSize ; ++j)
        {
            kCentral = 0; bXiVal = 0; bEtaVal = 0;
            // Index of (i,j) node entry in matrix A
            pos = (j-1)*(globalSizeX-2)+i-1;

            // (i-1, j)
            k = 1/( ( xData[i+1][j] - xData[i-1][j] )
                    * ( rhoData[i][j] + rhoData[i-1][j] )
                    * (xData[i][j] - xData[i-1][j]));
            kCentral -= k;
            if( i > 1 )
            {
                CHKERRMY(MatSetValue(A, pos, pos-1, k, INSERT_VALUES));
            }
            else
            {
                bXiVal   -= k * xiData[i-1][j];
                bEtaVal -= k * etaData[i-1][j];
            }

            // (i+1, j)
            k = 1/( ( xData[i+1][j] - xData[i-1][j] )
                    * ( rhoData[i+1][j] + rhoData[i][j] )
                    * (xData[i+1][j] - xData[i][j]));
            kCentral -= k;
            if ( i < globalSizeX-2 )
            {
                CHKERRMY(MatSetValue(A, pos, pos+1, k, INSERT_VALUES));
            }
            else
            {
                bXiVal   -= k * xiData[i+1][j];
                bEtaVal -= k * etaData[i+1][j];
            }

            // (i, j-1)
            k = 1/( ( yData[i][j+1] - yData[i][j-1] )
                    * ( rhoData[i][j] + rhoData[i][j-1] )
                    * (yData[i][j] - yData[i][j-1]));
            kCentral -= k;
            if( j > 1 )
            {
                CHKERRMY(MatSetValue(A, pos, pos-(globalSizeX-2), k, INSERT_VALUES));
            }
            else
            {
                bXiVal   -= k * xiData[i][j-1];
                bEtaVal -= k * etaData[i][j-1];
            }

            // (i, j+1)
            k = 1/( ( yData[i][j+1] - yData[i][j-1] )
                    * ( rhoData[i][j+1] + rhoData[i][j] )
                    * (yData[i][j+1] - yData[i][j]));
            kCentral -= k;
            if( j < globalSizeY-2 )
            {
                CHKERRMY(MatSetValue(A, pos, pos+(globalSizeX-2), k, INSERT_VALUES));
            }
            else
            {
                bXiVal   -= k * xiData[i][j+1];
                bEtaVal -= k * etaData[i][j+1];
            }

            // (i, j)
            CHKERRMY(MatSetValue(A, pos, pos, kCentral, INSERT_VALUES));

            // right-hand sides
            if ( fabs(bXiVal) > 0)
                CHKERRMY(VecSetValue(bXi, pos, bXiVal, INSERT_VALUES));
            if ( fabs(bEtaVal) > 0)
                CHKERRMY(VecSetValue(bEta, pos, bEtaVal, INSERT_VALUES));
        }

    CHKERRMY(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));

        CHKERRMY(DMDAVecRestoreArrayRead(da, localX, &xData));
        CHKERRMY(DMDAVecRestoreArrayRead(da, localY, &yData));
        CHKERRMY(DMDAVecRestoreArrayRead(da, localRho, &rhoData));
        CHKERRMY(DMDAVecRestoreArray(da, localXi, &xiData));
        CHKERRMY(DMDAVecRestoreArray(da, localEta, &etaData));

    CHKERRMY(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    CHKERRMY(VecAssemblyBegin(bXi));
    CHKERRMY(VecAssemblyBegin(bEta));
    CHKERRMY(VecAssemblyEnd(bXi));
    CHKERRMY(VecAssemblyEnd(bEta));

//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Matrix of the linear system:\n"));
//    CHKERRMY(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "\n"));
//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Right-had side for Xi:\n"));
//    CHKERRMY(VecView(bXi, PETSC_VIEWER_STDOUT_WORLD));
//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "\n"));
//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Right-had side for Eta:\n"));
//    CHKERRMY(VecView(bEta, PETSC_VIEWER_STDOUT_WORLD));
//    CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "\n"));
}

void WinslowMeshGenerator::solveComputationalCoords()
{
    const PetscScalar   *uData;
    PetscInt            uBegin, uEnd, xInd, yInd;
    PetscInt            *pos;
    Vec                 vecNat;

    if(!initFlag)
    {
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Run WinslowMeshGenerator::init() after initializing Petsc and before solveComputationalCoords()\n"));
        throw(ERROR_NOT_INITIALIZED);
    }

    try{fillLinearSystem();} catch(PetscErrorCode ierr){
        if(mpiRank == 0) printf("Error in makestep() while filling linear system to solve, "
                                "Petsc error code %i", ierr);
        throw(ierr);
    }

    CHKERRMY(KSPSetOperators(ksp, A, A));
    CHKERRMY(DMDACreateNaturalVector(da, &vecNat));

    // solve for Xi
    CHKERRMY(KSPSolve(ksp, bXi, u));

    // set solution to Xi
    CHKERRMY(DMDAGlobalToNaturalBegin(da, globalXi, INSERT_VALUES, vecNat));
        CHKERRMY(VecGetOwnershipRange(u, &uBegin, &uEnd));
        CHKERRMY(VecGetArrayRead(u, &uData));
    CHKERRMY(DMDAGlobalToNaturalEnd(da, globalXi, INSERT_VALUES, vecNat));

    pos = new PetscInt[uEnd-uBegin];

    for (PetscInt i = uBegin; i < uEnd; ++i)
    {
        // x index for matrix
        xInd = (i % (globalSizeX - 2)) + 1;
        // y index for matrix
        yInd = (i / (globalSizeX - 2)) + 1;
        // index in vecNat in natural petsc order
        pos[i-uBegin] = xInd * globalSizeY + yInd;
    }

    CHKERRMY( VecSetValues(vecNat, uEnd-uBegin, pos, uData, INSERT_VALUES) );

    CHKERRMY(VecAssemblyBegin(vecNat));
        CHKERRMY(VecRestoreArrayRead(u, &uData));
        delete []pos;
    CHKERRMY(VecAssemblyEnd(vecNat));

    CHKERRMY(DMDANaturalToGlobalBegin(da, vecNat, INSERT_VALUES, globalXi));
    CHKERRMY(DMDANaturalToGlobalEnd(da, vecNat, INSERT_VALUES, globalXi));


    // solve for Eta
    CHKERRMY(KSPSolve(ksp, bEta, u));

    // set solution to Eta
    CHKERRMY(DMDAGlobalToNaturalBegin(da, globalEta, INSERT_VALUES, vecNat));
        CHKERRMY(VecGetOwnershipRange(u, &uBegin, &uEnd));
        CHKERRMY(VecGetArrayRead(u, &uData));
    CHKERRMY(DMDAGlobalToNaturalEnd(da, globalEta, INSERT_VALUES, vecNat));

    pos = new PetscInt[uEnd-uBegin];

    for (PetscInt i = uBegin; i < uEnd; ++i)
    {
        // x index for matrix
        xInd = (i % (globalSizeX - 2)) + 1;
        // y index for matrix
        yInd = (i / (globalSizeX - 2)) + 1;
        // index in vecNat in natural petsc order
        pos[i-uBegin] = xInd * globalSizeY + yInd;
    }

    CHKERRMY( VecSetValues(vecNat, uEnd-uBegin, pos, uData, INSERT_VALUES) );

    CHKERRMY(VecAssemblyBegin(vecNat));
        CHKERRMY(VecRestoreArrayRead(u, &uData));
        delete []pos;
    CHKERRMY(VecAssemblyEnd(vecNat));

    CHKERRMY(DMDANaturalToGlobalBegin(da, vecNat, INSERT_VALUES, globalEta));
    CHKERRMY(DMDANaturalToGlobalEnd(da, vecNat, INSERT_VALUES, globalEta));

    // clear natural vector
    CHKERRMY(VecDestroy(&vecNat));
}

// WinslowMeshGenerator::clear() destroys all vector and matrix objects initiaized
void WinslowMeshGenerator::clear()
{
    if(domainIsSetFlag)
    {
        CHKERRMY(MatDestroy(&A));
        CHKERRMY(VecDestroy(&u));
        CHKERRMY(VecDestroy(&bXi));
        CHKERRMY(VecDestroy(&bEta));
        CHKERRMY(VecDestroy(&globalX));
        CHKERRMY(VecDestroy(&localX));
        CHKERRMY(VecDestroy(&globalY));
        CHKERRMY(VecDestroy(&localY));
        CHKERRMY(VecDestroy(&globalXi));
        CHKERRMY(VecDestroy(&localXi));
        CHKERRMY(VecDestroy(&globalEta));
        CHKERRMY(VecDestroy(&localEta));
        CHKERRMY(VecDestroy(&globalRho));
        CHKERRMY(VecDestroy(&localRho));
        CHKERRMY(DMDestroy(&da));
        domainIsSetFlag = false;
        densityIsSetFlag = false;
        boundaryXiIsSetFlag = false;
        boundaryEtaIsSetFlag = false;
    }
}

// WinslowMeshGenerator::destroy() destroys all Petsc objects, initialized by the mesh generator
void WinslowMeshGenerator::destroy()
{
    try{clear();} catch(PetscErrorCode err){
        CHKERRMY(PetscPrintf(PETSC_COMM_WORLD, "Error in destroy() while executing clear(), Petsc error code %i", err));
        throw(err);
    }
    if(initFlag)
    {
        CHKERRMY(KSPDestroy(&ksp));
        initFlag = false;
    }
}


