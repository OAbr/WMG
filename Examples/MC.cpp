#include <mpi.h>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <stdio.h>
#include <chrono>
#include <winslow_mesh_generator.h>

PetscErrorCode testRho(const WinslowMeshGenerator& g);
void SetPetscVector(Vec &pVec, double *cVec, int size, int vBegin, int vEnd);
void GetPetscVector(DM da, Vec pVec, double **&res);

//struct: Xi and Eta, the solutions of the mesh generation PDEs
struct XiEta{
    double Xi;
    double Eta;
};

double pi = 3.14159265359;
// global domain [a1, b1] x [a2, b2]
double a1 = 0, b1 = 1, a2 = 0 , b2 = 1 ;

//parameters for example 1 of w
double x_c = 0.75 ,y_c= 0.5 ;
double R = 15;
#define min(x,y) (std::min(x,y))
#define max(x,y) (std::max(x,y))

#undef __FUNCT__
//boundary values: boundaryX for Xi, boundaryY for Eta
double boundaryX(double x){
    return (x-a1)/(b1-a1);}

double boundaryY(double y){
    return (y-a2)/(b2-a2);}

//Mesh density function rho = 1/w.
double  w ( double x , double y )
{
    //Example 1
    /*double rho = (1+R*exp(-50*(x-x_c)*(x-x_c) -50*(y-y_c)*(y-y_c) ) );
    double w = 1/rho;
    return   w ;*/


    //Example 2
    //double rho = 2 + cos(20*pi*x)*sin(20*pi*y);
    //double w = 1/rho;
    //return w;

    //Example 3

    //double alpha = 10, R = 50;
    //double w = 1 + alpha*exp(-R*(y-0.5-0.25*sin(2*pi*x))*(y-0.5-0.25*sin(2*pi*x)));
    //return w;

    return 1;
}

//x-derivative of w
double w_x(double x, double y)
{
    //Example 1
    //return -w(x,y)*w(x,y)*(-100*R*(x-x_c)*exp(-50*(x-x_c)*(x-x_c) -50*(y-y_c)*(y-y_c) )  );

    //Example 2
    //return -w(x,y)*w(x,y)*(-20*pi*sin(20*pi*x)*sin(20*pi*y));

    //Example 3
    //double alpha = 10, R = 50;
    //double w_x = alpha*( -2*R* ( y-0.5-0.25*sin(2*pi*x) )*(-0.5*pi*cos(2*pi*x)))*exp(-R*(y-0.5-0.25*sin(2*pi*x))*(y-0.5-0.25*sin(2*pi*x)));
    //return w_x;

    return 0;
}

//y-derivative of w
double w_y(double x, double y)
{
    //Example 1
    //return -w(x,y)*w(x,y)*(-100*R*(y-y_c)*exp(-50*(x-x_c)*(x-x_c) -50*(y-y_c)*(y-y_c) )  );

    //Example 2
    //return -w(x,y)*w(x,y)*(20*pi*cos(20*pi*x)*cos(20*pi*y));

    //Example 3
    //double alpha = 10, R = 50;
    //double w_y = alpha*(-2*R* ( y-0.5-0.25*sin(2*pi*x) ))*exp(-R*(y-0.5-0.25*sin(2*pi*x))*(y-0.5-0.25*sin(2*pi*x)));
    //return w_y;

    return 0;
}

#define __FUNCT__ "rhoElement"
double rhoElement(double x, double y)
{
    return 1.0/w(x,y);
}
#undef __FUNCT__

//Monte Carlo code for computing the solution at the boundaries of the subdomains.
XiEta ** MonteCarlo(double a1, double b1, double a2, double b2, int sizeX, int sizeY,
                    double lx, double ly, double h , int walks, int my_id,
                    int ** flag, int interpolation_step)
{
    double sh = sqrt(2*h);
    std::mt19937 generator ; //object Mersienne Twisted generator.
    std::normal_distribution<long double> distribution(0,1) ; //Standard Normal distribution.

    // u is the solution, originally set up to be zero.
    XiEta ** u;

    u = new XiEta*[sizeY];
    for (int i = 0; i < sizeY; ++i){
        u[i] = new XiEta[sizeX];
        for (int j = 0; j < sizeX; ++j){
            u[i][j].Xi = 0; u[i][j].Eta = 0;
        }
    }

    //Now application of Feynmann Kak formula or interpolation

    double x0, y0; double x,y;

    // flag = -1 : set up the values at the boundary points, using the boundary functions
    // boundaryX and boundaryY
    for (int i = 0 ; i < sizeY; i++)
        for (int j = 0 ; j < sizeX; j++)
            if(flag[i][j] == -1)
            {
                x0 = a1+j*lx ; y0 = a2+i*ly ;
                u[i][j].Xi = boundaryX(x0);
                u[i][j].Eta = boundaryY(y0);
            }

    // flag = 1: do Monte Carlo at the interface points that are flagged as 1
    for (int i = 0 ; i < sizeY; i++)
        for (int j = 0 ; j < sizeX; j++)
            if(flag[i][j] == 1)
            {
                x0 = a1+j*lx ;
                y0 = a2+i*ly ;
                //if (my_id == 0){std::cout<<x0<<" "<<y0<<std::endl;}
                for (int t = 0 ; t < walks; t++){
                    generator.seed(t+my_id*walks);x = x0 ; y = y0 ;
                    while ( x>a1 && x<b1 && y>a2 && y<b2 ){
                        //if (my_id == 0){std::cout<<"Doing Monte Carlo"<<std::endl;}
                        x += (w_x(x,y)*h)/(w(x,y)) + sh * distribution(generator); //if (my_id == 0) {std::cout<<stepsx[t][m]<<std::endl;}
                        y += (w_y(x,y)*h)/(w(x,y)) + sh * distribution(generator);
                        /*x += w_x(x,y)*h + stepsx[t][m] * sqrt(w(x,y)); //if (my_id == 0) {std::cout<<stepsx[t][m]<<std::endl;}
                              y += w_y(x,y)*h + stepsy[t][m] * sqrt(w(x,y));*/
                        } //while not boundary

                    u[i][j].Xi = u[i][j].Xi + boundaryX(x) ;
                    u[i][j].Eta = u[i][j].Eta + boundaryY(y) ;
                    //if (warning > maxsteps ){std::cout<<"Warning: m ="<<warning<<" for walk number "<<t<<std::endl;} //warn the user that a walk exceeded maxsteps
                }
                //if (my_id == 0) {std::cout<<"end of step"<<std::endl;}
                u[i][j].Xi /= walks ;
                u[i][j].Eta /= walks ;
            }

    for (int i = 0 ; i < sizeY; i++)
        for (int j = 0 ; j < sizeX; j++)
            switch(flag[i][j])
            {
                case  2: {// flag = 2 : interpolation along horizontal segments
                    int j0 = j - j % (interpolation_step);
                    int j1 = j0 + (interpolation_step);
                    if (j1 > sizeX-1){j1 = sizeX-1;}
                    double A = j0; double B = j1;
                    u[i][j].Xi = u[i][j0].Xi + ((u[i][j1].Xi - u[i][j0].Xi) / (B-A)) * (j % interpolation_step);
                    u[i][j].Eta = u[i][j0].Eta + ((u[i][j1].Eta - u[i][j0].Eta) / (B-A)) * (j % interpolation_step);
                    break;
                }
                case  3: {// flag = 3 : interpolation along vertical segments
                    int i0 = i - i % (interpolation_step);
                    int i1 = i0 + (interpolation_step);
                    if (i1 > sizeY-1){i1 = sizeY-1;}
                    double C = i0; double D = i1;
                    u[i][j].Xi = u[i0][j].Xi + ((u[i1][j].Xi - u[i0][j].Xi) / (D-C)) * (i % interpolation_step);
                    u[i][j].Eta = u[i0][j].Eta + ((u[i1][j].Eta - u[i0][j].Eta) / (D-C)) * (i % interpolation_step);
                    break;
                }
            }

    return u ;
}

// function to get x coordinate from index
double getCoordX(int i, int size)
{
    return a1+i*(b1-a1)/(size-1);
}

// function to get y coordinate from index
double getCoordY(int i, int size)
{
    return a2+i*(b2-a2)/(size-1);
}

#define __FUNCT__ "main"
//main code:
int main(int argc, char** argv)
{
    // initialize MPI, get size and processor number
    MPI::Init(argc,argv);
    MPI::Status status;
    int size=MPI::COMM_WORLD.Get_size();
    int my_id=MPI::COMM_WORLD.Get_rank();

    // Creation of the MPI Data type XiEta
    int blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype MPI_XiEta;
    MPI_Aint offsets[2];

    offsets[0] = offsetof(XiEta , Xi);
    offsets[1] = offsetof(XiEta, Eta);

    MPI_Type_create_struct(2,blocklengths,offsets,types,&MPI_XiEta);
    MPI_Type_commit(&MPI_XiEta);

    // timing variables
    double t5,t4,t3,t2,t1,t0;

    // set up the geometry of the problem
    int subDomainX = atoi(argv[1]) ; int subDomainY = atoi(argv[2]) ; //number of subdomains along x and y diretions

    int totalSubDomains = subDomainX*subDomainY ;  // total number of subdomains

    int locsizeX = atoi(argv[3]) ; int locsizeY = atoi(argv[4]) ; //number of points per subdomain
    int sizeX = subDomainX*locsizeX-(subDomainX-1) , sizeY = subDomainY*locsizeY-(subDomainY-1) ; //points on global domain

    double sizeofintervalX = (b1-a1)/(sizeX-1) , sizeofintervalY = (b2-a2)/(sizeY-1) ; //length of small subinterval

    double sizeOfSubdomainX = (b1-a1)/subDomainX; //X length of subdomain
    double sizeOfSubdomainY = (b2-a2)/subDomainY; //Y length of subdomain

    // the number of processors must be a non-zero multiple of the number of subdomains
    if (size%(subDomainX*subDomainY) !=0  || size < subDomainX*subDomainY){
        if (my_id == 0){
            std::cout<<"The number of processors must be a multiple of the number of subdomains. Try again or buy a better computer."<<std::endl;
        }
        std::exit(0);
    }

    // creation of a new Communicator
    int processorsPerCommunicator = size/(totalSubDomains) ; //number of processors per communicators
    int color = my_id/processorsPerCommunicator ;
    MPI_Comm comm ;
    MPI_Comm_split(MPI_COMM_WORLD, color, my_id, &comm);
    int local_id ;
    MPI_Comm_rank(comm, &local_id);
    PETSC_COMM_WORLD = comm ; //set up local communicator for each subdomain
    PetscInitialize(&argc,&argv,(char*)0,NULL);

    // get optional settings

    // number of random walks performed by each processor
    double h = 1.0/max(sizeofintervalX, sizeofintervalY);
    if(h<100){h*=h; h*=h;}else{h=100000;}
    int walks_per_processor = (int)(h);
    CHKERRMY(PetscOptionsGetInt(NULL,NULL,"-random_walks",&walks_per_processor,NULL));

    // interpolation step
    int interpolation_step = sqrt(min(sizeX, sizeY));
    CHKERRMY(PetscOptionsGetInt(NULL,NULL,"-interpolation_step",&interpolation_step,NULL));

    // Monte Carlo time step
    double time_step = 1.0 / sqrt(walks_per_processor);
    CHKERRMY(PetscOptionsGetReal(NULL,NULL,"-time_step",&time_step,NULL));

    // print results flag
    PetscBool print_results = PETSC_TRUE;
    CHKERRMY(PetscOptionsGetBool(NULL,NULL,"-print_results",&print_results,NULL));

    // calculate residuals flag
    PetscBool calc_residuals = PETSC_FALSE;
    CHKERRMY(PetscOptionsGetBool(NULL,NULL,"-calc_residuals",&calc_residuals,NULL));

    // print residuals flag
    PetscBool print_residuals = PETSC_FALSE;
    CHKERRMY(PetscOptionsGetBool(NULL,NULL,"-print_residuals",&print_residuals,NULL));

    // start timing
    t0 = MPI_Wtime();

    // flag values for points: see explanation for specific values. The original set up is all =0.
    int ** flag;
    flag = new int*[sizeY];
    for (int i = 0; i < sizeY; ++i){
        flag[i] = new int[sizeX];
        for (int j = 0; j < sizeX; ++j){
            flag[i][j] = 0;
        }
    }

    //Horizontal boundary points flag = -1 because we fixed the values using boundary condition
    for (int i = 0; i < sizeY; ++i){
        flag[i][0] = -1;
        flag[i][sizeX-1] = -1;
    }

    //Vertical boundary points flag = -1 because we fixed the values using boundary condition
    for (int j = 0; j < sizeX; ++j){
        flag[0][j] = -1;
        flag[sizeY-1][j] = -1;
    }

    //Horizontal interfaces
    if (subDomainY != 1){
        for (int i = locsizeY-1; i < sizeY-1; i += locsizeY-1)
            for (int j = 1; j < sizeX-1; ++j)
                if (j % interpolation_step) flag[i][j] = 2;
                else flag[i][j] = 1;
    }

    //Vertical interfaces
    if (subDomainX != 1){
        for (int i = 1; i < sizeY-1; ++i)
            for (int j = locsizeX-1; j < sizeX-1; j += locsizeX-1)
                if (i % interpolation_step) {
                    if (flag[i][j] != 1) flag[i][j] = 3;
                }
                else flag[i][j] = 1;
    }

    //print out the flag values to make sure they are correct and to know what is done at each point
    /*if (my_id == 0){
        std::cout<<"Flags: -1 = Boundary Conditions, 0 = Subsolver 1 = Montecarlo, 2 = Horizontal interpolation, 3 = Vertical interpolation"<<std::endl;
        for (int i = 0 ; i < sizeY; ++i){
            for (int j = 0 ; j < sizeX; ++j)
                std::cout<<flag[i][j]<<" ";
            std::cout<<" "<<std::endl;
        }
    }*/

    //now we call the MonteCarlo function
    t1 = MPI_Wtime();

    XiEta ** u; //the solution

    if (totalSubDomains > 1) {
        u = MonteCarlo(a1, b1, a2, b2, sizeX, sizeY, sizeofintervalX, sizeofintervalY,
                       time_step, walks_per_processor/size, my_id, flag, interpolation_step);
    }
    else
    {
        // initialize solution array
        u = new XiEta*[sizeY];
        for (int i = 0; i < sizeY; ++i){
            u[i] = new XiEta[sizeX];
            for (int j = 0; j < sizeX; ++j){
                u[i][j].Xi = 0; u[i][j].Eta = 0;
            }
        }

        double x0, y0;
        // set vertical boundaries
        for (int i = 0; i < sizeY; ++i){
            y0 = (double) i / (sizeY-1);
            u[i][0].Eta = y0;
            u[i][sizeX-1].Eta = y0;
            u[i][sizeX-1].Xi = 1;
        }

        // set horizontal boundaries
        for (int j = 0; j < sizeX; ++j){
            x0 = (double) j / (sizeX-1);
            u[0][j].Xi = x0;
            u[sizeY-1][j].Xi = x0;
            u[sizeY-1][j].Eta = 1;
        }
    }

    // the non-zero processors send their partial solution to the zero processor,
    // this is done by first rearranging ** u into a * vectsend
    t2 = MPI_Wtime();

    if (totalSubDomains > 1){
        if (my_id !=0){
            XiEta * uvectsend;
            uvectsend = new XiEta[sizeX*sizeY];
            for (int i = 0; i <sizeY; ++i){
                for (int j = 0; j < sizeX; ++j){
                    uvectsend[i*sizeX + j] = u[i][j];
                }
            }
            MPI::COMM_WORLD.Send(uvectsend,sizeX*sizeY,MPI_XiEta,0,1997+my_id);
            delete [] uvectsend;
        }

        //this is the receive step from processor 0;, which includes sum
        else {
            XiEta * uvectrecv;
            uvectrecv = new XiEta[sizeX*sizeY];
            for (int r = 1; r < size; ++r){
                MPI::COMM_WORLD.Recv(uvectrecv,sizeX*sizeY,MPI_XiEta,r,1997+r);
                for (int i = 0; i <sizeY; ++i){
                    for (int j = 0; j < sizeX; ++j){
                        u[i][j].Xi =  u[i][j].Xi + uvectrecv[i*sizeX+j].Xi;
                        u[i][j].Eta =  u[i][j].Eta + uvectrecv[i*sizeX+j].Eta;
                    }
                }
            }
            delete [] uvectrecv;

            //final average computed by processor 0
            for (int i = 0; i< sizeY; ++i){
                for (int j = 0; j <sizeX; ++j){
                    u[i][j].Xi /= size;
                    u[i][j].Eta /= size;
                    //std::cout<<r<<" "<<urecv[i][j].Xi<<" "<<u[i][j].Xi<<std::endl;
                }
            }
        }
    }

    // Scatter: this is is important in this version since the subdomain solver part of the code
    // is in parallel. However this send is wasteful. Preliminary version.

    if (totalSubDomains > 1){
        if (my_id == 0){
            XiEta * uvectsend;
            uvectsend = new XiEta[sizeX*sizeY];
            for (int i = 0; i <sizeY; ++i){
                for (int j = 0; j < sizeX; ++j){
                    uvectsend[i*sizeX+j] = u[i][j];
                }
            }

            for (int r = 1; r < size; ++r){
                MPI::COMM_WORLD.Send(uvectsend,sizeX*sizeY,MPI_XiEta,r,1997+r);
            }
            delete [] uvectsend;
        }

        else{
            XiEta * uvectrecv;
            uvectrecv = new  XiEta[sizeX*sizeY];
            MPI::COMM_WORLD.Recv(uvectrecv,sizeX*sizeY,MPI_XiEta,0,1997+my_id);
            for (int i = 0; i< sizeY; ++i){
                for (int j = 0; j <sizeX; ++j){
                    u[i][j] =  uvectrecv[i*sizeX+j];
                    //std::cout<<r<<" "<<urecv[i][j].Xi<<" "<<u[i][j].Xi<<std::endl;
                }
            }
            delete [] uvectrecv;
        }

        /*if (my_id == 0){
            for (int i = 0; i < sizeY ; ++i){
                for (int j = 0; j < sizeX; ++j){
                    std::cout<<u[i][j].Xi<<" ";
                }
                std::cout<<std::endl;
            }
        }*/
    }

    if (my_id == 0){
        std::cout<<"End of construction of boundary conditions using Monte Carlo"<<std::endl;
    }

    //subdomain solver using Oleksandr petsc-based libray
    t3 = MPI_Wtime();

    if (my_id == 0){std::cout<<"Starting Petsc"<<std::endl;}
    //std::cout<<"My global rank "<<my_id<<", my local rank "<<local_id<<std::endl;

    //subdomain boundaries
    double *bottom=new double[locsizeX], *top=new double[locsizeX];
    double *left=new double[locsizeY], *right=new double[locsizeY];
    WinslowMeshGenerator g;
    Vec petscBottom, petscTop, petscLeft, petscRight;

    //the first two for loop must disappear and be parallelized according to the match id-subdomain
    int subDomainIndexX = my_id/(processorsPerCommunicator*subDomainY) + 1;
    int subDomainIndexY = (my_id/processorsPerCommunicator)%subDomainY + 1;

    // Setting minX, maxX, minY, maxY (the endpoints of the 4 intervals of the subdomain)
    double minX = a1 + sizeOfSubdomainX*(subDomainIndexX-1);
    double maxX = a1 + sizeOfSubdomainX*(subDomainIndexX);
    double minY = a2 + sizeOfSubdomainY*(subDomainIndexY-1);
    double maxY = a2 + sizeOfSubdomainY*(subDomainIndexY);

    // Set subdomain mesh
    g.setUniformMesh(minX, maxX, locsizeX, minY, maxY, locsizeY);

    // Set mesh density function
    g.setMeshDensityFunction(testRho);

    // set boundaries for Xi
    if(local_id ==0){
        //std::cout<<"X:"<<std::endl;
        //horizontal bottom and top boundary intervals for subdomain
        for (int j = 0; j < locsizeX; ++j){
            bottom[j] = u[(subDomainIndexY-1)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Xi;
            top[j] = u[(subDomainIndexY)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Xi;
            //std::cout<<bottomLocalXi[i]<<" "<<topLocalXi[i]<<std::endl;
        }

        //std::cout<<"Y:"<<std::endl;
        //vertical left and right boundary intervals for subdomain
        for (int i = 0; i < locsizeY; ++i){
            left[i] = u[i+(subDomainIndexY-1)*(locsizeY-1)][(subDomainIndexX-1)*(locsizeX-1)].Xi;
            right[i] = u[i+(subDomainIndexY-1)*(locsizeY-1)][(subDomainIndexX)*(locsizeX-1)].Xi;
            //std::cout<<leftLocalXi[j]<<" "<<rightLocalXi[j]<<std::endl;
        }

        // Set petsc vectors for Xi boundaries
        SetPetscVector<>(petscBottom, bottom, locsizeX, 0, locsizeX);
        SetPetscVector<>(petscTop, top, locsizeX, 0, locsizeX);
        SetPetscVector<>(petscLeft, left, locsizeY, 0, locsizeY);
        SetPetscVector<>(petscRight, right, locsizeY, 0, locsizeY);
    }
    else{
        // need to run empty SetPetscVector() on other nodes because it is a collective operation
        // in Petsc
        SetPetscVector<>(petscBottom, bottom, locsizeX, 0, 0);
        SetPetscVector<>(petscTop, top, locsizeX, 0, 0);
        SetPetscVector<>(petscLeft, left, locsizeY, 0, 0);
        SetPetscVector<>(petscRight, right, locsizeY, 0, 0);
    }

    // set computational boundary for Xi
    g.setComputationalBoundary(XI_ID, petscBottom, petscTop, petscLeft, petscRight);
    //g.printGlobal(g.globalXi);

    //destroy vectors that are not needed anymore
    CHKERRMY(VecDestroy(&petscBottom)); CHKERRMY(VecDestroy(&petscTop));
    CHKERRMY(VecDestroy(&petscLeft)); CHKERRMY(VecDestroy(&petscRight));

    // set boundaries for Eta
    if(local_id == 0){
        //std::cout<<"X:"<<std::endl;
        //horizontal bottom and top boundary intervals for subdomain
        for (int j = 0; j < locsizeX; ++j){
            bottom[j] = u[(subDomainIndexY-1)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Eta;
            top[j] = u[(subDomainIndexY)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Eta;
            //std::cout<<bottomLocalXi[i]<<" "<<topLocalXi[i]<<std::endl;
        }

        //std::cout<<"Y:"<<std::endl;
        //vertical left and right boundary intervals for subdomain
        for (int i = 0; i < locsizeY; ++i){
            left[i] = u[i+(subDomainIndexY-1)*(locsizeY-1)][(subDomainIndexX-1)*(locsizeX-1)].Eta;
            right[i] = u[i+(subDomainIndexY-1)*(locsizeY-1)][(subDomainIndexX)*(locsizeX-1)].Eta;
            //std::cout<<leftLocalXi[j]<<" "<<rightLocalXi[j]<<std::endl;
        }

        // Set petsc vectors for Xi boundaries
        SetPetscVector<>(petscBottom, bottom, locsizeX, 0, locsizeX);
        SetPetscVector<>(petscTop, top, locsizeX, 0, locsizeX);
        SetPetscVector<>(petscLeft, left, locsizeY, 0, locsizeY);
        SetPetscVector<>(petscRight, right, locsizeY, 0, locsizeY);
    }
    else{
        // need to run empty SetPetscVector() on other nodes because it is a collective operation
        // in Petsc
        SetPetscVector<>(petscBottom, bottom, locsizeX, 0, 0);
        SetPetscVector<>(petscTop, top, locsizeX, 0, 0);
        SetPetscVector<>(petscLeft, left, locsizeY, 0, 0);
        SetPetscVector<>(petscRight, right, locsizeY, 0, 0);
    }

    //Computational boundary for Eta
    g.setComputationalBoundary(ETA_ID, petscBottom, petscTop, petscLeft, petscRight);
    //g.printGlobal(g.globalEta);

    //destroy vectors that are not needed anymore
    CHKERRMY(VecDestroy(&petscBottom)); CHKERRMY(VecDestroy(&petscTop));
    CHKERRMY(VecDestroy(&petscLeft)); CHKERRMY(VecDestroy(&petscRight));

    delete [] bottom; delete [] top;
    delete [] left; delete [] right;

    if (my_id == 0){std::cout<<"Boundaries are set"<<std::endl;}

    //compute solution for Xi and Eta
    g.solveComputationalCoords();

    if (my_id == 0){std::cout<<"Solved"<<std::endl;}
    //g.printGlobal(g.globalXi);
    //g.printGlobal(g.globalEta);

    // collecting the solutions for Xi and Eta to local node zero
    double **resXi; double **resEta;
    resXi = g.getPetscVectorToZero(g.globalXi);
    resEta = g.getPetscVectorToZero(g.globalEta);

    // collecting the solutions for Xi and Eta to global node zero
    if (local_id == 0 && my_id!=0){
        XiEta * usend;
        usend = new XiEta[locsizeX*locsizeY];
        for(int i=0; i<locsizeY; ++i){
            for(int j=0; j<locsizeX; ++j){
                usend[i*locsizeX+j].Xi = resXi[i][j];
                usend[i*locsizeX+j].Eta = resEta[i][j];
            }
        }

        MPI::COMM_WORLD.Send(usend,locsizeX*locsizeY,MPI_XiEta,0,1999);
        delete [] usend;
    }
    if (my_id == 0){
        // copy data from current process
        for(int i=0; i<locsizeY; ++i){
            for(int j=0; j<locsizeX; ++j){
                u[i][j].Xi = resXi[i][j];
                u[i][j].Eta = resEta[i][j];
            }
        }

        XiEta* urecv1;
        urecv1 = new XiEta[locsizeX*locsizeY];

        for (int k = 1; k < subDomainX*subDomainY; ++k){
            int sender_id = processorsPerCommunicator*k;
            int subDomainIndexX = sender_id/(processorsPerCommunicator*subDomainY) + 1;
            int subDomainIndexY = (sender_id/processorsPerCommunicator)%subDomainY + 1;
            //MPI_Recv(&urecv1[0][0],locsizeX*locsizeY,MPI::DOUBLE,k,1999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            std::cout<<"Receive from "<<sender_id<<std::endl;
            MPI::COMM_WORLD.Recv(urecv1,locsizeX*locsizeY,MPI_XiEta,sender_id,1999);
            //MPI::COMM_WORLD.Recv(&testrecv,1,MPI::DOUBLE,processorsPerCommunicator*k,1999);
            //std::cout<<"Received from "<<k<<":"<<std::endl;
            for(int i=0; i<locsizeY; ++i){
                for(int j=0; j<locsizeX; ++j){
                    u[i+(subDomainIndexY-1)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Xi = urecv1[i*locsizeX+j].Xi;
                    u[i+(subDomainIndexY-1)*(locsizeY-1)][j+(subDomainIndexX-1)*(locsizeX-1)].Eta = urecv1[i*locsizeX+j].Eta;
                }
            }
        }
        delete [] urecv1;
    }

    g.destroy();

    PetscFinalize();
    MPI_Comm_free(&comm);

    MPI::COMM_WORLD.Barrier();
    //Calculate residual

    double **rXi = NULL, **rEta = NULL;
    double residualXi=0, residualEta=0, residualXiMax = 0, residualEtaMax = 0;
    if (calc_residuals) {
        if(my_id == 0)
        {
            rXi = new double*[sizeY];
            rEta = new double*[sizeY];
            for(int i=0; i<sizeY; ++i){
                rXi[i] = new double[sizeX];
                rEta[i] = new double[sizeX];
            }

            double k, kCentral, tmpResXi, tmpResEta;
            int count = 0;
            for(int i=0; i < sizeY; ++i){
                for(int j=0; j < sizeX; ++j){
                    //std::cout<<i<<" "<<j<<" "<<flag[i][j]<<std::endl;
                    if(flag[i][j]>=0){
                        kCentral = 0, tmpResXi=0, tmpResEta = 0;
                        // (i, j-1)
                        k = 1/( (getCoordX(j+1,sizeX) - getCoordX(j-1,sizeX))
                                * (rhoElement(getCoordX(j,sizeX),getCoordY(i,sizeY)) + rhoElement(getCoordX(j-1,sizeX),getCoordY(i,sizeY)))
                                * (getCoordX(j,sizeX) - getCoordX(j-1,sizeX)));
                        kCentral -= k;
                        tmpResXi += k * u[i][j-1].Xi;
                        tmpResEta += k * u[i][j-1].Eta;
                        //std::cout<<k<<" ";

                        // (i, j+1)
                        k = 1/( (getCoordX(j+1,sizeX) - getCoordX(j-1,sizeX))
                                * (rhoElement(getCoordX(j+1,sizeX),getCoordY(i,sizeY)) + rhoElement(getCoordX(j,sizeX),getCoordY(i,sizeY)))
                                * (getCoordX(j+1,sizeX) - getCoordX(j,sizeX)));
                        kCentral -= k;
                        tmpResXi += k * u[i][j+1].Xi;
                        tmpResEta += k * u[i][j+1].Eta;
                        //std::cout<<k<<" ";

                        // (i-1, j)
                        k = 1/( (getCoordY(i+1,sizeY) - getCoordY(i-1,sizeY))
                                * (rhoElement(getCoordX(j,sizeX),getCoordX(i,sizeY)) + rhoElement(getCoordX(j,sizeX),getCoordY(i-1,sizeY)))
                                * (getCoordY(i,sizeY) - getCoordY(i-1,sizeY)));
                        kCentral -= k;
                        tmpResXi += k * u[i-1][j].Xi;
                        tmpResEta += k * u[i-1][j].Eta;
                        //std::cout<<k<<" ";

                        // (i+1, j)
                        k = 1/( (getCoordY(i+1,sizeY) - getCoordY(i-1,sizeY))
                                * (rhoElement(getCoordX(j,sizeX),getCoordY(i+1,sizeY)) + rhoElement(getCoordX(j,sizeX),getCoordY(i,sizeY)))
                                * (getCoordY(i+1,sizeY) - getCoordY(i,sizeY)));
                        kCentral -= k;
                        tmpResXi += k * u[i+1][j].Xi;
                        tmpResEta += k * u[i+1][j].Eta;
                        //std::cout<<k<<" ";

                        tmpResXi += kCentral * u[i][j].Xi;
                        tmpResEta += kCentral * u[i][j].Eta;
                        //std::cout<<kCentral<<std::endl;

                        // L2 norm
                        residualXi += tmpResXi*tmpResXi;
                        residualEta += tmpResEta*tmpResEta;

                        // Maximum norm
                        if(fabs(tmpResXi)>residualXiMax) residualXiMax = fabs(tmpResXi);
                        if(fabs(tmpResEta)>residualEtaMax) residualEtaMax = fabs(tmpResEta);

                        if(flag[i][j]>0) ++count;
                        //++count;
                        rXi[i][j] = fabs(tmpResXi);
                        rEta[i][j] = fabs(tmpResEta);
                        //std::cout<<"("<<rXi[i][j]<<", "<<rEta[i][j]<<") ";
                    }
                    else {
                        rXi[i][j]=0;
                        rEta[i][j]=0;
                    }
                }
                //std::cout<<std::endl;
            }

            residualXi = sqrt(residualXi/count);
            residualEta = sqrt(residualEta/count);
        }
    }

    MPI::COMM_WORLD.Barrier();
    std::string str_id="_"+std::to_string(size)+"_"+std::to_string(subDomainX)+"_"
                        +std::to_string(subDomainY)+"_"+std::to_string(locsizeX)+"_"
                        +std::to_string(locsizeY)+"_"+std::to_string(interpolation_step)+"_"
                        +std::to_string(walks_per_processor);

    t4 = MPI_Wtime();

    for (int i = 0 ; i < sizeY; ++i)
        delete []flag[i];
    delete []flag;

    MPI::COMM_WORLD.Barrier();
    // print residuals for Xi on file "rXi", residuals for Eta on file "rEta".
    if ((my_id == 0) && (print_residuals)){
        if (!calc_residuals)
        {
            std::cout<<"Set -calc_residuals to true to print residuals"<<std::endl;
        }
        else {
            std::ofstream outputFile1;
            outputFile1.open("rXi"+str_id);
            //std::cout<<"rXi"<<std::endl;
            for(int i=0; i<sizeY; ++i){
                for(int j=0; j<sizeX; ++j){
                    //std::cout<<u[i][j]<<" ";
                    outputFile1 << rXi[i][j]<<" ";
                }
                //std::cout<<std::endl;
                outputFile1 <<std::endl;
            }
            outputFile1.close();
            std::ofstream outputFile2;
            outputFile2.open("rEta"+str_id);
            //std::cout<<"rEta"<<std::endl;
            for(int i=0; i<sizeY; ++i){
                for(int j=0; j<sizeX; ++j){
                    //std::cout<<u[i][j]<<" ";
                    outputFile2 << rEta[i][j]<<" ";
                }
                //std::cout<<std::endl;
                outputFile2 <<std::endl;
            }
            outputFile2.close();
        }
    }

    if ((my_id == 0) && (calc_residuals)){
        for(int i=0; i<sizeY; ++i){
            delete []rXi[i];
            delete []rEta[i];
        }
        delete []rXi;
        delete []rEta;
    }

    //print solution for Xi on file "Xi", solution for Eta on file "Eta".
    if ((my_id == 0) && (print_results)){
        std::ofstream outputFile1;
        outputFile1.open("Xi"+str_id);
        //std::cout<<"Xi"<<std::endl;
        for(int i=0; i<sizeY; ++i){
            for(int j=0; j<sizeX; ++j){
                //std::cout<<u[i][j]<<" ";
                outputFile1 << u[i][j].Xi<<" ";
            }
            //std::cout<<std::endl;
            outputFile1 <<std::endl;
        }
        outputFile1.close();
        std::ofstream outputFile2;
        outputFile2.open("Eta"+str_id);
        //std::cout<<"Eta"<<std::endl;
        for(int i=0; i<sizeY; ++i){
            for(int j=0; j<sizeX; ++j){
                //std::cout<<u[i][j]<<" ";
                outputFile2 << u[i][j].Eta<<" ";
            }
            //std::cout<<std::endl;
            outputFile2 <<std::endl;
        }
        outputFile2.close();
    }

    // printing timing
    t5 = MPI_Wtime();
    if (my_id == 0)
    {
        std::ofstream outputFile3;
        outputFile3.open("time"+str_id);
        outputFile3<<"# *********************************************************************"<<std::endl;
        outputFile3<<"# Overview"<<std::endl;
        outputFile3<<"# Number of processors = "<<size<<std::endl;
        outputFile3<<"# Number of x-subdomains = "<<subDomainX<<std::endl;
        outputFile3<<"# Number of y-subdomains = "<<subDomainY<<std::endl;
        outputFile3<<"# Number of points per each x-subdomain direction = "<<locsizeX<<std::endl;
        outputFile3<<"# number of points per each y-subdomain direction = "<<locsizeY<<std::endl;
        outputFile3<<"# Number of points we interopolate in between = "<<interpolation_step<<std::endl;
        outputFile3<<"# Number of random walks per mesh node = "<<walks_per_processor<<std::endl;
        outputFile3<<"# *********************************************************************"<<std::endl;
        outputFile3<<std::endl;
        outputFile3<<"# Timing. "<<std::endl;
        outputFile3<<std::endl;
        outputFile3<<"# Time: creation of flag  "<<std::endl<<t1-t0<<std::endl;
        outputFile3<<"# Time: parallel part of Monte Carlo "<<std::endl<<t2-t1<<std::endl;
        outputFile3<<"# Time: send-receive of partial solution and computation of final average "<<std::endl<<t3-t2<<std::endl;
        outputFile3<<"# Time: petsc solver "<<std::endl<<t4-t3<<std::endl;
        outputFile3<<"# Time: creation of output files "<<std::endl<<t5-t4<<std::endl;
        if (calc_residuals){
            outputFile3<<"# L2 norm residual: Xi"<<std::endl<<residualXi<<std::endl<<"# L2 norm residual: Eta"<<std::endl<<residualEta<<std::endl;
            outputFile3<<"# Maximum norm residual: Xi"<<std::endl<<residualXiMax<<std::endl<<"# Maximum norm residual: Eta"<<std::endl<<residualEtaMax<<std::endl;
        }
        outputFile3<<"# *********************************************************************"<<std::endl;
        outputFile3.close();
    }

    MPI::Finalize();
    return 0;
}
#undef __FUNCT__

#define __FUNCT__ "testRho"
PetscErrorCode testRho(const WinslowMeshGenerator& g)
{
    PetscErrorCode      ierr;

    ierr = directRhoComputation<>(g, rhoElement);CHKERRQ(ierr);
    return 0;
}
#undef __FUNCT__