#include <iostream>
#include <fstream>
#include <cmath>
#include "Random64.h"
#include "cuda_bf16.h"
using namespace std;

#define Lx 300
#define Ly 200
#define Lz 40
#define N 32 //Threads per Block
const int NCells=Lx*Ly*Lz;
const int M=(NCells+N-1)/N; //Blocks per Grid
#define Q 7
const int ArraySize=Lx*Ly*Lz*Q;
const int Vol=Lx*Ly*Lz;

const float W0=1.0/4;

const float C=0.5; // C<0.707 cells/click
const float C2=C*C;
const float AUX0=1-4*C2*(1-W0);

const float tau=0.5;
const float Utau=1.0/tau;
const float UmUtau=1-Utau;

//------------------------------------------------------
//------------ PROGRAMMING ON THE DEVICE ----------------
//---------------Constants (Symbols)----------------
__constant__ float d_w[7];
__constant__ int d_Vx[7];
__constant__ int d_Vy[7];
__constant__ int d_Vz[7];
__constant__ float d_C[3];   // d_C[0]=C,  d_C[1]=C2,  d_C[2]=AUX0, 
__constant__ float d_tau[3]; // d_tau[0]=tau,  d_tau[1]=Utau,  d_tau[2]=UmUtau, 

//----------Functions called by the device itself (__device__)
//data index
__device__ int d_n(int ix,int iy,int iz,int i){
  return ((ix*Ly+iy)*Lz+iz)*Q+i;
}
//macroscopic fields
__device__ float d_rho(int ix,int iy,int iz,float *d_f){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,iz,i); sum+=d_f[n0];
  }
  return sum;
}
__device__ float d_rhonew(int ix,int iy,int iz,float *d_fnew){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,iz,i); sum+=d_fnew[n0];
  }
  return sum;
}
__device__ float d_Jx(int ix,int iy,int iz,float *d_f){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,iz,i); sum+=d_Vx[i]*d_f[n0];
  }
  return sum;
}  
__device__ float d_Jy(int ix,int iy,int iz,float *d_f){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,iz,i); sum+=d_Vy[i]*d_f[n0];
  }
  return sum;
}
__device__ float d_Jz(int ix,int iy,int iz,float *d_f){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,iz,i); sum+=d_Vz[i]*d_f[n0];
  }
  return sum;
}
//equilibrium functions
__device__ float d_feq(float rho0,float Jx0,float Jy0,float Jz0,int i){
  return 4*d_w[i]*(d_C[1]*rho0+d_Vx[i]*Jx0+d_Vy[i]*Jy0+d_Vz[i]*Jz0);
}  
__device__ float d_f0eq(float rho0){
  return rho0*d_C[2];
}  
//---------------------KERNELS----------------------------
__global__ void d_Collision(float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,iz,i,n0;  float rho0,Jx0,Jy0,Jz0;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/(Ly*Lz); iy=(icell/Lz)%Ly; iz=icell%Lz;
  //Compute the macroscopic fields
  rho0=d_rho(ix,iy,iz,d_f); //rho
  Jx0=d_Jx(ix,iy,iz,d_f);   //Jx0
  Jy0=d_Jy(ix,iy,iz,d_f);   //Jy0
  Jz0=d_Jz(ix,iy,iz,d_f);   //Jz0
  //Collide and compute fnew
  n0=d_n(ix,iy,iz,0); d_fnew[n0]=d_tau[2]*d_f[n0]+d_tau[1]*d_f0eq(rho0);
  for(i=1;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,iz,i); d_fnew[n0]=d_tau[2]*d_f[n0]+d_tau[1]*d_feq(rho0,Jx0,Jy0,Jz0,i);
  }
}
__global__ void d_BB_xwalls(int x,int y,int z,int lz,float D,float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,iz,i,n0,n0f;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=x; iy=icell/lz+y; iz=icell%lz+z;
  //Compute the bounce back conditions
  for(i=1;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,iz,i); n0f=d_n(ix,iy,iz,(i-2*((int) floorf((i+1)/2)-1))%2+2*((int) floorf((i+1)/2)-1)+1);
    d_fnew[n0f]=d_f[n0]*D;
  }
}
__global__ void d_BB_ywalls(int x,int y,int z,int lz,float D,float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,iz,i,n0,n0f;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/lz+x; iy=y; iz=icell%lz+z;
  //Compute the bounce back conditions
  for(i=1;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,iz,i); n0f=d_n(ix,iy,iz,(i-2*((int) floorf((i+1)/2)-1))%2+2*((int) floorf((i+1)/2)-1)+1);
    d_fnew[n0f]=d_f[n0]*D;
  }
}
__global__ void d_BB_zwalls(int x,int y,int z,int ly,float D,float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,iz,i,n0,n0f;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/ly+x; iy=icell%ly+y; iz=z;
  //Compute the bounce back conditions
  for(i=1;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,iz,i); n0f=d_n(ix,iy,iz,(i-2*((int) floorf((i+1)/2)-1))%2+2*((int) floorf((i+1)/2)-1)+1);
    d_fnew[n0f]=d_f[n0]*D;
  }
}
__global__ void d_ImposeFields(float *d_f,float *d_fnew,float RhoSource,int ix,int iy,int iz){
  //Define internal registers
  int i,n0;  float rho0,Jx0,Jy0,Jz0;
  //Compute the macroscopic fields
  rho0=RhoSource; //rho
  Jx0=d_Jx(ix,iy,iz,d_f);   //Jx0
  Jy0=d_Jy(ix,iy,iz,d_f);   //Jy0
  Jz0=d_Jz(ix,iy,iz,d_f);   //Jz0
  //Collide and compute fnew
  n0=d_n(ix,iy,iz,0); d_fnew[n0]=d_f0eq(rho0);
  for(i=1;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,iz,i); d_fnew[n0]=d_feq(rho0,Jx0,Jy0,Jz0,i);
  }
}
__global__ void d_Advection(float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,iz,i,ixnext,iynext,iznext,n0,n0next;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/(Ly*Lz); iy=(icell/Lz)%Ly; iz=icell%Lz;
  //Move the contents to the neighboring cells
  for(i=0;i<Q;i++){ //on each direction
    ixnext=(ix+d_Vx[i]+Lx)%Lx; iynext=(iy+d_Vy[i]+Ly)%Ly; iznext=(iz+d_Vz[i]+Lz)%Lz;//periodic boundaries
    n0=d_n(ix,iy,iz,i); n0next=d_n(ixnext,iynext,iznext,i);
    d_f[n0next]=d_fnew[n0]; 
  }
}
__global__ void d_MaxMinRho(float *d_maxrho,float *d_minrho,float *d_fnew){
  double rho0; int ix, iy, iz;
  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){
      for(iz=0;iz<Lz;iz++){
        //if(ix!=Lz/2 && iy!=Ly/2 && iz!=Lz/2){
          rho0=d_rhonew(ix,iy,iz,d_fnew);
          if(rho0>=d_maxrho[((Ly*ix+iy)*Lz+iz)]){d_maxrho[((Ly*ix+iy)*Lz+iz)]=rho0;}
          else if(rho0<=d_minrho[((Ly*ix+iy)*Lz+iz)]){d_minrho[((Ly*ix+iy)*Lz+iz)]=rho0;}
        //}
      }
    }
  }
}
__global__ void dfill_Pow(float *d_maxrho,float *d_minrho,float *d_Pow){
  //Define internal registers
  int icell,ix,iy,iz; float A,P;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/(Ly*Lz); iy=(icell/Lz)%Ly; iz=icell%Lz;
  //take the amplitude and power
  A=(d_maxrho[((Ly*ix+iy)*Lz+iz)]-d_minrho[((Ly*ix+iy)*Lz+iz)])/2.0;
  P=A*A;
  d_Pow[((Ly*ix+iy)*Lz+iz)]=P;
}
__global__ void dT_Pow(float *d_maxrho,float *d_minrho,float *d_Result,float *d_Pow){
  //Define internal registers
  int icell,ix,iy,iz;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/(Ly*Lz); iy=(icell/Lz)%Ly; iz=icell%Lz;
  
  atomicAdd(&d_Result[0],d_Pow[((Ly*ix+iy)*Lz+iz)]);
}
//------------------------------------------------------
//------------ PROGRAMMING ON THE HOST ----------------
//--------------------- Clase LatticeBoltzmann ------------
class LatticeBoltzmann{
private:
  float h_C[3];   // h_C[0]=C,  h_C[1]=C2,  h_C[2]=AUX, 
  float h_tau[3]; // h_tau[0]=tau,  h_tau[1]=Utau,  h_tau[2]=UmUtau, 
  float h_w[Q];      //Weights 
  int h_Vx[Q],h_Vy[Q],h_Vz[Q];  //Velocity vectors
  float *h_f, *h_fnew;  float *d_f,*d_fnew; //Distribution Functions
  float *h_maxrho, *h_minrho; float *d_maxrho, *d_minrho; //presure matrixes
  float *h_Result; float *d_Result;//Arrays for power measurement
  float *h_Pow; float *d_Pow;
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  int h_n(int ix,int iy,int iz,int i){return ((ix*Ly+iy)*Lz+iz)*Q+i;};
  float h_rho(int ix,int iy,int iz);
  float h_Jx(int ix,int iy,int iz);
  float h_Jy(int ix,int iy,int iz);
  float h_Jz(int ix,int iy,int iz);
  float h_feq(float rho0,float Jx0,float Jy0,float Jz0,int i);
  void Start(float rho0,float Jx0,float Jy0,float Jz0);
  void Collision(void);
  void BB_xwalls(int x,int y,int z,int ly,int lz,float D);
  void BB_ywalls(int x,int y,int z,int lx,int lz,float D);
  void BB_zwalls(int x,int y,int z,int lx,int ly,float D);
  void ImposeFields(int N_source,float *Source,int t);
  void Advection(void);
  void MaxMinRho(void);
  float hT_Pow(void);
  void ResetMaxMinRho(void);
  void Print(const char * NameFile);

};
LatticeBoltzmann::LatticeBoltzmann(void){
  //CONSTANTS(d_Symbols)
  //---Charge constantes on the Host-----------------
  //running constants
  h_C[0]=C;  h_C[1]=C2;  h_C[2]=AUX0;
  h_tau[0]=tau;  h_tau[1]=Utau;  h_tau[2]=UmUtau;
  //Set the weights
  h_w[0]=W0; h_w[1]=h_w[2]=h_w[3]=h_w[4]=h_w[5]=h_w[6]=W0/2;
  //Set the velocity vectors
  h_Vx[0]=0;  h_Vx[1]=1;  h_Vx[2]=-1; h_Vx[3]=0; h_Vx[4]=0;  h_Vx[5]=0; h_Vx[6]=0;
  h_Vy[0]=0;  h_Vy[1]=0;  h_Vy[2]=0;  h_Vy[3]=1; h_Vy[4]=-1; h_Vy[5]=0; h_Vy[6]=0;
  h_Vz[0]=0;  h_Vz[1]=0;  h_Vz[2]=0;  h_Vz[3]=0; h_Vz[4]=0;  h_Vz[5]=1; h_Vz[6]=-1;
  //------Send to the Device-----------------
  cudaMemcpyToSymbol(d_w,h_w,Q*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vx,h_Vx,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vy,h_Vy,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vz,h_Vz,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_C,h_C,3*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_tau,h_tau,3*sizeof(float),0,cudaMemcpyHostToDevice);
  
  //DISTRIBUTION FUNCTIONS
  //Build the dynamic matrices on the host
  h_f=new float [ArraySize];  h_fnew=new float [ArraySize];
  h_maxrho=new float[Vol];    h_minrho=new float[Vol];
  h_Result=new float[1];      h_Pow=new float[Vol];
  //Build the dynamic matrices on the device
  cudaMalloc((void**) &d_f,ArraySize*sizeof(float));
  cudaMalloc((void**) &d_fnew,ArraySize*sizeof(float));
  cudaMalloc((void**) &d_maxrho,Vol*sizeof(float));
  cudaMalloc((void**) &d_minrho,Vol*sizeof(float));
  cudaMalloc((void**) &d_Result,sizeof(float));
  cudaMalloc((void**) &d_Pow,Vol*sizeof(float)); 
}
LatticeBoltzmann::~LatticeBoltzmann(void){
  delete[] h_f;       delete[] h_fnew;
  delete[] h_maxrho;  delete[] h_minrho;
  delete[] h_Result;  delete[] h_Pow;
  cudaFree(d_f);      cudaFree(d_fnew);
  cudaFree(d_maxrho); cudaFree(d_minrho);
  cudaFree(d_Result); cudaFree(d_Pow);
}
float LatticeBoltzmann::h_rho(int ix,int iy,int iz){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,iz,i); sum+=h_fnew[n0];
  }
  return sum;
}  
float LatticeBoltzmann::h_Jx(int ix,int iy,int iz){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,iz,i); sum+=h_Vx[i]*h_fnew[n0];
  }
  return sum;
}  
float LatticeBoltzmann::h_Jy(int ix,int iy,int iz){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,iz,i); sum+=h_Vy[i]*h_fnew[n0];
  }
  return sum;
}
float LatticeBoltzmann::h_Jz(int ix,int iy,int iz){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,iz,i); sum+=h_Vz[i]*h_fnew[n0];
  }
  return sum;
}
float  LatticeBoltzmann::h_feq(float rho0,float Jx0,float Jy0,float Jz0,int i){
  if(i>0)
    return 4*h_w[i]*(h_C[1]*rho0+h_Vx[i]*Jx0+h_Vy[i]*Jy0+h_Vz[i]*Jz0);
  else
    return rho0*AUX0;
}  
void LatticeBoltzmann::Start(float rho0,float Jx0,float Jy0,float Jz0){
  //Charge on the Host
  int ix,iy,iz,i,n0;
  for(ix=0;ix<Lx;ix++){ //for each cell
    for(iy=0;iy<Ly;iy++){
      for(iz=0;iz<Lz;iz++){
        for(i=0;i<Q;i++){ //on each direction
          n0=h_n(ix,iy,iz,i); h_f[n0]=h_feq(rho0,Jx0,Jy0,Jz0,i);
        }
      }
    }
  }  
  //Send to the Device
  cudaMemcpy(d_f,h_f,ArraySize*sizeof(float),cudaMemcpyHostToDevice);
}  
void LatticeBoltzmann::Collision(void){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Collision<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew);
  
    //Code Backup
    //Z:
    BB_zwalls(0, 0, 0, 300, 200, 0.99); //Piso
    BB_zwalls(0, 0, 20, 300, 40,  0.98); //Pared
    BB_zwalls(0, 0, 40, 300, 200, 0.70); //Techo
    //Y:
    BB_ywalls(0, 0, 0, 132, 40, 0.98);  //Pared
    BB_ywalls(132, 0, 20, 40, 20,0.98);  //Pared
    BB_ywalls(172, 0, 0, 128, 40, 0.98);  //Pared
    BB_ywalls(132, 0, 0, 40, 20,0);  //Entrada
    BB_ywalls(0, 200, 0, 300, 30,0.98);  //Pared
    BB_ywalls(0, 200, 30, 300, 10, 0.91 ); //Vidrio
    //X:
    BB_xwalls(0, 0, 0, 200, 40,0.98); //Pared
    BB_xwalls(300, 0, 0, 200, 40,0.98); //Pared
    //1:
    BB_ywalls(0, 41, 0, 39, 40,0.98); //Pared
    BB_xwalls(39, 41, 0, 1, 40,0.98); //Pared
    BB_ywalls(39, 42, 0, 3, 40,0.98); //Pared
    BB_xwalls(42, 0, 0, 41, 20,0.98); //Pared
    BB_xwalls(42, 41, 0, 1, 40,0.98); //Pared
    //2:
    BB_xwalls(76, 34, 0, 7, 20, 0.98); //Pared
    BB_xwalls(86, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(89, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(89, 34, 0, 7, 20, 0.98); //Pared
    BB_ywalls(76, 34, 0, 13, 20, 0.98); //Pared
    BB_ywalls(76, 41, 0, 10, 40, 0.98); //Pared
    BB_ywalls(86, 42, 0, 3, 40, 0.98); //Pared
    //3:
    BB_xwalls(129, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(132, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(129, 41, 1, 1, 40, 0.98); //Pared
    BB_xwalls(132, 41, 1, 1, 40, 0.98); //Pared
    BB_ywalls(129, 42, 1, 3, 40, 0.98); //Pared
    //4.
    BB_xwalls(172, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(175, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(172, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(175, 41, 0, 1, 40, 0.98); //Pared
    BB_ywalls(172, 42, 0, 3, 40, 0.98); //Pared
    //5.
    BB_xwalls(215, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(218, 0, 0, 34, 20, 0.98); //Pared
    BB_xwalls(215, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(218, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(228, 34, 0, 7, 20, 0.98); //Pared
    BB_ywalls(215, 42, 0, 3, 40, 0.98); //Pared
    BB_ywalls(218, 41, 0, 10, 40, 0.98); //Pared
    BB_ywalls(218, 34, 0, 10, 20, 0.98); //Pared
    //6:
    BB_xwalls(262, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(265, 0, 0, 41, 20, 0.98); //Pared
    BB_xwalls(262, 41, 0, 1, 40, 0.98); //Pared
    BB_xwalls(265, 41, 0, 1, 40, 0.98); //Pared
    BB_ywalls(262, 42, 0, 3, 40, 0.98); //Pared
    //7:
    BB_xwalls(39, 119, 0, 3, 40, 0.98 ); //Pared
    BB_xwalls(42, 119, 0, 3, 40, 0.98 ); //Pared
    BB_ywalls(39, 119, 0, 3, 40, 0.98 ); //Pared
    BB_ywalls(39, 122, 0, 3, 40, 0.98 ); //Pared
    //8:
    BB_xwalls(82, 115, 0, 7, 40, 0.98 ); //Pared
    BB_xwalls(89, 115, 0, 7, 40, 0.98 ); //Pared
    BB_ywalls(82, 115, 0, 7, 40, 0.98 ); //Pared
    BB_ywalls(82, 122, 0, 7, 40, 0.98 ); //Pared
    //9:
    BB_xwalls(129, 119, 0, 3, 40, 0.98 ); //Pared
    BB_xwalls(132, 119, 0, 3, 40, 0.98 ); //Pared
    BB_ywalls(129, 119, 0, 3, 40, 0.98 ); //Pared
    BB_ywalls(129, 122, 0, 3, 40, 0.98 ); //Pared
    //10
    BB_xwalls(172, 119, 0, 3, 40,0.98); //Pared
    BB_xwalls(175, 119, 0, 3, 40,0.98); //Pared
    BB_ywalls(172, 119, 0, 3, 40,0.98); //Pared
    BB_ywalls(172, 122, 0, 3, 40,0.98); //Pared
    //11
    BB_xwalls(215, 115, 0, 7, 40,0.98); //Pared
    BB_xwalls(222, 115, 0, 7, 40,0.98); //Pared
    BB_ywalls(215, 115, 0, 7, 40,0.98); //Pared
    BB_ywalls(215, 122, 0, 7, 40,0.98); //Pared
    //12
    BB_xwalls(262, 119, 0, 3, 40,0.98); //Pared
    BB_xwalls(265, 119, 0, 3, 40,0.98); //Pared
    BB_ywalls(262, 119, 0, 3, 40,0.98); //Pared
    BB_ywalls(262, 122, 0, 3, 40,0.98); //Pared
    //13
    BB_xwalls(39, 197, 0, 3, 40,0.98); //Pared
    BB_xwalls(42, 197, 0, 3, 40,0.98); //Pared
    BB_ywalls(39, 197, 0, 3, 40,0.98); //Pared
    //14
    BB_xwalls(86, 197, 0, 3, 40,0.98); //Pared
    BB_xwalls(89, 197, 0, 3, 40,0.98); //Pared
    BB_ywalls(86, 197, 0, 3, 40,0.98); //Pared
    //15
    BB_xwalls(129, 197, 0, 3, 40, 0.98); //Pared
    BB_xwalls(132, 197, 0, 3, 40, 0.98); //Pared
    BB_ywalls(129, 197, 0, 3, 40, 0.98); //Pared
    //16
    BB_xwalls(172, 197, 0, 3, 40, 0.98); //Pared
    BB_xwalls(175, 197, 0, 3, 40, 0.98); //Pared
    BB_ywalls(172, 197, 0, 3, 40, 0.98); //Pared
    //17
    BB_xwalls(219, 197, 0, 3, 40, 0.98); //Pared
    BB_xwalls(222, 197, 0, 3, 40, 0.98); //Pared
    BB_ywalls(219, 197, 0, 3, 40, 0.98); //Pared
    //18
    BB_xwalls(262, 197, 0, 3, 40, 0.98); //Pared
    BB_xwalls(265, 197, 0, 3, 40, 0.98); //Pared
    BB_ywalls(262, 197, 0, 3, 40, 0.98); //Pared
    //Cajas segundo piso
    BB_ywalls(42, 41, 20, 34, 20, 0.98); //Pared
    BB_ywalls(89, 41, 20, 40, 20, 0.98); //Pared
    BB_ywalls(132, 41, 20, 40, 20, 0.98); //Pared
    BB_ywalls(175, 41, 20, 40, 20, 0.98); //Pared
    BB_ywalls(228, 41, 20, 34, 20, 0.98); //Pared
    BB_ywalls(265, 41, 20, 35, 20, 0.98); //Pared
    //This is a line of Junk Code
}
void LatticeBoltzmann::BB_xwalls(int x,int y,int z,int ly,int lz,float D){
  //Here we impose bounceback on wall (x=0)->that is a ly*lz array to acces and startin on cell x,y,z
  int n=ly*lz; //total cells for the wall
  int m=(n+N-1)/N;
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(m,1,1);
  d_BB_xwalls<<<BlocksPerGrid,ThreadsPerBlock>>>(x,y,z,lz,D,d_f,d_fnew);
}
void LatticeBoltzmann::BB_ywalls(int x,int y,int z,int lx,int lz,float D){
  //Here we impose bounceback on the left wall (x=0)->that is a 128*128 array to acces
  int n=lx*lz; //total cells for the wall
  int m=(n+N-1)/N;
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(m,1,1);
  d_BB_ywalls<<<BlocksPerGrid,ThreadsPerBlock>>>(x,y,z,lz,D,d_f,d_fnew);
}
void LatticeBoltzmann::BB_zwalls(int x,int y,int z,int lx,int ly,float D){
  //Here we impose bounceback on the left wall (x=0)->that is a 128*128 array to acces
  int n=lx*ly; //total cells for the wall
  int m=(n+N-1)/N;
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(m,1,1);
  d_BB_zwalls<<<BlocksPerGrid,ThreadsPerBlock>>>(x,y,z,ly,D,d_f,d_fnew);
}
void LatticeBoltzmann::ImposeFields(int N_source,float *Source,int t){
  dim3 ThreadsPerBlock(1,1,1); //A single thread (in this case)
  dim3 BlocksPerGrid(1,1,1);
  float lambda=10,omega=2*M_PI/lambda*C,fase;
  int ix,iy,iz; 
  for(int p=0;p<N_source;p++){ //Para esta función, se toma la información de las fuentes del array Source, que contiene las
    ix=Source[4*p];            //posiciones y fases de las fuentes y ejecuta el imposefield sobre cada una de ellas.
    iy=Source[4*p+1];
    iz=Source[4*p+2];
    fase=Source[4*p+3];
    float RhoSource=10*sin(omega*t+fase);
    d_ImposeFields<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew,RhoSource,ix,iy,iz);
  }
}
void LatticeBoltzmann::Advection(void){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Advection<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew);
}
void LatticeBoltzmann::MaxMinRho(void){
  
  //Do everything on the Device
  dim3 ThreadsPerBlock(1,1,1);
  dim3 BlocksPerGrid(1,1,1);
  d_MaxMinRho<<<BlocksPerGrid,ThreadsPerBlock>>>(d_maxrho,d_minrho,d_fnew);
}
float LatticeBoltzmann::hT_Pow(void){
  h_Result[0]=0.0;
  cudaMemcpy(d_Result,h_Result,sizeof(float),cudaMemcpyHostToDevice);
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  dfill_Pow<<<BlocksPerGrid,ThreadsPerBlock>>>(d_maxrho,d_minrho,d_Pow);
  dT_Pow<<<BlocksPerGrid,ThreadsPerBlock>>>(d_maxrho,d_minrho,d_Result,d_Pow);
  //return to host
  cudaMemcpy(h_Result,d_Result,sizeof(float),cudaMemcpyDeviceToHost);
  
  return h_Result[0];
}
void LatticeBoltzmann::ResetMaxMinRho(void){
  for(int i=0;i<Vol;i++){
    h_maxrho[i]=0.0;
    h_minrho[i]=0.0;
  }
  cudaMemcpy(d_maxrho,h_maxrho,Vol*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_minrho,h_minrho,Vol*sizeof(float),cudaMemcpyHostToDevice);
}
void LatticeBoltzmann::Print(const char * NameFile){
  ofstream MyFile(NameFile); float rho0; int ix,iy,iz;
  //Bring back the data from Device to Host
  cudaMemcpy(h_fnew,d_fnew,ArraySize*sizeof(float),cudaMemcpyDeviceToHost);
  //Print for gnuplot splot  
  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){
      for(iz=0;iz<Lz;iz++){
        rho0=h_rho(ix,iy,iz);
        MyFile<<ix<<" "<<iy<<" "<<iz<<" "<<rho0<<endl;
      }
      MyFile<<endl;
    }
    MyFile<<endl;
  }
  MyFile.close();
}

/*---Funciones extras---*/
/*función de fuente: Dado un numero de fuentes y 4 características: ix,iy,iz y fase,
  se rellena el array con valores random.*/
void RanSource(int N_source, float *Source,int x,int y,int z,int lx,int ly,int lz,Crandom &Ran64){
  double Aux;

  for(int n=0;n<N_source;n++){
    Aux=Ran64.r()*lx+x;
    Source[4*n]=(int) Aux;
    Aux=Ran64.r()*ly+y;
    Source[4*n+1]=(int) Aux;
    Aux=Ran64.r()*lz+z;
    Source[4*n+2]=(int) Aux;
    Aux=Ran64.r()*2*M_PI;
    Source[4*n+3]=Aux;
  }
}

void Print_sources(int N_source,float *Source){
  for(int n=0;n<N_source;n++){
    for(int i=0;i<3;i++){
      cout<<Source[4*n+i]<<" ";
    }
    cout<<"\n";
  }
}

//------------------- Funciones Globales ------------

int main(void){
  LatticeBoltzmann Ondas;
  Crandom Ran64(1);
  int t,tp,tsaturado=1000, tmax=2500,N_source=8;
  float rho0=0,Jx0=0,Jy0=0,Jz0=0;
  float Source1[4*N_source];
  float Source2[4*N_source];
    
  //initialize the random sources
  RanSource(N_source,Source1,1,60,1,298,50,38,Ran64);
  RanSource(N_source,Source2,1,170,1,298,20,38,Ran64);
  
  cout<<"Sources positions: \n";
  Print_sources(N_source,Source1);
  Print_sources(N_source,Source2);

  //Start
  Ondas.Start(rho0,Jx0,Jy0,Jz0);
  //Run
  ofstream MyFile("Pot.dat");
  for(tp=0,t=0;t<tmax;t++,tp++){
    Ondas.Collision();
    if(t <= tsaturado){
        Ondas.ImposeFields(N_source,Source1,t);
        Ondas.ImposeFields(N_source,Source2,t);
    }
    Ondas.Advection();
    Ondas.MaxMinRho();
    if(tp>25){
      MyFile<<t<<" "<<Ondas.hT_Pow()<<"\n";
      tp=0;
      cout<<"Simulation: "<<(1.0*t)/(1.0*tmax)*100<<" %"<<"\n";
      Ondas.ResetMaxMinRho();
    }
  }
  MyFile.close();
  //Print
  Ondas.Print("Ondas.dat");

  return 0;
}
