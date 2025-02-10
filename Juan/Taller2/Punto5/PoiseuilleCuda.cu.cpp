//Poiseuille on Cuda
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
using namespace std;

#define Lx 1
#define Ly 64
#define N 32 //Threads per Block
const int NCells=Lx*Ly;
const int M=(NCells+N-1)/N; //Blocks per Grid
#define Q 9
const int ArraySize=Lx*Ly*Q;

const float tau=1.2;
const float Utau=1.0/tau;
const float UmUtau=1-Utau;
const float ThreeUmU2tau=3*(1-1/(2*tau));


//------------------------------------------------------
//------------ PROGRAMMING ON THE DEVICE ----------------
//---------------Constants (Symbols)----------------
__constant__ float d_w[9];
__constant__ int d_Vx[9];
__constant__ int d_Vy[9];
__constant__ float d_tau[4]; // d_tau[0]=tau,  d_tau[1]=Utau,   
			                      // d_tau[2]=UmUtau,d_tau[3]=ThreeUmU2tau

//----------Functions called by the device itself (__device__)
//data index
__device__ int d_n(int ix,int iy,int i){
  return (ix*Ly+iy)*Q+i;
}

//macroscopic fields
__device__ float d_rho(int ix,int iy,float *d_f){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_f[n0];
  }
  return sum;
}
__device__ float d_Jx(int ix,int iy,float *d_f,float d_Fx){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_Vx[i]*d_f[n0];
  }
  return sum+0.5*d_Fx;
}
__device__ float d_Jy(int ix,int iy,float *d_f,float d_Fy){
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_Vy[i]*d_f[n0];
  }
  return sum+0.5*d_Fy;
}
__device__ float d_Fi(float d_Ux0,float d_Uy0,float d_Fx,float d_Fy,int i){
  float d_UdotVi=d_Ux0*d_Vx[i]+d_Uy0*d_Vy[i];
  float d_FdotVi=d_Fx*d_Vx[i]+d_Fy*d_Vy[i];
  float d_UdotF=d_Ux0*d_Fx+d_Uy0*d_Fy;
  return d_tau[3]*d_w[i]*(d_FdotVi-d_UdotF+3*d_UdotVi*d_FdotVi);  
}

//equilibrium functions
__device__ float d_feq(float d_rho0,float d_Ux0,float d_Uy0,int i){
  double d_UdotVi=d_Ux0*d_Vx[i]+d_Uy0*d_Vy[i], d_U2=d_Ux0*d_Ux0+d_Uy0*d_Uy0;
  return d_rho0*d_w[i]*(1+3*d_UdotVi+4.5*d_UdotVi*d_UdotVi-1.5*d_U2);
}  


//---------------------KERNELS----------------------------
__global__ void d_Collision(float *d_f,float *d_fnew,float d_gx, float d_gy){
  //Define internal registers
  int icell,ix,iy,i,n0;  float rho0,Ux0,Uy0,d_Fx,d_Fy;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/Ly; iy=icell%Ly;
  //Compute the macroscopic fields
  rho0=d_rho(ix,iy,d_f); //rho							
   d_Fx=d_gx*rho0; 
   d_Fy=d_gy*rho0;
   Ux0=d_Jx(ix,iy,d_f,d_Fx)/rho0; //Ux0
   Uy0=d_Jy(ix,iy,d_f,d_Fy)/rho0; //Uy0
  //Collide and compute fnew                                                     
  for(i=0;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,i); d_fnew[n0]=d_tau[2]*d_f[n0]+d_tau[1]*d_feq(rho0,Ux0,Uy0,i)+d_Fi(Ux0,Uy0,d_Fx,d_Fy,i); 
  }
}
__global__ void d_ImposeFields(float *d_f,float *d_fnew){     
 //Define internal registers
  int ix,iy,i,n0;  float rho0;
  iy=0;//Lower Wall
  for(ix=0;ix<Lx;ix++){
	rho0=d_rho(ix,iy,d_f);
	for(i=0;i<Q;i++){n0=d_n(ix,iy,i);d_fnew[n0]=d_feq(rho0,0,0, i);} 
  }
  
  iy=Ly-1; //UpperWall 
  for(ix=0;ix<Lx;ix++){
	rho0=d_rho(ix,iy,d_f);
	for(i=0;i<Q;i++){n0=d_n(ix,iy,i);d_fnew[n0]=d_feq(rho0,0,0, i);} 
  }
}

__global__ void d_Advection(float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,i,ixnext,iynext,n0,n0next;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/Ly; iy=icell%Ly;
  //Move the contents to the neighboring cells
  for(i=0;i<Q;i++){ //on each direction
    ixnext=(ix+d_Vx[i]+Lx)%Lx; iynext=(iy+d_Vy[i]+Ly)%Ly;//periodic boundaries
    n0=d_n(ix,iy,i); n0next=d_n(ixnext,iynext,i);
    d_f[n0next]=d_fnew[n0]; 
  }
}


//------------------------------------------------------
//------------ PROGRAMMING ON THE HOST ----------------
//--------------------- Clase LatticeBoltzmann ------------
class LatticeBoltzmann{
private:
  float h_tau[4]; // h_tau[0]=tau,  h_tau[1]=Utau,  h_tau[2]=UmUtau, h_tau[3]=ThreeUmU2tau
  float h_w[Q];      //Weights 
  int h_Vx[Q],h_Vy[Q];  //Velocity vectors
  float *h_f, *h_fnew;  float *d_f,*d_fnew; //Distribution Functions
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  int h_n(int ix,int iy,int i){return (ix*Ly+iy)*Q+i;};
  float h_rho(int ix,int iy);
  float h_Jx(int ix,int iy,float Fx); 
  float h_Jy(int ix,int iy,float Fy);
  float h_Fi(float Ux0,float Uy0,float Fx,float Fy,int i);
  float h_feq(float rho0,float Ux0,float Uy0,int i);
  void Start(float rho0,float Ux0,float Uy0);
  void Collision(float gx,float gy);
  void ImposeFields(void);
  void Advection(void);
  void Print(const char * NameFile,float gx,float gy);
								
};
LatticeBoltzmann::LatticeBoltzmann(void){
  
  //CONSTANTS(d_Symbols)
  //---Charge constantes on the Host-----------------
  //running constants
  h_tau[0]=tau;  h_tau[1]=Utau;  h_tau[2]=UmUtau; h_tau[3]=ThreeUmU2tau;
  //Set the weights
  h_w[0]=4.0/9; h_w[1]=h_w[2]=h_w[3]=h_w[4]=1.0/9; h_w[5]=h_w[6]=h_w[7]=h_w[8]=1.0/36;
  //Set the velocity vectors
  h_Vx[0]=0;  h_Vx[1]=1;  h_Vx[2]=0;  h_Vx[3]=-1; h_Vx[4]=0;
  h_Vy[0]=0;  h_Vy[1]=0;  h_Vy[2]=1;  h_Vy[3]=0;  h_Vy[4]=-1;

              h_Vx[5]=1;  h_Vx[6]=-1; h_Vx[7]=-1; h_Vx[8]=1;
              h_Vy[5]=1;  h_Vy[6]=1;  h_Vy[7]=-1; h_Vy[8]=-1;
  //------Send to the Device-----------------
  cudaMemcpyToSymbol(d_w,h_w,Q*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vx,h_Vx,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vy,h_Vy,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_tau,h_tau,4*sizeof(float),0,cudaMemcpyHostToDevice);
  										
  //DISTRIBUTION FUNCTIONS
  //Build the dynamic matrices on the host
  h_f=new float [ArraySize];  h_fnew=new float [ArraySize];
  //Build the dynamic matrices on the device
  cudaMalloc((void**) &d_f,ArraySize*sizeof(float));
  cudaMalloc((void**) &d_fnew,ArraySize*sizeof(float));
}
LatticeBoltzmann::~LatticeBoltzmann(void){
  delete[] h_f;  delete[] h_fnew;
  cudaFree(d_f);  cudaFree(d_fnew);
}
float LatticeBoltzmann::h_rho(int ix,int iy){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,i); sum+=h_fnew[n0];
  }
  return sum;
}  
float LatticeBoltzmann::h_Jx(int ix,int iy,float Fx){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,i); sum+=h_Vx[i]*h_fnew[n0];
  }
  return sum+0.5*Fx;
}  
float LatticeBoltzmann::h_Jy(int ix,int iy,float Fy){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=h_n(ix,iy,i); sum+=h_Vy[i]*h_fnew[n0];
  }
  return sum+0.5*Fy;
}  


float  LatticeBoltzmann::h_Fi(float Ux0,float Uy0,float Fx,float Fy,int i){
  float UdotVi=Ux0*h_Vx[i]+Uy0*h_Vy[i];
  float FdotVi=Fx*h_Vx[i]+Fy*h_Vy[i]; float UdotF=Ux0*Fx+Uy0*Fy;
  return h_tau[3]*h_w[i]*(FdotVi-UdotF+3*UdotVi*FdotVi);
}

float  LatticeBoltzmann::h_feq(float rho0,float Ux0,float Uy0,int i){
 float UdotVi=Ux0*h_Vx[i]+Uy0*h_Vy[i], U2=Ux0*Ux0+Uy0*Uy0;   
 return rho0*h_w[i]*(1+3*UdotVi+4.5*UdotVi*UdotVi-1.5*U2);
}  										
void LatticeBoltzmann::Start(float rho0,float Ux0,float Uy0){
  //Charge on the Host
  int ix,iy,i,n0;
  for(ix=0;ix<Lx;ix++) //for each cell
    for(iy=0;iy<Ly;iy++)
      for(i=0;i<Q;i++){ //on each direction
	n0=h_n(ix,iy,i); h_f[n0]=h_feq(rho0,Ux0,Uy0,i);
      }
  //Send to the Device
  cudaMemcpy(d_f,h_f,ArraySize*sizeof(float),cudaMemcpyHostToDevice);
}  

void LatticeBoltzmann::Collision(float gx,float gy){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Collision<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew,gx,gy);
}

void LatticeBoltzmann::ImposeFields(void){
//Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1); 
  dim3 BlocksPerGrid(M,1,1);
  d_ImposeFields<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew);
}
void LatticeBoltzmann::Advection(void){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Advection<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew);  	
}

void LatticeBoltzmann::Print(const char * NameFile,float gx,float gy){
  ofstream MyFile(NameFile); float rho0, Ux0,Uy0; int ix,iy; float Fx,Fy;
  //Bring back the data from Device to Host
  cudaMemcpy(h_fnew,d_fnew,ArraySize*sizeof(float),cudaMemcpyDeviceToHost);
  //Print for gnuplot plot  
  ix=0;
    for(iy=0;iy<Ly;iy++){
      rho0=h_rho(ix,iy); Fx=gx*rho0; Fy=gy*rho0;
      Ux0=h_Jx(ix,iy,Fx)/rho0;
      Uy0=h_Jy(ix,iy,Fy)/rho0;         		
      MyFile<<iy<<" "<<Ux0<<endl;
     }
  MyFile.close();
}

//------------------------------------------------
//------------------------Global Functions--------

int main(void){
LatticeBoltzmann Aire;
  int t,tmax=100000;
  float rho0=1.0, g=0.01;

  //Start
  Aire.Start(rho0,0,0);
  //Run
  for(t=0;t<tmax;t++){
    Aire.Collision(g,0);
    Aire.ImposeFields();
    Aire.Advection();
  }
  //Print
  Aire.Print("Poiseuille.dat",g,0);


return 0;
}
