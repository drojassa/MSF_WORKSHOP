#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

const int Lx=64;
const int Ly=64;
const int Lz=64; //Se agrega la dimensión z

const int Q=7;
const double W0=1.0/4; //Se cambia el peso a 1/4 en lugar de 1/3

const double C=0.5; // C<0.707 cells/click
const double C2=C*C;
const double AUX0=1-3*C2*(1-W0);

const double tau=0.5;
const double Utau=1.0/tau;
const double UmUtau=1-Utau;

//-----Clase LatticeBoltzmann----//
class LatticeBoltzmann{
private:
  double w[Q]; //pesos
  int Vx[Q],Vy[Q],Vz[Q]; //vectores de vel, se agrega en dirección z
  double *f,*fnew; //func. de distribución
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  int n(int ix,int iy,int iz,int i){return (ix*Ly+iy*Lz+iz)*Q+i;}; //Se modifica la función indice, puede que esté mal
//Se agrega el índice iz para la declaración de todas las funciones 
  double rho(int ix,int iy,int iz,bool UseNew);
  double Jx(int ix,int iy,int iz,bool UseNew);
  double Jy(int ix,int iy,int iz,bool UseNew);
  double Jz(int ix,int iy,int iz,bool UseNew); //Se agrega el campo macroscópico en dirección z
  double feq(double rho0,double Jx0, double Jy0,double Jz0,int i); //Se agrega Jz0
  void Start(double rho0, double Jx0, double Jy0, double Jz0);//Se agrega Jz0
  void Collision(void);
  void ImposeFields(int t);
  void Advection(void);
  void Print(const char * NameFile);
};

LatticeBoltzmann::LatticeBoltzmann(void){
  //Set the weights
  w[0]=W0; w[1]=w[2]=w[3]=w[4]=w[5]=w[6]=W0/2; //Se agregan los pesos correspondientes a los nuevos vectores del D3Q7, y se cambian para recuperar ondas acústicas
  //Set the velocity vectors
  Vx[0]=0;  Vx[1]=1;  Vx[2]=-1;  Vx[3]=0; Vx[4]=0; Vx[5]=0; Vx[6]=0;
  Vy[0]=0;  Vy[1]=0;  Vy[2]=0;  Vy[3]=1;  Vy[4]=-1; Vy[5]=0;Vy[6]=0;
  Vz[0]=0;  Vz[1]=0;  Vz[2]=0;  Vz[3]=0;  Vz[4]=0; Vz[5]=1; Vz[6];
  //Create the dynamic arrays
  int ArraySize=Lx*Ly*Lz*Q; //Se agrega la dimension z
  f=new double [ArraySize];  fnew=new double [ArraySize];
}
LatticeBoltzmann::~LatticeBoltzmann(void){
    delete[] f;  delete[] fnew;
}

double LatticeBoltzmann::rho(int ix,int iy, int iz,bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,iz,i);
    if(UseNew) sum+=fnew[n0]; else sum+=f[n0];
  }
  return sum;
}
double LatticeBoltzmann::Jx(int ix,int iy,int iz,bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,iz,i);
    if(UseNew) sum+=Vx[i]*fnew[n0]; else sum+=Vx[i]*f[n0];
  }
  return sum;
}
double LatticeBoltzmann::Jy(int ix,int iy,int iz,bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,iz,i);
    if(UseNew) sum+=Vy[i]*fnew[n0]; else sum+=Vy[i]*f[n0];
  }
  return sum;
}
double LatticeBoltzmann::Jz(int ix,int iy,int iz,bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,iz,i);
    if(UseNew) sum+=Vz[i]*fnew[n0]; else sum+=Vz[i]*f[n0];
  }
  return sum;
}
double LatticeBoltzmann::feq(double rho0,double Jx0,double Jy0, double Jz0, int i){
  if(i>0)
    return 3*w[i]*(C2*rho0+Vx[i]*Jx0+Vy[i]*Jy0+Vz[i]*Jz0);
  else 
    return rho0*AUX0; 
}
void LatticeBoltzmann::Start(double rho0, double Jx0, double Jy0, double Jz0){
  int ix,iy,iz,i,n0;
  for(ix=0;ix<Lx;ix++)//for each cell
    for(iy=0;iy<Ly;iy++)
      for(iz=0;iz<Lz;iz++)
        for(i=0;i<Q;i++){ //on each direction 
          n0=n(ix,iy,iz,i);
          f[n0]=feq(rho0,Jx0,Jy0,Jz0,i);
        }
}
void LatticeBoltzmann::Collision(void){
  int ix,iy,iz,i,n0; double rho0,Jx0,Jy0,Jz0;
  for(ix=0;ix<Lx;ix++)
    for(iy=0;iy<Ly;iy++)
      for(iz=0;iz<Lz;iz++){
      rho0=rho(ix,iy,iz,false); Jx0=Jx(ix,iy,iz,false); Jy0=Jy(ix,iy,iz,false);Jz0=Jz(ix,iy,iz,false);
      for(i=0;i<Q;i++){
        n0=n(ix,iy,iz,i);
        fnew[n0]=UmUtau*f[n0]+Utau*feq(rho0,Jx0,Jy0,Jz0,i);
      }
    }
}
void LatticeBoltzmann::ImposeFields(int t){
  int i,ix,iy,iz,n0;
  double lambda,omega,rho0,Jx0,Jy0,Jz0; lambda=10; omega=2*M_PI/lambda*C;
  ix=Lx/2; iy=Ly/2; iz=Lz/2;
  rho0=10*sin(omega*t); Jx0=Jx(ix,iy,iz,false); Jy0=Jy(ix,iy,iz,false); Jz0=Jz(ix,iy,iz,false);
  for(i=0;i<Q;i++){
    n0=n(ix,iy,iz,i);
    fnew[n0]=feq(rho0,Jx0,Jy0,Jz0,i);
  }
}
void LatticeBoltzmann::Advection(void){
  int ix,iy,iz,i,ixnext,iynext, iznext,n0,n0next;
  for(ix=0;ix<Lx;ix++)
    for(iy=0;iy<Ly;iy++)
      for(iz=0;iz<Lz;iz++)
        for(i=0;i<Q;i++){
          ixnext=(ix+Vx[i]+Lx)%Lx; iynext=(iy+Vy[i]+Ly)%Ly; iznext=(iz+Vz[i]+Lz)%Lz;
          n0=n(ix,iy,iz,i); n0next=n(ixnext,iynext, iznext,i);
          f[n0next]=fnew[n0];
        }
}
void LatticeBoltzmann::Print(const char * NameFile){
  ofstream MyFile(NameFile); double rho0; int ix,iy,iz;
  for(ix=0;ix<Lx;ix++){
      for(iy=0;iy<Ly;iy++){
        for(iz=0;iz<Lz;iz++){
          rho0=rho(ix,iy,iz,true);
          MyFile<<ix<<" "<<iy<<" "<<iz<<" "<<rho0<<endl;
        }
  }
    
  }
  MyFile<<endl;    //Cuidado con éstas lineas vacias en el archivo
  MyFile.close();
}
//------Programa Principal------//
int main(void){
  LatticeBoltzmann Ondas;
  int t,tmax=100;
  double rho0=0, Jx0=0, Jy0=0, Jz0=0;
  
  Ondas.Start(rho0,Jx0,Jy0,Jz0);
  for(t=0;t<tmax;t++){
    Ondas.Collision();
    Ondas.ImposeFields(t);
    Ondas.Advection();
  }
  Ondas.Print("Ondas.dat");
  return 0;
}
