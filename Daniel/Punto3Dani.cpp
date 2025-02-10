#include <iostream>
#include <fstream>
#include <cmath>
#include "Vector.h"
using namespace std;

const int Lx=512;
const int Ly=64;

const int Q=9;

const int N=24, ixc=128, iyc=32, R=8;

//-----Clase LatticeBoltzmann----//
class LatticeBoltzmann{
private:
  double w[Q]; //pesos
  int Vx[Q],Vy[Q]; //vectores de vel
  double *f,*fnew; //func. de distribución
  double Elementos[N][2];
  
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  int n(int ix,int iy,int i){return (ix*Ly+iy)*Q+i;};
  double rho(int ix,int iy,bool UseNew);
  double Jx(int ix,int iy,bool UseNew);
  double Jy(int ix,int iy,bool UseNew);
  double feq(double rho0,double Ux0, double Uy0,int i);
  double sigmaxx(int ix,int iy, double eta);
  double sigmayy(int ix,int iy, double eta);
  double sigmaxy(int ix,int iy, double eta);
  void dFuerza(double x,double y,double dAx,double dAy,double & Foutx, double & Fouty, double eta);
  void Ftotal(double & Ftotx, double & Ftoty, double eta);
  void Start(double rho0, double Ux0, double Uy0);
  void Collision(double tau);
  void ImposeFields(double Ufan);
  void Advection(void);
  void Print(const char * NameFile,double Ufan);
};

LatticeBoltzmann::LatticeBoltzmann(void){
  //Set the weights
  w[0]=4.0/9;  w[1]=w[2]=w[3]=w[4]=1.0/9;  w[5]=w[6]=w[7]=w[8]=1.0/36;
  //Set the velocity vectors
  Vx[0]=0;  Vx[1]=1;  Vx[2]=0;  Vx[3]=-1; Vx[4]=0;
  Vy[0]=0;  Vy[1]=0;  Vy[2]=1;  Vy[3]=0;  Vy[4]=-1;

            Vx[5]=1;  Vx[6]=-1; Vx[7]=-1; Vx[8]=1;
            Vy[5]=1;  Vy[6]=1;  Vy[7]=-1; Vy[8]=-1;
 //Create the dynamic arrays
  int ArraySize=Lx*Ly*Q;
  
  for(int k=0;k<N;k++){
    Elementos[k][0]=ixc+R*cos(k*2*M_PI/N);
    Elementos[k][1]=iyc+R*sin(k*2*M_PI/N);
    //  cout<<Elementos[k][0]<<' '<< Elementos[k][1]<<endl;
  }
  
  f=new double [ArraySize];  fnew=new double [ArraySize];
}
LatticeBoltzmann::~LatticeBoltzmann(void){
    delete[] f;  delete[] fnew;
}

double LatticeBoltzmann::rho(int ix,int iy,bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i);
    if(UseNew) sum+=fnew[n0]; else sum+=f[n0];
  }
  return sum;
}
double LatticeBoltzmann::Jx(int ix, int iy, bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i);
    if(UseNew) sum+=Vx[i]*fnew[n0]; else sum+=Vx[i]*f[n0];
  }
  return sum;
}
double LatticeBoltzmann::Jy(int ix, int iy, bool UseNew){
  double sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i);
    if(UseNew) sum+=Vy[i]*fnew[n0]; else sum+=Vy[i]*f[n0];
  }
  return sum;
}
double LatticeBoltzmann::feq(double rho0,double Ux0,double Uy0, int i){
  double UdotVi=Ux0*Vx[i]+Uy0*Vy[i], U2=Ux0*Ux0+Uy0*Uy0;
  return rho0*w[i]*(1+3*UdotVi+4.5*UdotVi*UdotVi-1.5*U2); 
}
void LatticeBoltzmann::Start(double rho0, double Ux0, double Uy0){
  int ix,iy,i,n0;
  for(ix=0;ix<Lx;ix++)//for each cell
    for(iy=0;iy<Ly;iy++)
      for(i=0;i<Q;i++){ //on each direction 
        n0=n(ix,iy,i);
        f[n0]=feq(rho0,Ux0,Uy0,i);
      }
}
double LatticeBoltzmann::sigmaxx(int ix,int iy, double eta){
  double derivxx=0; int i; double rho0i,rho0f;
  rho0i=rho(ix,iy,false);
  for(i=0;i<Q;i++){
    rho0f=rho(ix+Vx[i],iy+Vy[i],false);
    derivxx+=3*w[i]*Vx[i]*Jx(ix+Vx[i],iy+Vy[i],false)/rho0f;
  }
  return -rho0i/3+eta*2*derivxx;
}
double LatticeBoltzmann::sigmayy(int ix,int iy, double eta){
  double derivyy=0; int i; double rho0i,rho0f;
  rho0i=rho(ix,iy,false);
  for(i=0;i<Q;i++){
    rho0f=rho(ix+Vx[i],iy+Vy[i],false);
    derivyy+=3*w[i]*Vy[i]*Jy(ix+Vx[i],iy+Vy[i],false)/rho0f;
  }
  return -rho0i/3+eta*2*derivyy;
}
double LatticeBoltzmann::sigmaxy(int ix,int iy, double eta){
  double derivxy=0,derivyx=0; int i; double rho0i,rho0f;
  rho0i=rho(ix,iy,false);
  for(i=0;i<Q;i++){
    rho0f=rho(ix+Vx[i],iy+Vy[i],false);
    derivxy+=3*w[i]*Vy[i]*Jx(ix+Vx[i],iy+Vy[i],false)/rho0f;
    derivyx+=3*w[i]*Vx[i]*Jy(ix+Vx[i],iy+Vy[i],false)/rho0f;
  }
  return eta*(derivxy+derivyx);
}
void LatticeBoltzmann::dFuerza(double x,double y,double dAx, double dAy,double & Foutx, double & Fouty, double eta){
  double phixx, phiyy, phixy, u, v;int i, ix=floor(x), iy=floor(y);
  u=(x-ix); v=(y-iy);
  phixx=sigmaxx(ix,iy,eta)*(1-u)*(1-v)+sigmaxx(ix+1,iy, eta)*u*(1-v)+
    sigmaxx(ix,iy+1,eta)*(1-u)*v+sigmaxx(ix+1,iy+1,eta)*u*v;
  
  phiyy=sigmayy(ix,iy,eta)*(1-u)*(1-v)+sigmayy(ix+1,iy,eta)*u*(1-v)+
    sigmayy(ix,iy+1,eta)*(1-u)*v+sigmayy(ix+1,iy+1,eta)*u*v;

  phixy=sigmaxy(ix,iy,eta)*(1-u)*(1-v)+sigmaxy(ix+1,iy,eta)*u*(1-v)+
    sigmaxy(ix,iy+1,eta)*(1-u)*v+sigmaxy(ix+1,iy+1,eta)*u*v;

  Foutx=phixx*dAx+phixy*dAy;
  Fouty=phixy*dAx+phiyy*dAy;
}

void LatticeBoltzmann::Ftotal(double & Ftotx, double & Ftoty, double eta){
  int k; double dAx, dAy ;
  double AuxFx=0,AuxFy=0;
  for(k=0;k<N;k++){
    dAx=(2*R*sin(M_PI/N))*cos(k*2*M_PI/N);
    dAy=(2*R*sin(M_PI/N))*sin(k*2*M_PI/N);
    dFuerza(Elementos[k][0],Elementos[k][1],dAx,dAy,AuxFx,AuxFy,eta);
    Ftotx+=AuxFx;
    Ftoty+=AuxFy;
  }
}


void LatticeBoltzmann::Collision(double tau){
  int ix,iy,i,n0; double rho0,Ux0,Uy0;
  double Utau=1.0/tau;
  double UmUtau=1-Utau;
  for(ix=0;ix<Lx;ix++)
    for(iy=0;iy<Ly;iy++){
      rho0=rho(ix,iy,false); Ux0=Jx(ix,iy,false)/rho0; Uy0=Jy(ix,iy,false)/rho0;
      for(i=0;i<Q;i++){
        n0=n(ix,iy,i);
        fnew[n0]=UmUtau*f[n0]+Utau*feq(rho0,Ux0,Uy0,i);
      }
    }
}
void LatticeBoltzmann::ImposeFields(double Ufan){
  int i,ix,iy,n0; double rho0; int ixc=Lx/4, iyc=Ly/2, R=Ly/8; double R2=R*R;
  //go through all cells, looking if there are fan or obstacle 
  for(ix=0;ix<Lx;ix++)
    for(iy=0;iy<Ly;iy++){
      rho0=rho(ix,iy,false);
      //fan
      if(ix==0)
        for(i=0;i<Q;i++){n0=n(ix,iy,i); fnew[n0]=feq(rho0,Ufan,0,i);}
      //obstacle
      else if((ix-ixc)*(ix-ixc)+(iy-iyc)*(iy-iyc)<=R2)
        for(i=0;i<Q;i++){n0=n(ix,iy,i); fnew[n0]=feq(rho0,0,0,i);}
      //An extra point at one side to break the isotropy
      else if(ix==ixc && iy==iyc+R+1)
        for(i=0;i<Q;i++){n0=n(ix,iy,i);fnew[n0]=feq(rho0,0,0,i);} 
    }
}
void LatticeBoltzmann::Advection(void){
  int ix,iy,i,ixnext,iynext,n0,n0next;
  for(ix=0;ix<Lx;ix++)
    for(iy=0;iy<Ly;iy++)
      for(i=0;i<Q;i++){
        ixnext=(ix+Vx[i]+Lx)%Lx; iynext=(iy+Vy[i]+Ly)%Ly;
        n0=n(ix,iy,i); n0next=n(ixnext,iynext,i);
        f[n0next]=fnew[n0];
      }
}
void LatticeBoltzmann::Print(const char * NameFile,double Ufan){
  ofstream MyFile(NameFile); double rho0,Ux0,Uy0; int ix,iy;
  for(ix=0;ix<Lx;ix+=4){
    for(iy=0;iy<Ly;iy+=4){
      rho0=rho(ix,iy,true); Ux0=Jx(ix,iy,true)/rho0; Uy0=Jy(ix,iy,true)/rho0;
      MyFile<<ix<<" "<<iy<<" "<<Ux0/Ufan*4<<" "<<Uy0/Ufan*4<<endl;
    }
    MyFile<<endl;
  }
  MyFile.close();
}
//------Programa Principal------//
int main(void){
  ofstream Fuerza("CoefArrastreThin.txt");
  for(double tau=0.51;tau<2.0;tau+=0.01){
  double eta = (tau - 1/2.0)/3.0;
  double CoefArrastre,Re;
  LatticeBoltzmann Air;
  int t,tmax=3000;
  double rho0=1.0, Ufan0=0.1;
  double FXdrag = 0;
  double FYdrag = 0;
  Air.Start(rho0,Ufan0,0);
  for(t=0;t<tmax;t++){
    Air.Collision(tau);
    Air.ImposeFields(Ufan0);
    Air.Advection();
  }
  Air.Ftotal(FXdrag, FYdrag, eta);
  Air.Print("WindChannel.dat",Ufan0);
  CoefArrastre=2*FXdrag/(2*R*rho0*Ufan0*Ufan0);
  Re=rho0*Ly*Ufan0/eta;
  Fuerza<<Re<<" "<<CoefArrastre<<endl;
  }
  /*
  for(double tau=2.0;tau<6.0;tau+=0.2){
  double eta = (tau - 1/2.0)/3.0;
  double CoefArrastre,CoefMagnus, Re;
  LatticeBoltzmann Air;
  int t,tmax=3000;
  double rho0=1.0, Ufan0=0.1;
  double FXdrag = 0;
  double FYdrag = 0;
  Air.Start(rho0,Ufan0,0);
  for(t=0;t<tmax;t++){
    Air.Collision(tau);
    Air.ImposeFields(Ufan0);
    Air.Advection();
  }
  Air.Ftotal(FXdrag, FYdrag, eta);
  Air.Print("WindChannel.dat",Ufan0);
  CoefArrastre=2*FXdrag/(2*R*rho0*Ufan0*Ufan0);
  CoefMagnus=2*sqrt(FXdrag*FXdrag+FYdrag*FYdrag)/(2*R*R*2*M_PI/1000*Ufan0);
  Re=rho0*Ly*Ufan0/eta;
  Fuerza<<Re<<" "<<CoefArrastre<<endl;
  }*/
  return 0;
} 
