#include<iostream>
#include <fstream>
#include<cmath>

const int Lx=256; //x casillas de la grilla
const int Ly=256;
const double p=0.25;  //P de doblar a izq o derec
const double p0=0.25; //P de no cambiar dirección: P0+2P<=1.
const int Q=4; //cantidad de direcciones

/*---Clase Lattice---*/
class Lattice{
private:
  double f[Q*Lx*Ly];
  double fnew[Q*Lx*Ly];
  int Vx[Q];
  int Vy[Q];
public:
  Lattice(void);
  int n(int ix, int iy, int i){return Q*Lx*ix+Q*iy+i;};
  void start(int N,
	     double mux,double sigmax,
	     double muy,double sigmay);
  void colisione(void);
  void adveccione(void);
  double rho(int ix,int iy,bool usenew);
  double varianza(bool usenew);
  void print(const char *filename,bool usenew);
};

/*---implementaciones---*/
Lattice::Lattice(void){
  Vx[0]=1; Vx[1]=0; Vx[2]=-1; Vx[3]=0;  //se inician las 4 direcciones de
  Vy[0]=0; Vy[1]=1; Vy[2]=0;  Vy[3]=-1; //movimiento

  int n0;
  
  for(int ix=0;ix<Lx;ix++){   //se inicializan las dos grillas con cero
    for(int iy=0;iy<Ly;iy++){ //partículas en cada casilla y cada
      for(int i=0;i<Q;i++){   //dicercción
	n0=n(ix,iy,i);
	f[n0]=fnew[n0]=0;
      }
    }
  }
}

void Lattice::start(int N,double mux,double sigmax,double muy,double sigmay){
  int ix,iy,i,n0;
  double rho,M;

  M=N/(2*M_PI*sigmax*sigmay); /*se calcula rho como una función de distribución
				bigaussiana en x y y de tal manera que sumando
				rho sobre todas las celdas de N. Se le asigna
				a cada dirección la misma cantidad de rho para
				que la suma sobre la celda en las 4 direcciones
				de rho: rho/4=rho/Q para cada dirección.*/

  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){
      rho=M*std::exp(-1/2.0*(std::pow((ix-mux)/sigmax,2)
			     +std::pow((iy-muy)/sigmay,2)));
      for(i=0;i<Q;i++){
	n0=n(ix,iy,i);
	f[n0]=rho/Q;
      }
    }
  } 
}

void Lattice::colisione(void){
  int ix,iy,i,n0,n1,n2,n3,n0new;
  
  /*para cada celda y cada dirección, la nueva f será la prob de que
    lleguen ahí partículas de todas las direcciones de la casilla ix iy*/
  for(ix=0;ix<Lx;ix++){   
    for(iy=0;iy<Ly;iy++){ 
      for(i=0;i<Q;i++){   
	n0new=n0=n(ix,iy,i); /*dirección de la nueva f (tambien es la dirección
			       0° de la antigua f)*/
	n1=n(ix,iy,(i+1)%Q);   /*direccion de la antigua f a 90° de la nueva f*/
	n2=n(ix,iy,(i+Q-1)%Q); /*direccion de la antigua f a 270° de la nueva f*/
	n3=n(ix,iy,(i+2)%Q);   /*direccion de la antigua f a 180° de la nueva f*/

	fnew[n0new]=p0*f[n0]+p*(f[n1]+f[n2])+(1-2*p-p0)*f[n3];
      }
    }
  }
}

void Lattice::adveccione(void){
  int ix,iy,i,n0,n0new;

  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){
      for(i=0;i<Q;i++){
	n0new=n(ix,iy,i);
	n0=n((ix+Vx[i]+Lx)%Lx,(iy+Vy[i]+Ly)%Ly,i);
	f[n0]=fnew[n0new];
      }
    }
  }
}

double Lattice::rho(int ix,int iy,bool usenew){
  double suma; int i,n0;
  for(suma=0,i=0;i<Q;i++){
    n0=n(ix,iy,i);
    if(usenew){suma+=fnew[n0];}
    else{ suma+=f[n0];}
  }
  return suma;
}

double Lattice::varianza(bool usenew){
  int ix,iy;
  double N=0,Xpro=0,Ypro=0,sigma2=0;

  for(ix=0;ix<Lx;ix++){   //se recuentan todas las partículas, en caso de que
    for(iy=0;iy<Ly;iy++){ //alguna desaparezca
      N+=rho(ix,iy,usenew);
    }
  }

  for(ix=0;ix<Lx;ix++){   //para cada celda se calcula la posición por la
    for(iy=0;iy<Ly;iy++){ //densidad->centro de masa
      Xpro+=ix*rho(ix,iy,usenew);
      Ypro+=iy*rho(ix,iy,usenew);
    }
  }
  Xpro/=N;Ypro/=N;
                          //para cada celda se calcula la varianza:
  for(ix=0;ix<Lx;ix++){   //sum |r_i-Rpro|²/(N-1)*rho-> rho para dar cuenta del
    for(iy=0;iy<Ly;iy++){ //número de partículas por celda
      sigma2+=(std::pow(ix-Xpro,2)+std::pow(iy-Ypro,2))*rho(ix,iy,usenew);
    }
  }
  sigma2/=N;
  
  return sigma2;
}

void Lattice::print(const char *filename,bool usenew){
  std::ofstream Myfile(filename);
  int ix,iy;
  double rho0;

  for(ix=0;ix<Lx;ix+=4){
    for(iy=0;iy<Ly;iy+=4){
      rho0=rho(ix,iy,usenew);
      Myfile<<ix<<"\t"<<iy<<"\t"<<rho0<<"\n";
    }
    Myfile<<"\n";
  }
  Myfile.close();
}

/*---Programa Principal---*/
int main(){
  Lattice difusion;
  
  int N=2400; double mu=Lx/2,sigma=16;
  int t, tmax=350;
  
  difusion.start(N,mu,sigma,mu,sigma);
  
  for(t=0;t<tmax;t++){
    std::cout<<t<<"\t"<<difusion.varianza(false)<<"\n";
    difusion.colisione();
    difusion.adveccione();
  }
  //difusion.print("data_init.txt",false);
  
  return 0;
}
