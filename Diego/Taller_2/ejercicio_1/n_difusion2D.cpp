#include<iostream>
#include<fstream>
#include<cmath>
#include"Random64.h"

const int Lx=256; //x casillas de la grilla
const int Ly=256;
const double p=0.25;  //P de doblar a izq o derec
const double p0=0.25; //P de no cambiar dirección: P0+2P<=1.
const int Q=4; //cantidad de direcciones

/*---Clase Lattice---*/
class Lattice{
private:
  int n[Q*Lx*Ly];
  int nnew[Q*Lx*Ly];
  int Vx[Q];
  int Vy[Q];
public:
  Lattice(void);
  int n_i(int ix, int iy, int i){return Q*Lx*ix+Q*iy+i;};
  void start(int N,
	     double mux,double sigmax,
	     double muy,double sigmay,
	     Crandom &ran64x,
	     Crandom &ran64y,
	     Crandom &ran64Q);
  void colisione(Crandom &ran64);
  void adveccione(void);
  int rho(int ix,int iy,bool usenew);
  double varianza(bool usenew);
  void show(bool usenew);
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
	n0=n_i(ix,iy,i);
	n[n0]=nnew[n0]=0;
      }
    }
  }
}

void Lattice::start(int N,double mux, double sigmax, double muy, double sigmay, Crandom &ran64x,Crandom &ran64y, Crandom &ran64Q){
  int ix,iy,i,n0;

  while(N>0){
    ix=(int) ran64x.gauss(mux,sigmax); //se escoge aleatoriamente una celda x con
    if(ix<0){ix=0;}                    //distribución gaussiana y se tiene cuidado
    if(ix>Lx-1){ix=Lx-1;}              //de no salirse del array
    
    iy=(int) ran64y.gauss(muy,sigmay); //se escoge aleatoriamente una celda y con
    if(iy<0){iy=0;}                    //distribución gaussiana y se tiene cuidado
    if(iy>Ly-1){iy=Ly-1;}              //de no salirse del array
    
    i=(int) Q*ran64Q.r();              //se escoge aleatoriamente una dirección
    n0=n_i(ix,iy,i);
    if(n[n0]==0){                      //si la casilla de esa dirección está
      n[n0]=1;                         //vacía se llena con una partícula
    }
    N--;                               //se disminuye N
  }
}

void Lattice::colisione(Crandom &ran64){
  int ix,iy,i,n0,n0new;
  double P;

  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){ //para cada celda del arreglo se genera
      P=ran64.r();        //un número aleatorio
     
      if(P<=p0){           //con P=p0 se quedan quietas las partículas
	for(i=0;i<Q;i++){ //se copia el n al nnew
	  n0=n_i(ix,iy,i);
	  nnew[n0]=n[n0];
	}
      }
    
      if(P>p0 && P<=p0+p){ //con P=p se mueven las partpículas a 90°
	for(i=0;i<Q;i++){ //es decir i -> (i+1)%Q
	  n0=n_i(ix,iy,i);
	  n0new=n_i(ix,iy,(i+1)%Q);
	  nnew[n0new]=n[n0];
	}
      }

      if(P>p0+p && P<=p0+2*p){ //con P=p se mueven las partpículas a 270°
	for(i=0;i<Q;i++){ //es decir i -> (i+Q-1)%Q
	  n0=n_i(ix,iy,i);
	  n0new=n_i(ix,iy,(i+Q-1)%Q);
	  nnew[n0new]=n[n0];
	}
      }

      if(P>p0+2*p && P<1){ //con P=1-p0-2p se mueven las partpículas a 180°
	for(i=0;i<Q;i++){ //es decir i -> (i+2)%Q
	  n0=n_i(ix,iy,i);
	  n0new=n_i(ix,iy,(i+2)%Q);
	  nnew[n0new]=n[n0];
	}
      }
    }
  }
}

void Lattice::adveccione(void){
  int ix,iy,i,n0,n0new;

  for(ix=0;ix<Lx;ix++){
    for(iy=0;iy<Ly;iy++){
      for(i=0;i<Q;i++){
	n0new=n_i(ix,iy,i);
	n0=n_i((ix+Vx[i]+Lx)%Lx,(iy+Vy[i]+Ly)%Ly,i);
	n[n0]=nnew[n0new];
      }
    }
  }
}

int Lattice::rho(int ix,int iy,bool usenew){
  int i,n0,sum;

  for(sum=0,i=0;i<Q;i++){ //se hace un conteo del número de partículas que hay por celda.
    n0=n_i(ix,iy,i);
    if(usenew){
      sum+=nnew[n0];
    }
    else{
      sum+=n[n0];
    }
  }

  return sum;
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
  sigma2/=(N-1);
  
  return sigma2;
}

void Lattice::show(bool usenew){
  int ix,iy,i,n0;
  
  for(i=0;i<Q;i++){
    std::cout<<"Lattice for direction "<<i<<":\n"; //se imprime la lattice para cada dirección i de la forma: (ix,iy)
                                                   // y|(0,Ly-1)(1,Ly-1)...(Lx-2,Ly-1)(Lx-1,Ly-1)
    for(int iy=Ly-1;iy>=0;iy--){                   //  |(0,Ly-2)(1,Ly-2)...(Lx-2,Ly-2)(Lx-1,Ly-2)
      for(int ix=0;ix<Lx;ix++){                    //  ...
	n0=n_i(ix,iy,i);                           //  |(0,2)(1,2)(2,2)... (Lx-2,2)(Lx-1,2)
	if(usenew){                                //  |(0,1)(1,1)(2,1)... (Lx-2,1)(Lx-1,1)
	  std::cout<<nnew[n0]<<" ";                //  |(0,0)(1,0)(2,0)... (Lx-2,0)(Lx-1,0)
	}                                          //  --------------------------------------------x
	else{
	  std::cout<<n[n0]<<" ";
	}	
      }
      std::cout<<"\n";
    }
  }
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

/*---Programa principal---*/
int main(void){
  /*construcción de clases*/
  Lattice difusion;
  Crandom ran64(1);

  /*Variables de simulación*/
  int N=2400;
  double mu=Lx/2;
  double sigma=16;
  int t, tmax=350;

  /*Inicio*/
  difusion.start(N,mu,sigma,mu,sigma,ran64,ran64,ran64);
  
  for(t=0;t<tmax;t++){
    difusion.colisione(ran64);
    difusion.adveccione();
    //difusion.show(false);
    std::cout<<t<<"\t"<<difusion.varianza(false)<<"\n";
  }
  //difusion.print("distr.txt",false);
  
  return 0;
}
