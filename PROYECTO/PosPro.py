import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def take_inf(name):
    f=open(name,"r")
    x=f.readlines()
    L=len(x)
    Lx=int(x[-3].split()[0])
    Ly=int(x[-3].split()[1])
    Lz=int(x[-3].split()[2])
    f.close()
    return L,Lx+1,Ly+1,Lz+1

def rho(L,Lx,Ly,Lz,name):
    ix=np.arange(0,Lx,1)
    iy=np.arange(0,Ly,1)
    iz=np.arange(0,Lz,1)
    p=[]

    f=open(name,"r")
    for i in range(L):
        b=f.readline()
        if(b[0]!="\n"):
            p.append(float(b.split()[3]))
    return np.array(p),np.array(ix),np.array(iy),np.array(iz)

def print_xlayer(rho0,Lx,Ly,Lz,ix,y,z):
    z=np.array([rho0[ix*Ly*Lz+i*Lz+j] for j in z for i in y]).reshape(Lz,Ly)

    plt.imshow(z,origin="lower")
    plt.colorbar()

def print_ylayer(rho0,Lx,Ly,Lz,x,iy,z):
    z=np.array([rho0[i*Ly*Lz+iy*Lz+j] for j in z for i in x]).reshape(Lz,Lz)

    plt.imshow(z,origin="lower")
    plt.colorbar()

def print_zlayer(rho0,Lx,Ly,Lz,x,y,iz):
    z=np.array([rho0[i*Ly*Lz+j*Lz+iz] for j in y for i in x]).reshape(Ly,Lx)

    plt.imshow(z,origin="lower")
    plt.colorbar()

def print_profile(rho0,Lx,Ly,Lz,x,iy,iz):
    z=np.array([rho0[i*Ly*Lz+iy*Lz+iz] for i in x])

    plt.plot(x,z,"-")

def count_lines(name):
    f=open(name,"r")
    x=f.readlines()
    f.close()
    return len(x)
    
def SPL(name):
    L=count_lines(name)
    t=np.zeros(L)
    f=np.zeros(L)
    eps=1.8
    p=open(name,"r")

    for i in range(L):
        l=p.readline().split()
        t[i]=int(l[0])
        f[i]=float(l[1])
    ask=int(input("Qué quiere plotear:\n1:Potencia\n2:SPL\n"))

    if(ask==1):
      plt.figure()
      plt.xlabel("t")
      plt.ylabel("Pot")
      plt.plot(t,f,".")
      plt.show()

    if(ask==2):
      p0=input("presión de referencia? por defecto es la presión mayor.\n") 
      ts=int(input("tiempo de saturación?\n"))
      if(p0==""):
          for i in range(1,L-1):
              if(f[i]>f[i+1] and f[i]>f[i-1] and t[i]<ts):
                  p0=f[i]
      else: p0=float(p0)

      dB=np.array([20*np.log10(i/p0) for i in f])

      for i in range(L):
          if(np.abs(dB[i]+30)<eps and t[i]>ts):
              print("tiempo de reverberación T30: ",t[i]-ts," clks")
                  
      plt.figure()
      plt.xlabel("t")
      plt.ylabel("SPL (dB)")
      plt.plot(t,dB,".")
      plt.show() 
    

if __name__=="__main__":
    p=int(input("Qué análisis desea:\n1: superficies de nivel\n2: perfil de onda\n3: SPL\n"))
    if(p==1):
        L,Lx,Ly,Lz=take_inf("Ondas.dat")
        rho0,x,y,z=rho(L,Lx,Ly,Lz,"Ondas.dat")
        niv=int(input("Qué tipo de superficie desea:\n1: x cte\n2: y cte\n3: z cte\n"))
        if(niv==1):
          l=int(input("corte en x?\n"))
          plt.figure(dpi=100)
          print_xlayer(rho0,Lx,Ly,Lz,l,y,z)
          plt.show()
        if(niv==2):
          l=int(input("corte en y?\n"))
          plt.figure(dpi=100)
          print_ylayer(rho0,Lx,Ly,Lz,x,l,z)
          plt.show()
        if(niv==3):
          l=int(input("corte en z?\n"))
          plt.figure(dpi=100)
          print_zlayer(rho0,Lx,Ly,Lz,x,y,l)
          plt.show()
    
    if(p==2):
        iy=int(input("corte en y?\n"))
        iz=int(input("corte en z?\n"))
        L,Lx,Ly,Lz=take_inf("Ondas.dat")
        rho0,x,y,z=rho(L,Lx,Ly,Lz,"Ondas.dat")
        plt.figure()
        print_profile(rho0,Lx,Ly,Lz,x,iy,iz)
        plt.show()
        
    if(p==3):
        SPL("Pot.dat")
    
