reset 
clear 
set xrange [100:180]
set yrange [120:200]
set terminal jpeg enhanced 
set output "LongitudDeOnda45.jpg"

R(x)= -a*(x-b)
T(x)= -c*(x-d)
V(x)= -e*(x-f)
S(x)= -g*(x-h)
U(x)= -i*(x-j)

a=c=e=g=i=2.1
b=235
d=230
f=225
h=220
j=215

fit R(x) "1Datos45.dat" via a,b
fit T(x) "2Datos45.dat" via c,d
fit V(x) "3Datos45.dat" via e,f
fit S(x) "4Datos45.dat" via g,h
fit U(x) "5Datos45.dat" via i,j

plot "1Datos45.dat","2Datos45.dat","3Datos45.dat","4Datos45.dat", "5Datos45.dat",R(x),T(x),V(x),S(x),U(x)
