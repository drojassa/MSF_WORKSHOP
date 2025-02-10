reset 
clear 
set xrange [100:180]
set yrange [75:200]
set terminal jpeg enhanced 
set output "LongitudDeOnda30.jpg"

R(x)= -a*(x-b)
T(x)= -c*(x-d)
V(x)= -e*(x-f)
S(x)= -g*(x-h)
U(x)= -i*(x-j)
a=c=e=g=i=3.5
b=195
d=190
f=185
h=180
j=175

fit R(x) "1Datos30.dat" via a,b
fit T(x) "2Datos30.dat" via c,d
fit V(x) "3Datos30.dat" via e,f
fit S(x) "4Datos30.dat" via g,h
fit U(x) "5Datos30.dat" via i,j

plot "1Datos30.dat","2Datos30.dat","3Datos30.dat","4Datos30.dat", "5Datos30.dat",R(x),T(x),V(x),S(x),U(x)
