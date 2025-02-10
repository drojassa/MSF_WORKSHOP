reset 
clear 
set xrange [100:180]
set yrange [150:200]
set terminal jpeg enhanced 
set output "LongitudDeOnda60.jpg"

R(x)= -a*(x-b)
T(x)= -c*(x-d)
V(x)= -e*(x-f)
S(x)= -g*(x-h)

a=c=e=g=1.4
b=288
d=282
f=276
h=270

fit R(x) "1Datos60.dat" via a,b
fit T(x) "2Datos60.dat" via c,d
fit V(x) "3Datos60.dat" via e,f
fit S(x) "4Datos60.dat" via g,h

plot "1Datos60.dat","2Datos60.dat","3Datos60.dat","4Datos60.dat", R(x),T(x),V(x),S(x)
