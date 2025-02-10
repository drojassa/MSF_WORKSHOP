reset 
clear 
set xrange [100:180]
set yrange [0:200]
set terminal jpeg enhanced 
set output "LongitudDeOnda20.jpg"

R(x)= -a*(x-b)
T(x)= -c*(x-d)
V(x)= -e*(x-f)
U(x)= -g*(x-h)
S(x)= -i*(x-j)
a=c=e=q=i=5.2
b=174
d=169
f=164
h=159
j=154

fit R(x) "1Datos20.dat" via a,b
fit T(x) "2Datos20.dat" via c,d
fit V(x) "3Datos20.dat" via e,f
fit U(x) "4Datos20.dat" via g,h
fit S(x) "5Datos20.dat" via i,j
plot "1Datos20.dat","2Datos20.dat","3Datos20.dat", "4Datos20.dat", "5Datos20.dat",R(x),T(x),V(x),U(x),S(x)
