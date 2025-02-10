reset
clear
set xrange [0:200]
set yrange [0:200]
set pm3d map
set size ratio 1
set terminal jpeg enhanced 
set output "Ondas_4.jpg"
splot "Ondas_4.dat" 
