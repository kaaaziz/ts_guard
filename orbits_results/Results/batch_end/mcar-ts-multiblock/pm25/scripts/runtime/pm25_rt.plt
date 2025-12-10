set terminal postscript eps enhanced color "Helvetica" 30
set output "runtime/plots/pm25_rt.eps"

set xrange [10:100]
set xtics 10,10
#set log y

set key above width -2 vertical maxrows 3
set tmargin 4.0

set xlabel "percentage of time series with missing values"
set ylabel "running time (microseconds)" offset 1.5 

plot\
	'runtime/values/orbits_k3_runtime.txt' index 0 using 1:2 title 'orbits\_k3' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "cyan" pointsize 1.2, \
	'runtime/values/orbits_k2_runtime.txt' index 0 using 1:2 title 'orbits\_k2' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "blue" pointsize 1.2, \


set output "runtime/plots/pm25_rt_log.eps"
set log y

plot\
	'runtime/values/orbits_k3_runtime.txt' index 0 using 1:2 title 'orbits\_k3' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "cyan" pointsize 1.2, \
	'runtime/values/orbits_k2_runtime.txt' index 0 using 1:2 title 'orbits\_k2' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "blue" pointsize 1.2, \


