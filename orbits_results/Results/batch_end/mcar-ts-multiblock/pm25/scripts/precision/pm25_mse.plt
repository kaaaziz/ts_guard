set terminal postscript eps enhanced color "Helvetica" 30
set output "error/plots/pm25_mse.eps"

set xrange [10-1:100+1]
set xtics 10,10
set yrange [0:2]
#set log y

set key above width -2 vertical maxrows 3
set tmargin 4.0

set xlabel "percentage of time series with missing values"
set ylabel "mean squared error" offset 1.5 

plot\
	'error/mse/MSE_orbits_k3.dat' index 0 using 1:2 title 'orbits\_k3' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "cyan" pointsize 1.2, \
	'error/mse/MSE_orbits_k2.dat' index 0 using 1:2 title 'orbits\_k2' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "blue" pointsize 1.2, \


set output "error/plots/pm25_rmse.eps"
set ylabel "root mean squared error" offset 1.5 

plot\
	'error/rmse/RMSE_orbits_k3.dat' index 0 using 1:2 title 'orbits\_k3' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "cyan" pointsize 1.2, \
	'error/rmse/RMSE_orbits_k2.dat' index 0 using 1:2 title 'orbits\_k2' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "blue" pointsize 1.2, \

set output "error/plots/pm25_mae.eps"
set ylabel "mean absolute error" offset 1.5 

plot\
	'error/mae/MAE_orbits_k3.dat' index 0 using 1:2 title 'orbits\_k3' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "cyan" pointsize 1.2, \
	'error/mae/MAE_orbits_k2.dat' index 0 using 1:2 title 'orbits\_k2' with linespoints lt 8 lw 3 pt 7 lc rgbcolor "blue" pointsize 1.2, \
