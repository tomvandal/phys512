#we can generate timing from the command line via "python -m cProfile -o simple_hist_timing.dat simple_hist_2d.py"
import pstats
p=pstats.Stats('simple_hist_timing.dat')
p.sort_stats('time')
p.print_stats(10)


