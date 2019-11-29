#include <stdio.h>
#include <math.h>
#include <omp.h>
int main(int argc, char *argv[])
{
  long npt=10;
  double xmin=-5;
  double xmax=5;
  double dx=(xmax-xmin)/(npt-1);

  double tot=0;


  double t1=omp_get_wtime();
#pragma omp parallel 
  {
    double mytot=0;
#pragma omp for
    for (long i=0;i<npt-1;i++) {
      double x_left=xmin+i*dx;
      double x_right=xmin+(i+1)*dx;
      double x0=0.5*(x_left+x_right);
      double f0=exp(-0.5*x0*x0);
      mytot=mytot+f0*dx;
    }
    //printf("my integral is %12.6f\n",mytot);
#pragma omp critical
    tot+=mytot;
  }
  double t2=omp_get_wtime();
  printf("tot is now %12.6f\n",tot);
  printf("elapsed time is %12.4e\n",t2-t1);
  
}
