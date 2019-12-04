#include <stdio.h>
#include <math.h>
#include <omp.h>
int main(int argc, char *argv[])
{
  int npt=100000;
  double xmin=-5;
  double xmax=5;
  double dx=(xmax-xmin)/(npt-1);

  double tot=0;
#pragma omp parallel for
  for (int i=0;i<npt-1;i++) {
    double x_left=xmin+i*dx;
    double x_right=xmin+(i+1)*dx;
    double x0=0.5*(x_left+x_right);
    double f0=exp(-0.5*x0*x0);
    tot=tot+f0*dx;
  }
  printf("my integral is %12.6f\n",tot);
    
}
