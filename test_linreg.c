/*

file: test_linreg.c
author: LuigiG - LG@THLG.NL
license: MIT

test the ASM linreg implementation

build: clang -std=c11 asm_linreg.s test_linreg.c -o test_linreg

*/

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>

#define _DO_BENCHMARK 1

extern double mean(double arr[], long int n);
extern void linreg(double arr1[], double arr2[], long int n, double *result1, double *result2);

#define _DBL_EPSILON (0.000000001L)
static bool dbleq(double a, double b) {
  return fabs(a - b) < _DBL_EPSILON;
}

// reference implementation
static double ref_test_mean(double arr[], int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }
  return sum / n;
}

// reference implementation
static void ref_test_linreg(double x[], double y[], int n, double *slope, double *intercept) {
  double x_mean = ref_test_mean(x, n);
  double y_mean = ref_test_mean(y, n);
  double numerator = 0.0;
  double denominator = 0.0;
  for (int i = 0; i < n; i++) {
    numerator += (x[i] - x_mean) * (y[i] - y_mean);
    denominator += (x[i] - x_mean) * (x[i] - x_mean);
  }
  *slope = numerator / denominator;
  *intercept = y_mean - (*slope * x_mean);
}

static void generate_svg_out(const char *filename, double x[], double y[], int n, double slope, double intercept) {
  const double PAD = 1.0;
  const double W = 400;
  const double H = 400;
  const double BORD = 20;
  const double VIEWPORT_W = W + BORD + BORD;
  const double VIEWPORT_H = H + BORD + BORD;
  double x_min = x[0];
  double x_max = x[0];
  double y_min = y[0];
  double y_max = y[0];
  for (int i = 1; i < n; i++) {
      if (x[i] < x_min) x_min = x[i];
      if (x[i] > x_max) x_max = x[i];
      if (y[i] < y_min) y_min = y[i];
      if (y[i] > y_max) y_max = y[i];
  }
  x_min -= PAD;
  x_max += PAD;
  y_min -= PAD;
  y_max += PAD;
  FILE *f = fopen(filename, "w");
  assert(f);
  fprintf(f,"<svg width='%f' height='%f' xmlns='http://www.w3.org/2000/svg'>\n", VIEWPORT_W, VIEWPORT_H);
  fprintf(f,"<rect width='100%%' height='100%%' fill='white' />\n");
  // x and y axes
  double x_axis_y = (W + BORD) - W * (0 - y_min) / (y_max - y_min);
  double y_axis_x = (H + BORD) * (0 - x_min) / (x_max - x_min);
  if (x_axis_y < BORD) x_axis_y = BORD;
  if (x_axis_y > (H + BORD)) x_axis_y = (H + BORD);
  if (y_axis_x < BORD) y_axis_x = BORD;
  if (y_axis_x > (H + BORD)) y_axis_x = (H + BORD);
  fprintf(f,"<line x1='%f' y1='%f' x2='%f' y2='%f' stroke='black' stroke-width='2' />\n", BORD, x_axis_y, (BORD + H), x_axis_y); // x-axis
  fprintf(f,"<line x1='%f' y1='%f' x2='%f' y2='%f' stroke='black' stroke-width='2' />\n", y_axis_x, (BORD + H), y_axis_x, BORD); // y-axis
  // the points
  for (int i = 0; i < n; i++) {
    double cx = BORD + W * (x[i] - x_min) / (x_max - x_min);
    double cy = (BORD + H) - H * (y[i] - y_min) / (y_max - y_min);
    fprintf(f,"<circle cx='%f' cy='%f' r='2' fill='blue' />\n", cx, cy);
  }
  // the line
  double x1 = x_min;
  double y1 = slope * x1 + intercept;
  double x2 = x_max;
  double y2 = slope * x2 + intercept;
  double x1_svg = BORD + W * (x1 - x_min) / (x_max - x_min);
  double y1_svg = (H + BORD) - H * (y1 - y_min) / (y_max - y_min);
  double x2_svg = BORD + W * (x2 - x_min) / (x_max - x_min);
  double y2_svg = (H + BORD) - H * (y2 - y_min) / (y_max - y_min);
  fprintf(f,"<line x1='%f' y1='%f' x2='%f' y2='%f' stroke='red' stroke-width='2' />\n", x1_svg, y1_svg, x2_svg, y2_svg);
  fprintf(f,"</svg>\n");
  fclose(f);
}

static uint64_t get_time() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);    
    return  (uint64_t)ts.tv_sec * 1000000000UL + (uint64_t)ts.tv_nsec;
}

int main() {
  double x[] = {0.000,1.000,2.000,3.000,4.000,5.000,6.000,7.000,8.000,9.000,10.000,11.000,12.000,
    13.000,14.000,15.000,16.000,17.000,18.000,19.000,20.000,21.000,22.000,23.000,24.000,25.000,
    26.000,27.000,28.000,29.000,30.000,31.000,32.000,33.000,34.000,};
  double y[] = {0.288,0.838,4.079,10.250,11.262,14.048,11.461,12.791,14.018,22.224,17.557,23.743,
    24.713,28.150,32.643,28.921,33.026,38.332,36.190,41.655,38.742,40.546,41.546,48.274,53.270,
    47.678,53.868,57.131,58.412,56.124,59.963,65.423,68.936,66.615,67.652,};

  int n = sizeof(x) / sizeof(x[0]);
  double slope, intercept;
  double ref_slope, ref_intercept;

  ref_test_linreg(x, y, n, &ref_slope, &ref_intercept);
  linreg(x, y, n, &slope, &intercept);
  assert(dbleq(slope,ref_slope));
  assert(dbleq(intercept,ref_intercept));
  
  printf("Slope: %f -- Slope-ctrl: %f\n", slope, ref_slope);
  printf("Intercept: %f -- Intercept-ctrl: %f\n", intercept, ref_intercept);
  const char *svgfile = "out.svg";
  generate_svg_out(svgfile, x, y, n, slope, intercept);
  printf("Graph written to %s\n", svgfile);
#ifdef _DO_BENCHMARK
  const unsigned REPS = 16777216;
  printf("RUNNING BENCHMARK: NON-VECTORIZED...\n");
  uint64_t t1 = get_time();
  for(int i=0;i<REPS;i++){ref_test_linreg(x, y, n, &ref_slope, &ref_intercept);}
  uint64_t t2 = get_time();
  printf("RUNNING BENCHMARK: VECTORIZED...\n");
  uint64_t t3 = get_time();
  for(int i=0;i<REPS;i++){linreg(x, y, n, &ref_slope, &ref_intercept);}
  uint64_t t4 = get_time();
  uint64_t time_norm = t2-t1;
  uint64_t time_simd = t4-t3;
  double simd_p100 = ((double)time_simd)/((double)time_norm/100.0L);
  printf("%u CYCLES NON-VECTORIZED LINREG TIME : %llu - 100%%\n", REPS, t2-t1);
  printf("%u CYCLES VECTORIZED LINREG TIME     : %llu - %.2f%%\n", REPS, t4-t3, simd_p100);
#endif
  return 0;
}
