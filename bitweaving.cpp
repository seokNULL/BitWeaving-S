//Copyright <2020> <Tianhong SHEN>

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define __STDC_LIMIT_MACROS    1

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>

#include "SIMD_operations.h"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
// Compiler name
#define MACTOSTR(x)    #x
#define MACROVALUESTR(x)    MACTOSTR(x)
#if defined(__ICL)    // Intel C++
#  if defined(__VERSION__)
#    define COMPILER_NAME    "Intel C++ " __VERSION__
#  elif defined(__INTEL_COMPILER_BUILD_DATE)
#    define COMPILER_NAME    "Intel C++ (" MACROVALUESTR(__INTEL_COMPILER_BUILD_DATE) ")"
#  else
#    define COMPILER_NAME    "Intel C++"
#  endif    // #  if defined(__VERSION__)
#elif defined(_MSC_VER)    // Microsoft VC++
#  if defined(_MSC_FULL_VER)
#    define COMPILER_NAME    "Microsoft VC++ (" MACROVALUESTR(_MSC_FULL_VER) ")"
#  elif defined(_MSC_VER)
#    define COMPILER_NAME    "Microsoft VC++ (" MACROVALUESTR(_MSC_VER) ")"
#  else
#    define COMPILER_NAME    "Microsoft VC++"
#  endif    // #  if defined(_MSC_FULL_VER)
#elif defined(__GNUC__)    // GCC
#  if defined(__CYGWIN__)
#    define COMPILER_NAME    "GCC(Cygmin) " __VERSION__
#  elif defined(__MINGW32__)
#    define COMPILER_NAME    "GCC(MinGW) " __VERSION__
#  else
#    define COMPILER_NAME    "GCC " __VERSION__
#  endif    // #  if defined(_MSC_FULL_VER)
#else
#  define COMPILER_NAME    "Unknown Compiler"
#endif    // #if defined(__ICL)    // Intel C++

#define __forceinline __attribute__((always_inline))

using namespace std;

#define MAX_ROWS 8000000

//int C_length = 128; //number of data in the database
int C_length = 2000000; //number of data in the database
int B = 32;  // length of each data
int C[32][MAX_ROWS];
uint32_t V[MAX_ROWS]; // original values, kept for result verification

// Supported comparison predicates
enum Predicate { PRED_LT, PRED_LE, PRED_GT, PRED_GE, PRED_EQ, PRED_NEQ, PRED_BETWEEN };

static const char *pred_name(Predicate p)
{
  switch (p)
  {
    case PRED_LT:  return "x < c";
    case PRED_LE:  return "x <= c";
    case PRED_GT:  return "x > c";
    case PRED_GE:  return "x >= c";
    case PRED_EQ:  return "x == c";
    case PRED_NEQ: return "x != c";
    default:       return "c1 < x < c2";
  }
}

static uint32_t rand32()
{
  return (((uint32_t)rand() << 17) ^ (uint32_t)rand());
}

// wall-clock time in seconds; clock() cannot be used with OpenMP because it
// sums the CPU time of all threads
static double wall_time()
{
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

static void usage(const char *prog)
{
  printf("Usage: %s [options]\n", prog);
  printf("  -n <rows>   number of rows in the database (128..%d, default 2000000,\n", MAX_ROWS);
  printf("              rounded down to a multiple of 128)\n");
  printf("  -b <bits>   number of bits per value (1..32, default 32)\n");
  printf("  -l <loops>  number of measurement loops (default 50)\n");
  printf("  -p <pred>   predicate: lt | le | gt | ge | eq | neq | between (default between)\n");
  printf("  -x <value>  constant c (or c1 for 'between'); random per loop if omitted\n");
  printf("  -y <value>  constant c2, only used with 'between'; random per loop if omitted\n");
  printf("  -t <num>    number of OpenMP threads (default: all available)\n");
  printf("  -h          show this help\n");
}

int main (int argc, char **argv)
{
  int loop_num = 50;
  Predicate pred = PRED_BETWEEN;
  bool has_x = false, has_y = false;
  uint32_t user_c1 = 0, user_c2 = 0;
  int num_threads = 0; // 0 = OpenMP default (all available)

  int opt;
  while ((opt = getopt(argc, argv, "n:b:l:p:x:y:t:h")) != -1)
  {
    switch (opt)
    {
      case 'n':
      {
        long n = atol(optarg);
        if (n < 128 || n > MAX_ROWS)
        {
          fprintf(stderr, "Error: -n must be between 128 and %d\n", MAX_ROWS);
          return EXIT_FAILURE;
        }
        if (n % 128 != 0)
        {
          fprintf(stderr, "Warning: -n %ld rounded down to %ld (multiple of 128)\n", n, n - n % 128);
          n -= n % 128;
        }
        C_length = (int)n;
        break;
      }
      case 'b':
        B = atoi(optarg);
        if (B < 1 || B > 32)
        {
          fprintf(stderr, "Error: -b must be between 1 and 32\n");
          return EXIT_FAILURE;
        }
        break;
      case 'l':
        loop_num = atoi(optarg);
        if (loop_num < 1)
        {
          fprintf(stderr, "Error: -l must be at least 1\n");
          return EXIT_FAILURE;
        }
        break;
      case 'p':
        if      (strcmp(optarg, "lt") == 0)      pred = PRED_LT;
        else if (strcmp(optarg, "le") == 0)      pred = PRED_LE;
        else if (strcmp(optarg, "gt") == 0)      pred = PRED_GT;
        else if (strcmp(optarg, "ge") == 0)      pred = PRED_GE;
        else if (strcmp(optarg, "eq") == 0)      pred = PRED_EQ;
        else if (strcmp(optarg, "neq") == 0)     pred = PRED_NEQ;
        else if (strcmp(optarg, "between") == 0) pred = PRED_BETWEEN;
        else
        {
          fprintf(stderr, "Error: unknown predicate '%s'\n", optarg);
          usage(argv[0]);
          return EXIT_FAILURE;
        }
        break;
      case 'x':
        user_c1 = (uint32_t)strtoul(optarg, NULL, 0);
        has_x = true;
        break;
      case 'y':
        user_c2 = (uint32_t)strtoul(optarg, NULL, 0);
        has_y = true;
        break;
      case 't':
        num_threads = atoi(optarg);
        if (num_threads < 1)
        {
          fprintf(stderr, "Error: -t must be at least 1\n");
          return EXIT_FAILURE;
        }
        break;
      case 'h':
        usage(argv[0]);
        return EXIT_SUCCESS;
      default:
        usage(argv[0]);
        return EXIT_FAILURE;
    }
  }

  uint32_t mask = (B == 32) ? 0xffffffffu : ((1u << B) - 1);
  if (has_x && user_c1 > mask)
  {
    fprintf(stderr, "Error: -x %u does not fit in %d bits\n", user_c1, B);
    return EXIT_FAILURE;
  }
  if (has_y && user_c2 > mask)
  {
    fprintf(stderr, "Error: -y %u does not fit in %d bits\n", user_c2, B);
    return EXIT_FAILURE;
  }

#ifdef _OPENMP
  if (num_threads > 0) omp_set_num_threads(num_threads);
  int active_threads = omp_get_max_threads();
#else
  int active_threads = 1;
  if (num_threads > 1) fprintf(stderr, "Warning: built without OpenMP, -t ignored (single thread)\n");
#endif

  printf("*****Configuration: ********************************************************************************\n");
  printf("Rows (dataset size) : %d\n", C_length);
  printf("Bits per value      : %d\n", B);
  printf("Loops               : %d\n", loop_num);
  printf("Predicate           : %s\n", pred_name(pred));
  if (pred == PRED_BETWEEN)
  {
    if (has_x) printf("Constant c1         : %u\n", user_c1); else printf("Constant c1         : random per loop\n");
    if (has_y) printf("Constant c2         : %u\n", user_c2); else printf("Constant c2         : random per loop\n");
  }
  else
  {
    if (has_x) printf("Constant c          : %u\n", user_c1); else printf("Constant c          : random per loop\n");
  }
  printf("Threads (OpenMP)    : %d\n", active_threads);
  printf("Compiler            : %s\n", COMPILER_NAME);

  srand( (unsigned)time( NULL ) );
  for (int i = 0; i < C_length; i++)
  {
  	uint32_t data = rand32() & mask; // generate the data by random
  	V[i] = data;
  	for (int bit = 0; bit < B; bit++)
  	{
  	  C[bit][i] = (data >> (B-bit-1)) & 1;
  	}
  }
  double total_time = 0;
  for (int loop = 0; loop < loop_num; loop++)
  {
	  printf("*****Loop #%d***************************************************************************************\n", loop);
	  uint32_t c1, c2;
	  if (pred == PRED_BETWEEN)
	  {
	  	c1 = has_x ? user_c1 : (rand32() & mask);
	  	c2 = has_y ? user_c2 : (rand32() & mask);
	  	if (c1 == c2) {c2 = (c2+1) & mask; }
	  	if (c1 > c2)
	  	{
	  		uint32_t temp = c1;
	  		c1 = c2;
	  		c2 = temp;
	  	}
	  	printf("C1 and C2: %u  %u\n", c1, c2);
	  }
	  else
	  {
	  	c1 = has_x ? user_c1 : (rand32() & mask);
	  	c2 = c1; // single-constant predicates use the same constant on both sides
	  	printf("C: %u\n", c1);
	  }
	  double startTime, endTime;
	  long long matches = 0;

	  startTime = wall_time();
	  vector<int> c1_bits;
	  for (int bit = 0; bit < B; bit++) {c1_bits.push_back((c1 >> (B-bit-1)) & 1);}
	  vector<int> c2_bits;
	  for (int bit = 0; bit < B; bit++) {c2_bits.push_back((c2 >> (B-bit-1)) & 1);}
	  
	  vector<__m128i> c1_128;
	  for (int bit = 0; bit < B; bit++) {c1_128.push_back(_mm_set_epi32(c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff));}
	  	  
	  vector<__m128i> c2_128;
      for (int bit = 0; bit < B; bit++) {c2_128.push_back(_mm_set_epi32(c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff));}
	  
	  int section_num = C_length / 32;
	  int period_num = C_length / 128; //4 sections executed in parallel
	  // periods are independent of each other, so they can be distributed
	  // across OpenMP threads; matches is summed up by the reduction
	  #pragma omp parallel for reduction(+:matches)
	  for (int period = 0; period < period_num; period++)
	  {
		// section #0, #1, #2, #3 
		__m128i m_lt_128 = _mm_set_epi32(0,0,0,0);
		__m128i m_gt_128 = _mm_set_epi32(0,0,0,0);
		__m128i m_eq1_128 = _mm_set_epi32(0xffffffff,0xffffffff,0xffffffff,0xffffffff);
		__m128i m_eq2_128 = _mm_set_epi32(0xffffffff,0xffffffff,0xffffffff,0xffffffff);
		//print128i_4(m_lt_128); print128i_4(m_gt_128); print128i_4(m_eq1_128); print128i_4(m_eq2_128);
		
		for (int i = 0; i < B; i++)
		{
	  	  int s_vi_0 = bits2int(C[i][period*128], C[i][period*128+1], C[i][period*128+2], C[i][period*128+3],C[i][period*128+4], C[i][period*128+5], C[i][period*128+6], C[i][period*128+7],C[i][period*128+8], C[i][period*128+9], C[i][period*128+10], C[i][period*128+11],C[i][period*128+12], C[i][period*128+13], C[i][period*128+14], C[i][period*128+15],C[i][period*128+16], C[i][period*128+17], C[i][period*128+18], C[i][period*128+19],C[i][period*128+20], C[i][period*128+21], C[i][period*128+22], C[i][period*128+23],C[i][period*128+24], C[i][period*128+25], C[i][period*128+26], C[i][period*128+27],C[i][period*128+28], C[i][period*128+29], C[i][period*128+30], C[i][period*128+31]);
	  	  //printf("%d\n", s_vi_0);
	  	  int s_vi_1 = bits2int(C[i][period*128+32], C[i][period*128+1+32], C[i][period*128+2+32], C[i][period*128+3+32],C[i][period*128+4+32], C[i][period*128+5+32], C[i][period*128+6+32], C[i][period*128+7+32],C[i][period*128+8+32], C[i][period*128+9+32], C[i][period*128+10+32], C[i][period*128+11+32],C[i][period*128+12+32], C[i][period*128+13+32], C[i][period*128+14+32], C[i][period*128+15+32],C[i][period*128+16+32], C[i][period*128+17+32], C[i][period*128+18+32], C[i][period*128+19+32],C[i][period*128+20+32], C[i][period*128+21+32], C[i][period*128+22+32], C[i][period*128+23+32],C[i][period*128+24+32], C[i][period*128+25+32], C[i][period*128+26+32], C[i][period*128+27+32],C[i][period*128+28+32], C[i][period*128+29+32], C[i][period*128+30+32], C[i][period*128+31+32]);
	  	  int s_vi_2 = bits2int(C[i][period*128+64], C[i][period*128+1+64], C[i][period*128+2+64], C[i][period*128+3+64],C[i][period*128+4+64], C[i][period*128+5+64], C[i][period*128+6+64], C[i][period*128+7+64],C[i][period*128+8+64], C[i][period*128+9+64], C[i][period*128+10+64], C[i][period*128+11+64],C[i][period*128+12+64], C[i][period*128+13+64], C[i][period*128+14+64], C[i][period*128+15+64],C[i][period*128+16+64], C[i][period*128+17+64], C[i][period*128+18+64], C[i][period*128+19+64],C[i][period*128+20+64], C[i][period*128+21+64], C[i][period*128+22+64], C[i][period*128+23+64],C[i][period*128+24+64], C[i][period*128+25+64], C[i][period*128+26+64], C[i][period*128+27+64],C[i][period*128+28+64], C[i][period*128+29+64], C[i][period*128+30+64], C[i][period*128+31+64]);
	  	  int s_vi_3 = bits2int(C[i][period*128+96], C[i][period*128+1+96], C[i][period*128+2+96], C[i][period*128+3+96],C[i][period*128+4+96], C[i][period*128+5+96], C[i][period*128+6+96], C[i][period*128+7+96],C[i][period*128+8+96], C[i][period*128+9+96], C[i][period*128+10+96], C[i][period*128+11+96],C[i][period*128+12+96], C[i][period*128+13+96], C[i][period*128+14+96], C[i][period*128+15+96],C[i][period*128+16+96], C[i][period*128+17+96], C[i][period*128+18+96], C[i][period*128+19+96],C[i][period*128+20+96], C[i][period*128+21+96], C[i][period*128+22+96], C[i][period*128+23+96],C[i][period*128+24+96], C[i][period*128+25+96], C[i][period*128+26+96], C[i][period*128+27+96],C[i][period*128+28+96], C[i][period*128+29+96], C[i][period*128+30+96], C[i][period*128+31+96]);
	  	  __m128i s_vi_128 = _mm_set_epi32(s_vi_3, s_vi_2, s_vi_1, s_vi_0);  //remind the order
	  	  m_gt_128 = _mm_or_si128(m_gt_128, _mm_and_si128(m_eq1_128, _mm_and_si128(_mm_not_si128(c1_128[i]), s_vi_128)));
	  	  m_lt_128 = _mm_or_si128(m_lt_128, _mm_and_si128(m_eq2_128, _mm_and_si128(c2_128[i], _mm_not_si128(s_vi_128))));
		  m_eq1_128 = _mm_and_si128(m_eq1_128, _mm_not_si128(_mm_xor_si128(s_vi_128, c1_128[i])));
		  m_eq2_128 = _mm_and_si128(m_eq2_128, _mm_not_si128(_mm_xor_si128(s_vi_128, c2_128[i])));
		}
		__m128i period_result;
		switch (pred)
		{
		  case PRED_LT:  period_result = m_lt_128; break;
		  case PRED_LE:  period_result = _mm_or_si128(m_lt_128, m_eq2_128); break;
		  case PRED_GT:  period_result = m_gt_128; break;
		  case PRED_GE:  period_result = _mm_or_si128(m_gt_128, m_eq1_128); break;
		  case PRED_EQ:  period_result = m_eq1_128; break;
		  case PRED_NEQ: period_result = _mm_not_si128(m_eq1_128); break;
		  default:       period_result = _mm_and_si128(m_gt_128, m_lt_128); break;
		}
		//print128i_b(period_result);
		uint32_t r[4];
		memcpy(r, &period_result, sizeof(r));
		matches += __builtin_popcount(r[0]) + __builtin_popcount(r[1]) + __builtin_popcount(r[2]) + __builtin_popcount(r[3]);
	  }
	  endTime = wall_time();
	  double time = endTime - startTime;
	  cout << "Time: " << time << ", matching rows: " << matches << endl;
	  total_time += time;
	  //cout << total_time << endl;

	  // verify against a plain scalar scan (not part of the measured time)
	  long long expected = 0;
	  for (int i = 0; i < C_length; i++)
	  {
	  	bool match;
	  	switch (pred)
	  	{
	  	  case PRED_LT:  match = (V[i] <  c1); break;
	  	  case PRED_LE:  match = (V[i] <= c1); break;
	  	  case PRED_GT:  match = (V[i] >  c1); break;
	  	  case PRED_GE:  match = (V[i] >= c1); break;
	  	  case PRED_EQ:  match = (V[i] == c1); break;
	  	  case PRED_NEQ: match = (V[i] != c1); break;
	  	  default:       match = (V[i] > c1 && V[i] < c2); break;
	  	}
	  	if (match) expected++;
	  }
	  if (expected != matches)
	  {
	  	printf("VERIFICATION FAILED: scalar scan found %lld matching rows\n", expected);
	  }
  }
  printf("*****Summary: **************************************************************************************\n");
  printf("%d Rows of data, each data represented by %d bits.\n", C_length, B);
  printf("Predicate: %s, %d OpenMP thread(s).\n", pred_name(pred), active_threads);
  cout << "Repeated for " << loop_num << " times. " << endl;
  cout << "Average Time: " << double(total_time / loop_num) << " s" << endl;

  return EXIT_SUCCESS;
}













