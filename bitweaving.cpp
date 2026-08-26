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

//int C_length = 128; //number of data in the database
int C_length = 2000000; //number of data in the database
int B = 32;  // length of each data
uint32_t *W[32]; // packed bit-planes: one uint32 word holds one bit of 32 rows
uint32_t *V;     // original values, kept for result verification

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

// ---- CPU energy via RAPL ---------------------------------------------------
// Reads the CPU package energy counters exposed by the Linux powercap
// interface (/sys/class/powercap/intel-rapl:N/energy_uj). The counters cover
// the whole socket (all processes), update roughly every millisecond, and are
// often readable only by root since kernel 5.10. In containers/VMs they are
// usually not exposed at all; energy is then simply not reported.
#define RAPL_MAX_DOMAINS 16
static char rapl_energy_path[RAPL_MAX_DOMAINS][160];
static double rapl_range_uj[RAPL_MAX_DOMAINS];
static int rapl_is_dram[RAPL_MAX_DOMAINS];
static int rapl_domains = 0;
static int rapl_pkg_domains = 0;
static int rapl_dram_domains = 0;

// reads <base>/name into name; returns 0 on success
static int rapl_read_name(const char *base, char *name, size_t name_len)
{
  char path[192];
  snprintf(path, sizeof(path), "%s/name", base);
  FILE *f = fopen(path, "r");
  if (f == NULL) return -1;
  if (fgets(name, name_len, f) == NULL) name[0] = '\0';
  fclose(f);
  return 0;
}

static void rapl_add_domain(const char *base, int is_dram)
{
  char path[192];
  FILE *f;
  if (rapl_domains >= RAPL_MAX_DOMAINS) return;
  snprintf(path, sizeof(path), "%s/energy_uj", base);
  f = fopen(path, "r");
  if (f == NULL) return; // not readable without root
  fclose(f);
  snprintf(rapl_energy_path[rapl_domains], sizeof(rapl_energy_path[0]), "%s", path);
  snprintf(path, sizeof(path), "%s/max_energy_range_uj", base);
  double range = 0;
  f = fopen(path, "r");
  if (f != NULL)
  {
    if (fscanf(f, "%lf", &range) != 1) range = 0;
    fclose(f);
  }
  rapl_range_uj[rapl_domains] = range;
  rapl_is_dram[rapl_domains] = is_dram;
  if (is_dram) rapl_dram_domains++; else rapl_pkg_domains++;
  rapl_domains++;
}

static void rapl_init()
{
  for (int i = 0; i < 8; i++)
  {
    char base[128];
    char name[64];
    snprintf(base, sizeof(base), "/sys/class/powercap/intel-rapl:%d", i);
    if (rapl_read_name(base, name, sizeof(name)) != 0) continue;
    if (strncmp(name, "psys", 4) == 0) continue; // overlaps the package domains
    rapl_add_domain(base, 0); // package (cores + uncore, without DRAM)
    // the dram counter is a subdomain of the package and NOT included in it,
    // so package + dram together give the total energy
    for (int j = 0; j < 8; j++)
    {
      char sub[144];
      snprintf(sub, sizeof(sub), "/sys/class/powercap/intel-rapl:%d:%d", i, j);
      if (rapl_read_name(sub, name, sizeof(name)) != 0) continue;
      if (strncmp(name, "dram", 4) == 0) rapl_add_domain(sub, 1);
    }
  }
}

static int rapl_read(double *uj)
{
  for (int d = 0; d < rapl_domains; d++)
  {
    FILE *f = fopen(rapl_energy_path[d], "r");
    if (f == NULL) return -1;
    if (fscanf(f, "%lf", &uj[d]) != 1) { fclose(f); return -1; }
    fclose(f);
  }
  return 0;
}

// energy in joules between two counter snapshots, split into package and dram,
// correcting counter wraparound
static void rapl_delta_j(const double *before, const double *after, double *pkg_j, double *dram_j)
{
  double pkg_uj = 0, dram_uj = 0;
  for (int d = 0; d < rapl_domains; d++)
  {
    double delta = after[d] - before[d];
    if (delta < 0 && rapl_range_uj[d] > 0) delta += rapl_range_uj[d];
    if (rapl_is_dram[d]) dram_uj += delta; else pkg_uj += delta;
  }
  *pkg_j = pkg_uj / 1e6;
  *dram_j = dram_uj / 1e6;
}

// Runs the VBP scan over all periods and returns the number of matching rows.
// The predicate P is a compile-time constant, so only the mask updates that
// this predicate actually needs are compiled into the bit loop:
//   eq/neq            m_eq1                        2 instructions per bit
//   lt/le             m_lt, m_eq2                  5 instructions per bit
//   gt/ge             m_gt, m_eq1                  5 instructions per bit
//   between           all four masks              10 instructions per bit
template <Predicate P>
static long long scan_kernel(int period_num, const __m128i *c1v, const __m128i *c2v)
{
  long long matches = 0;
  // periods are independent of each other, so they can be distributed
  // across OpenMP threads; matches is summed up by the reduction
  #pragma omp parallel for reduction(+:matches)
  for (int period = 0; period < period_num; period++)
  {
    // section #0, #1, #2, #3
    __m128i m_lt_128 = _mm_setzero_si128();
    __m128i m_gt_128 = _mm_setzero_si128();
    __m128i m_eq1_128 = _mm_set1_epi32(0xffffffff);
    __m128i m_eq2_128 = _mm_set1_epi32(0xffffffff);
    for (int i = 0; i < B; i++)
    {
      // one bit of the 128 rows of this period, from the packed bit-plane
      __m128i s_vi_128 = _mm_loadu_si128((const __m128i*)&W[i][period*4]);
      if (P == PRED_GT || P == PRED_GE || P == PRED_BETWEEN)
      {
        // _mm_andnot_si128(a, b) computes (~a & b), so ~c1 needs no extra instruction
        m_gt_128 = _mm_or_si128(m_gt_128, _mm_and_si128(m_eq1_128, _mm_andnot_si128(c1v[i], s_vi_128)));
      }
      if (P == PRED_LT || P == PRED_LE || P == PRED_BETWEEN)
      {
        m_lt_128 = _mm_or_si128(m_lt_128, _mm_and_si128(m_eq2_128, _mm_andnot_si128(s_vi_128, c2v[i])));
      }
      if (P == PRED_GT || P == PRED_GE || P == PRED_EQ || P == PRED_NEQ || P == PRED_BETWEEN)
      {
        m_eq1_128 = _mm_andnot_si128(_mm_xor_si128(s_vi_128, c1v[i]), m_eq1_128);
      }
      if (P == PRED_LT || P == PRED_LE || P == PRED_BETWEEN)
      {
        m_eq2_128 = _mm_andnot_si128(_mm_xor_si128(s_vi_128, c2v[i]), m_eq2_128);
      }
    }
    __m128i period_result;
    switch (P)
    {
      case PRED_LT:  period_result = m_lt_128; break;
      case PRED_LE:  period_result = _mm_or_si128(m_lt_128, m_eq2_128); break;
      case PRED_GT:  period_result = m_gt_128; break;
      case PRED_GE:  period_result = _mm_or_si128(m_gt_128, m_eq1_128); break;
      case PRED_EQ:  period_result = m_eq1_128; break;
      case PRED_NEQ: period_result = _mm_not_si128(m_eq1_128); break;
      default:       period_result = _mm_and_si128(m_gt_128, m_lt_128); break;
    }
    uint32_t r[4];
    memcpy(r, &period_result, sizeof(r));
    matches += __builtin_popcount(r[0]) + __builtin_popcount(r[1]) + __builtin_popcount(r[2]) + __builtin_popcount(r[3]);
  }
  return matches;
}

static void usage(const char *prog)
{
  printf("Usage: %s [options]\n", prog);
  printf("  -n <rows>   number of rows in the database (default 2000000, min 128,\n");
  printf("              rounded down to a multiple of 128; limited only by memory,\n");
  printf("              roughly (bits/8 + 4) bytes per row)\n");
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
        if (n < 128 || n > 2000000000L)
        {
          fprintf(stderr, "Error: -n must be between 128 and 2000000000\n");
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
  rapl_init();
  if (rapl_domains > 0)
    printf("CPU energy (RAPL)   : available, %d package + %d dram domain(s); counters are per socket, not per process\n", rapl_pkg_domains, rapl_dram_domains);
  else
    printf("CPU energy (RAPL)   : not available (powercap not exposed or not readable; bare-metal Linux + root needed)\n");
  double mem_mb = ((double)C_length * B / 8.0 + (double)C_length * sizeof(uint32_t)) / (1024.0 * 1024.0);
  printf("Memory for the data : %.1f MB\n", mem_mb);

  // a bit-plane stores one bit of 32 rows in one uint32 word
  int word_num = C_length / 32;
  for (int bit = 0; bit < B; bit++)
  {
    W[bit] = (uint32_t*)malloc((size_t)word_num * sizeof(uint32_t));
    if (W[bit] == NULL)
    {
      fprintf(stderr, "Error: failed to allocate %.1f MB for the bit-planes, use a smaller -n\n", mem_mb);
      return EXIT_FAILURE;
    }
  }
  V = (uint32_t*)malloc((size_t)C_length * sizeof(uint32_t));
  if (V == NULL)
  {
    fprintf(stderr, "Error: failed to allocate %.1f MB for the data, use a smaller -n\n", mem_mb);
    return EXIT_FAILURE;
  }

  srand( (unsigned)time( NULL ) );
  for (int i = 0; i < C_length; i++)
  {
  	V[i] = rand32() & mask; // generate the data by random
  }
  // pack the bit-planes; word w of plane 'bit' holds bit (B-bit-1) of rows
  // [w*32, w*32+32), the first row in the highest word bit. This is done once
  // up front and is not part of the measured scan time.
  double packStart = wall_time();
  #pragma omp parallel for
  for (int w = 0; w < word_num; w++)
  {
  	for (int bit = 0; bit < B; bit++)
  	{
  	  uint32_t word = 0;
  	  for (int k = 0; k < 32; k++)
  	  {
  	  	word = (word << 1) | ((V[w*32 + k] >> (B-bit-1)) & 1);
  	  }
  	  W[bit][w] = word;
  	}
  }
  printf("Packing the bit-planes took %f s (excluded from the scan times)\n", wall_time() - packStart);
  double total_time = 0;
  double total_pkg_energy = 0;
  double total_dram_energy = 0;
  int energy_loops = 0;
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

	  // build the constant vectors; setup work, not part of the measured scan time
	  __m128i c1v[32], c2v[32];
	  for (int bit = 0; bit < B; bit++)
	  {
	  	c1v[bit] = _mm_set1_epi32(((c1 >> (B-bit-1)) & 1) ? 0xffffffff : 0);
	  	c2v[bit] = _mm_set1_epi32(((c2 >> (B-bit-1)) & 1) ? 0xffffffff : 0);
	  }
	  int period_num = C_length / 128; //4 sections executed in parallel

	  // only the predicate kernel itself is measured (time and energy)
	  double e_before[RAPL_MAX_DOMAINS], e_after[RAPL_MAX_DOMAINS];
	  bool have_energy = (rapl_domains > 0 && rapl_read(e_before) == 0);
	  startTime = wall_time();
	  switch (pred)
	  {
	    case PRED_LT:  matches = scan_kernel<PRED_LT>(period_num, c1v, c2v); break;
	    case PRED_LE:  matches = scan_kernel<PRED_LE>(period_num, c1v, c2v); break;
	    case PRED_GT:  matches = scan_kernel<PRED_GT>(period_num, c1v, c2v); break;
	    case PRED_GE:  matches = scan_kernel<PRED_GE>(period_num, c1v, c2v); break;
	    case PRED_EQ:  matches = scan_kernel<PRED_EQ>(period_num, c1v, c2v); break;
	    case PRED_NEQ: matches = scan_kernel<PRED_NEQ>(period_num, c1v, c2v); break;
	    default:       matches = scan_kernel<PRED_BETWEEN>(period_num, c1v, c2v); break;
	  }
	  endTime = wall_time();
	  double time = endTime - startTime;
	  double pkg_j = -1, dram_j = -1;
	  if (have_energy && rapl_read(e_after) == 0) rapl_delta_j(e_before, e_after, &pkg_j, &dram_j);
	  cout << "Scan time: " << time << ", matching rows: " << matches << endl;
	  if (pkg_j >= 0)
	  {
	  	printf("CPU energy: package %.4f J (%.1f W)", pkg_j, pkg_j / time);
	  	if (rapl_dram_domains > 0) printf(" + DRAM %.4f J (%.1f W)", dram_j, dram_j / time);
	  	printf(" [whole socket]\n");
	  	total_pkg_energy += pkg_j;
	  	total_dram_energy += dram_j;
	  	energy_loops++;
	  }
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
  cout << "Average Time: " << double(total_time / loop_num) << " s (scan kernel only)" << endl;
  if (energy_loops > 0)
  {
    printf("Average package energy: %f J per scan (RAPL, whole socket)\n", total_pkg_energy / energy_loops);
    if (rapl_dram_domains > 0)
      printf("Average DRAM energy: %f J per scan (RAPL, whole socket)\n", total_dram_energy / energy_loops);
  }

  for (int bit = 0; bit < B; bit++) free(W[bit]);
  free(V);
  return EXIT_SUCCESS;
}













