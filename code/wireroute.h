/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <omp.h>

typedef struct bend{
    int x;
    int y;
} bend_t;

typedef struct totalCost {
    int maxValue;
    int cost;
} total_cost_t;

typedef struct { /* Define the data structure for wire here */
    int start_x;
    int start_y;
    int end_x;
    int end_y;
    int numBends;
    bend_t bend[2];
} wire_t;

typedef int cost_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif
