/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <linux/limits.h>

using namespace std;

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}

static void initialize(wire_t *wires, int *costs, int dim_x,int dim_y,int num_wires) {
    for (int i = 0; i < num_wires; i++){
        wire_t *cur = &wires[i];
        if(cur->start_x==cur->end_x || cur->start_y==cur->end_y){
            cur->numBends=0;
        }else{
            cur->numBends=1;
            cur->bend[0].x = cur->end_x;
            cur->bend[0].y = cur->start_y;
        }
        wire_t curr=wires[i];
        int start_x = curr.start_x;
        int start_y = curr.start_y;
        int end_x,end_y;
        for(int i=0; i < curr.numBends+1; i++){
            if(i==curr.numBends){
                end_x=curr.end_x;
                end_y=curr.end_y;
            }else{
                end_x=curr.bend[i].x;
                end_y=curr.bend[i].y;
            }
            for(int i = min(start_x,end_x); i < max(end_x,start_x)+1; i++){
                for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
                    costs[i*dim_y+j]+=1;
                }
            }
            if(start_x==end_x){
                start_y=end_y;
                if(end_x<curr.end_x){
                    start_x=end_x+1;
                }else{
                    start_x=end_x-1;
                }
            }else{
                start_x=end_x;
                if(end_y<curr.end_y){
                    start_y=end_y+1;
                }else{
                    start_y=end_y-1;
                }
            }
        }
    }
}

static total_cost_t calculateCost(wire_t* wires,int index, int *costs, int dim_x, int dim_y) {
    int cost=0;
    int maxVal=0;
    wire_t curr= wires[index];
    int start_x=curr.start_x;
    int start_y=curr.start_y;
    int end_x,end_y;
    for(int i=0;i<curr.numBends+1;i++){
        if(i==curr.numBends){
            end_x=curr.end_x;
            end_y=curr.end_y;
        }else{
            end_x=curr.bend[i].x;
            end_y=curr.bend[i].y;
        }
        for(int i = min(start_x,end_x); i < max(end_x,start_x)+1; i++){
            for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
                int curr_cost=costs[i*dim_y+j];
                cost+=curr_cost;
                maxVal=max(maxVal,curr_cost);
            }
        }
        start_x=end_x;
        start_y=end_y;
    }
    total_cost_t total_cost;
    total_cost.cost=cost;
    total_cost.maxValue=maxVal;
    return total_cost;
}

static void updateCosts(wire_t old_wire, wire_t new_wire, int *costs,int dim_x, int dim_y) {
    //removing costs of the old wire
    wire_t curr= old_wire;
    int start_x=curr.start_x;
    int start_y=curr.start_y;
    int end_x,end_y;
    for(int i=0;i<curr.numBends+1;i++){
        if(i==curr.numBends){
            end_x=curr.end_x;
            end_y=curr.end_y;
        }else{
            end_x=curr.bend[i].x;
            end_y=curr.bend[i].y;
        }
        for(int i = min(start_x,end_x); i < max(end_x,start_x)+1; i++){
            for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
                costs[i*dim_y+j]-=1;
            }
        }
        start_x=end_x;
        start_y=end_y;
    }
    //updating costs for new route
    curr= new_wire;
    start_x=curr.start_x;
    start_y=curr.start_y;
    for(int i=0;i<curr.numBends+1;i++){
        if(i==curr.numBends){
            end_x=curr.end_x;
            end_y=curr.end_y;
        }else{
            end_x=curr.bend[i].x;
            end_y=curr.bend[i].y;
        }
        for(int i = min(start_x,end_x); i < max(end_x,start_x)+1; i++){
            for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
            costs[i*dim_y+j]+=1;
            }
        }
        start_x=end_x;
        start_y=end_y;
    }
}

static void update(wire_t *wires, int *costs, int dim_x, int dim_y, int num_wires) {

    // Find better cost for each wire
    for (int i = 0; i < num_wires; i++) {
        int numBends = wires[i].numBends;
        if (numBends == 0) {
            continue;
        }
        // Wires we are modifying
        wire_t oldWire = wires[i];
        wire_t newWire;
        int start_x = oldWire.start_x;
        int start_y = oldWire.start_y;
        int end_x = oldWire.end_x;
        int end_y = oldWire.end_y;
        total_cost_t currCost = calculateCost(wires, i, costs, dim_x, dim_y);
        int sign_x=1,sign_y=1;
        if(start_x > end_x){
            sign_x=-1;
        }
        if(start_y > end_y){
            sign_y=-1;
        }
        
        // Check Horizontal First Paths
        for (int j = 0; j < abs(end_x - start_x); j++) {
            newWire.bend[0].x = start_x + sign_x*(j + 1);
            newWire.bend[0].y = start_y;
            if (start_x + j + 1 == end_x) {
                // One Bend Case
                newWire.numBends = 1;
            }
            else {
                // Two Bend Case
                newWire.bend[1].x = start_x +sign_x*(j + 1);
                newWire.bend[1].y = end_y;
                newWire.numBends = 2;
            }

            // Check if newWire is better than oldWire and replace if so
            updateCosts(oldWire, newWire, costs, dim_x, dim_y);
            total_cost_t newCost = calculateCost(wires, i, costs, dim_x, dim_y);

            if(newCost.maxValue < currCost.maxValue){
                currCost = newCost;
                oldWire = newWire;
            }else if (newCost.maxValue == currCost.maxValue && newCost.cost < currCost.cost) {
                currCost = newCost;
                oldWire = newWire;
            }else {
                updateCosts(newWire, oldWire, costs, dim_x, dim_y);
            }
        } 

        // Check Vertical First Paths
        for (int j = 0; j < abs(end_y - start_y); j++) {
            newWire.bend[0].y = start_y + sign_y*(j + 1);
            newWire.bend[0].x = start_x;
            if (start_y + j + 1 == end_y) {
                // One Bend Case
                newWire.numBends = 1;
            }
            else {
                // Two Bend Case
                newWire.bend[1].y = start_y + sign_y*(j + 1);
                newWire.bend[1].x = end_x;
                newWire.numBends = 2;
            }

            // Check if newWire is better than oldWire and replace if so
            updateCosts(oldWire, newWire, costs, dim_x, dim_y);
            total_cost_t newCost = calculateCost(wires, i, costs, dim_x, dim_y);

            if(newCost.maxValue < currCost.maxValue){
                currCost = newCost;
                oldWire = newWire;
            }else if (newCost.maxValue == currCost.maxValue && newCost.cost < currCost.cost) {
                currCost = newCost;
                oldWire = newWire;
            }else {
                updateCosts(newWire, oldWire, costs, dim_x, dim_y);
            }
        }

        // Create Random Path
        // Horizontal first
        wire_t hWire = oldWire;
        hWire.bend[0].x = (rand() % (abs(end_x-start_x)))*sign_x + start_x;
        hWire.bend[0].y = start_y;
        if (hWire.bend[0].x == end_x) {
            hWire.numBends = 1;
        }
        else {
            hWire.numBends = 2;
            hWire.bend[1].x = hWire.bend[0].x;
            hWire.bend[1].y = end_y;
        }

        // Vertical First
        wire_t vWire = oldWire;
        vWire.bend[0].y = (rand() % (abs(end_y-start_y)))*sign_y + start_y;
        vWire.bend[0].x = start_x;
        if (vWire.bend[0].y == end_y) {
            vWire.numBends = 1;
        }
        else {
            vWire.numBends = 2;
            vWire.bend[1].y = vWire.bend[0].y;
            vWire.bend[1].x = end_x;
        }

        int h_or_v = rand() % 2;
        int annealP = rand() % 100;
        // Replace best wire with random wire
        if (annealP < 10) {
            newWire = (h_or_v) ? hWire : vWire;
            updateCosts(oldWire, newWire, costs, dim_x, dim_y);
        }
    }
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    double SA_prob = get_option_float("-p", 0.1f);
    int SA_iters = get_option_int("-i", 5);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Probability parameter for simulated annealing: %lf.\n", SA_prob);
    printf("Number of simulated annealing iterations: %d\n", SA_iters);
    printf("Input file: %s\n", input_filename);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int dim_x, dim_y;
    int num_of_wires;

    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    fscanf(input, "%d\n", &num_of_wires);

    wire_t *wires = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
    /* Read the grid dimension and wire information from file */

    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    /* Initialize cost matrix */

    /* Initailize additional data structures needed in the algorithm */
    for (int i = 0; i < num_of_wires; i++) {
        fscanf(input, "%d %d %d %d\n", &(wires[i].start_x), &(wires[i].start_y), &(wires[i].end_x), &(wires[i].end_y));
    }

    /* Conduct initial wire placement */
    initialize(wires,costs,dim_x,dim_y,num_of_wires);

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;

    /**
     * Implement the wire routing algorithm here
     * Feel free to structure the algorithm into different functions
     * Don't use global variables.
     * Use OpenMP to parallelize the algorithm.
     */
    // for (int i = 0; i < SA_iters; i++) {
    //     update(wires, costs, dim_x, dim_y, num_of_wires);
    // }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    /* Write wires and costs to files */
    char resolved_path[PATH_MAX];
    realpath(input_filename, resolved_path);
    char *base = basename(resolved_path);
    std::string baseS = std::string(base);
    size_t lastindex = baseS.find_last_of("."); 
    string rawname = baseS.substr(0, lastindex); 

    std::stringstream OutputCosts;
    OutputCosts << "outputs//costs_" << rawname.c_str() << "_" << num_of_threads << ".txt";
    std::string OutputCostsFile = OutputCosts.str();

    std::stringstream OutputWires;
    OutputWires << "outputs//output_" << rawname.c_str() << "_" << num_of_threads << ".txt";
    std::string OutputWiresFile = OutputWires.str();

    const char *ocf = OutputCostsFile.c_str();
    FILE *costFile = fopen(ocf, "w");
    if (!costFile) {
        printf("Unable to open file: %s.\n", ocf);
        return 1;
    }
    const char *owf = OutputWiresFile.c_str();
    FILE *outFile = fopen(owf, "w");
    if (!outFile) {
        printf("sad\n");
        printf("Unable to open file: %s.\n", owf);
        return 1;
    }

    // Write to cost file
    fprintf(costFile, "%d %d\n", dim_x, dim_y);
    for(int i = 0; i < dim_y; i++){
        for(int j = 0; j < dim_x; j++){
            fprintf(costFile, "%d ", costs[i*dim_x+j]);
        }
        fprintf(costFile, "\n");
    }

    // Write to output wire file
    fprintf(outFile, "%d %d\n", dim_x, dim_y);
    fprintf(outFile, "%d\n", num_of_wires);
    for (int i = 0; i < num_of_wires; i++) {
        wire_t curr = wires[i];
        int start_x = curr.start_x;
        int start_y = curr.start_y;
        int end_x,end_y;
        for(int i = 0; i < curr.numBends+1; i++){
            if(i==curr.numBends){
                end_x=curr.end_x;
                end_y=curr.end_y;
            }else{
                end_x=curr.bend[i].x;
                end_y=curr.bend[i].y;
            }
            if(start_x==end_x){
                int sign;
                if(start_y<end_y){
                    sign=1;
                }else{
                    sign=-1;
                }
                for(int j = 0; j < abs(end_y-start_y)+1; j++){
                    fprintf(outFile, "%d %d ", start_x, start_y+j*(sign));
                }
                start_y=end_y;
                if(end_x<curr.end_x){
                    start_x=end_x+1;
                }else{
                    start_x=end_x-1;
                }
            }else{
                int sign;
                if(start_x<end_x){
                    sign=1;
                }else{
                    sign=-1;
                }
                for(int j = 0; j < abs(end_x-start_x)+1; j++){
                    fprintf(outFile, "%d %d ", start_x+j*(sign),start_y);
                }
                start_x=end_x;
                if(end_y<curr.end_y){
                    start_y=end_y+1;
                }else{
                    start_y=end_y-1;
                }
            }
        }
        fprintf(outFile, "\n");
    }

    // Close all files
    fclose(input);
    fclose(costFile);
    fclose(outFile);

    return 0;
}

