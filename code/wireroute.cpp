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

//update val is either 1,-1
static void update_route(wire_t wire,int *costs,int dim_x,int dim_y,int updateVal){
        int start_x = wire.start_x;
        int start_y = wire.start_y;
        int end_x,end_y;
        for(int i=0; i < wire.numBends+1; i++){
            if(i==wire.numBends){
                end_x=wire.end_x;
                end_y=wire.end_y;
            }else{
                end_x=wire.bend[i].x;
                end_y=wire.bend[i].y;
            }
            if(start_x==end_x){
                for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
                    if (i>0 && wire.bend[i-1].y==j){
                        continue;
                    }
                    costs[j*dim_y+start_x]+=updateVal;
                } 
                start_y=end_y;
            }else{
                for(int k = min(start_x,end_x); k < max(end_x,start_x)+1; k++){
                    if (i>0 && wire.bend[i-1].x==k){
                        continue;
                    }
                    costs[start_y*dim_y+k]+=updateVal;
                }
                start_x=end_x;
            }
        }
}

static void initialize(wire_t *wires, int *costs, int dim_x,int dim_y,int num_wires) {
    //looks messy maybe optimize
    for (int i = 0; i < num_wires; i++){
        wire_t *init_wire = &wires[i];
        if(init_wire->start_x==init_wire->end_x || init_wire->start_y==init_wire->end_y){
            init_wire->numBends=0;
        }else{
            init_wire->numBends=1;
            init_wire->bend[0].x = init_wire->end_x;
            init_wire->bend[0].y = init_wire->start_y;
        }
        update_route(*init_wire,costs,dim_x,dim_y,1);
    }
}

static total_cost_t calculateCost(wire_t curr, int *costs, int dim_x, int dim_y) {
    int cost=0;
    int maxVal=0;
    int start_x=curr.start_x;
    int start_y=curr.start_y;
    int end_x,end_y;
    int currCost;
    for(int i=0; i < curr.numBends+1; i++){
        if(i==curr.numBends){
            end_x=curr.end_x;
            end_y=curr.end_y;
        }else{
            end_x=curr.bend[i].x;
            end_y=curr.bend[i].y;
        }
        if(start_x==end_x){
            for(int j = min(start_y,end_y); j < max(end_y,start_y)+1; j++){
                if (i>0 && curr.bend[i-1].y==j){
                    continue;
                }
                currCost=costs[j*dim_y+start_x];
            
            } 
            start_y=end_y;
        }else{
            for(int k = min(start_x,end_x); k < max(end_x,start_x)+1; k++){
                if (i>0 && curr.bend[i-1].x==k){
                    continue;
                }
                currCost=costs[start_y*dim_y+k];
            }
            start_x=end_x;
        }
        cost+=currCost;
        maxVal=max(maxVal,currCost);
    }
    total_cost_t total_cost;
    total_cost.cost=cost;
    total_cost.maxValue=maxVal;
    return total_cost;
}

static void updateCosts(wire_t old_wire, wire_t new_wire, int *costs,int dim_x, int dim_y) {
    //removing costs of the old wire
    update_route(old_wire,costs,dim_x,dim_y,-1);
    update_route(new_wire,costs,dim_x,dim_y,1);
}

static void update(wire_t *wires, int *costs, int dim_x, int dim_y, int num_wires, int random_prob) {
    // Find better cost for each wire
    for (int i = 0; i < num_wires; i++) {
        int numBends = wires[i].numBends;
        if (numBends == 0) {
            continue;
        }

        // Wires we are modifying
        wire_t oldWire = wires[i];
        int start_x = oldWire.start_x;
        int start_y = oldWire.start_y;
        int end_x = oldWire.end_x;
        int end_y = oldWire.end_y;
        total_cost_t currCost = calculateCost(oldWire,costs, dim_x, dim_y);
        int xDist = abs(start_x - end_x);
        int yDist = abs(start_y - end_y);

        int sign_x=1,sign_y=1;
        if(start_x > end_x){
            sign_x=-1;
        }
        if(start_y > end_y){
            sign_y=-1;
        }

        int threadCount = omp_get_num_threads();
        cost_t *subsetCostsArray = (cost_t *)calloc(xDist * yDist * threadCount, sizeof(cost_t));
        wire_t threadWireArray[threadCount];
        #pragma omp parallel //private(subsetCosts, oldWireRel, newWireRel, tempOldWire)
        {
            // Initialize sub-set of cost matrix
            int threadNum = omp_get_thread_num();
            cost_t *subsetCosts = &subsetCostsArray[threadNum * xDist * yDist];
    
            // Create Relative copies of our wires to their bounding box
            wire_t oldWireRel, newWireRel;
            int boundingTopCornerX = start_x, boundingTopCornerY = start_y; // Top Corners of Bounding Box
            oldWireRel.start_x = 0;
            oldWireRel.end_x = xDist;
            oldWireRel.start_y = 0;
            oldWireRel.end_y = yDist;
            if(sign_x == -1){
                oldWireRel.start_x = xDist;
                oldWireRel.end_x = 0;
                boundingTopCornerX = start_x - xDist;
            }
            if(sign_y == -1){
                oldWireRel.start_x = yDist;
                oldWireRel.end_x = 0;
                boundingTopCornerY = start_y - yDist;
            }
            // Get relative start_x and start_y
            int rel_start_x = oldWireRel.start_x;
            int rel_start_y = oldWireRel.start_y;
            int rel_end_x = oldWireRel.end_x;
            int rel_end_y = oldWireRel.end_y;

            // Copy over costs array into local copy
            for (int h = 0; h < xDist; h++) {
                for (int g = 0; g < yDist; g++) {
                    subsetCosts[g*yDist+h] = costs[(boundingTopCornerY+g)*dim_y + (boundingTopCornerX+h)];
                }
            }
            

            // Check Horizontal First Paths
            #pragma omp for
            for (int j = 0; j < xDist; j++) {
                newWireRel.bend[0].x = rel_start_x + sign_x*(j + 1);
                newWireRel.bend[0].y = rel_start_y;
                if (rel_start_x + sign_x*(j + 1) == rel_end_x) {
                    // One Bend Case
                    newWireRel.numBends = 1;
                }
                else {
                    // Two Bend Case
                    newWireRel.bend[1].x = rel_start_x + sign_x*(j + 1);
                    newWireRel.bend[1].y = rel_end_y;
                    newWireRel.numBends = 2;
                }
                // Check if newWire is better than oldWire and replace if so
                updateCosts(oldWireRel, newWireRel, subsetCosts, xDist+1, yDist+1);
                total_cost_t newCost = calculateCost(newWireRel, subsetCosts, xDist+1, yDist+1);
                if(newCost.maxValue < currCost.maxValue){
                    currCost = newCost;
                    oldWireRel = newWireRel;
                }else if (newCost.maxValue == currCost.maxValue && newCost.cost < currCost.cost) {
                    currCost = newCost;
                    oldWireRel = newWireRel;
                }else {
                    updateCosts(newWireRel, oldWireRel, subsetCosts, xDist+1, yDist+1);
                }
            }
 
            // Check Vertical First Paths
            #pragma omp for
            for (int j = 0; j < yDist; j++) {
                newWireRel.bend[0].y = rel_start_y + sign_y*(j + 1);
                newWireRel.bend[0].x = rel_start_x;
                if (rel_start_y + sign_y*(j + 1) == rel_end_y) {
                    // One Bend Case
                    newWireRel.numBends = 1;
                }
                else {
                    // Two Bend Case
                    newWireRel.bend[1].y = rel_start_y + sign_y*(j + 1);
                    newWireRel.bend[1].x = rel_end_x;
                    newWireRel.numBends = 2;
                }
                // Check if newWire is better than oldWire and replace if so
                updateCosts(oldWireRel, newWireRel, subsetCosts, xDist + 1, yDist + 1);
                total_cost_t newCost = calculateCost(newWireRel, subsetCosts, xDist + 1, yDist + 1);
                if(newCost.maxValue < currCost.maxValue){
                    currCost = newCost;
                    oldWireRel = newWireRel;
                }else if (newCost.maxValue == currCost.maxValue && newCost.cost < currCost.cost) {
                    currCost = newCost;
                    oldWireRel = newWireRel;
                }else {
                    updateCosts(newWireRel, oldWireRel, subsetCosts, xDist + 1, yDist + 1);
                }
            }

            // Return wire to real values
            wire_t tempOldWire;
            tempOldWire = oldWire;
            tempOldWire.numBends = oldWireRel.numBends;
            if (tempOldWire.numBends == 1) {
                tempOldWire.bend[0].x = tempOldWire.start_x + sign_x*(oldWireRel.bend[0].x);
                tempOldWire.bend[0].y = tempOldWire.start_y + sign_y*(oldWireRel.bend[0].y);
            }
            else if (oldWire.numBends == 2) {
                tempOldWire.bend[0].x = tempOldWire.start_x + sign_x*(oldWireRel.bend[0].x);
                tempOldWire.bend[0].y = tempOldWire.start_y + sign_y*(oldWireRel.bend[0].y);
                tempOldWire.bend[1].x = tempOldWire.start_x + sign_x*(oldWireRel.bend[1].x);
                tempOldWire.bend[1].y = tempOldWire.start_y + sign_y*(oldWireRel.bend[1].y);
            }

            threadWireArray[threadNum] = tempOldWire;
            #pragma omp barrier
        }
        // Exit Parallel

        // Sequential determine best wire canidate out of potential thread candidates
        for (int k = 0; k < threadCount; k++) {
            // Check if newWire is better than oldWire and replace if so
            updateCosts(oldWire, threadWireArray[k], costs, dim_x, dim_y);
            total_cost_t newCost = calculateCost(threadWireArray[k], costs, dim_x, dim_y);
            if(newCost.maxValue < currCost.maxValue){
                currCost = newCost;
                oldWire = threadWireArray[k];
            }else if (newCost.maxValue == currCost.maxValue && newCost.cost < currCost.cost) {
                currCost = newCost;
                oldWire = threadWireArray[k];
            }else {
                updateCosts(threadWireArray[k], oldWire, costs, dim_x, dim_y);
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
        int randomProb = rand() % 100;
        // Replace best wire with random wire
        if (randomProb < random_prob) {
            wire_t randomWire = (h_or_v) ? hWire : vWire;
            updateCosts(oldWire,randomWire,costs, dim_x, dim_y);
            wires[i] = randomWire;
        }else{
            wires[i] = oldWire;
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
    omp_set_num_threads(num_of_threads);
    for (int i = 0; i < SA_iters; i++) {
        update(wires, costs, dim_x, dim_y, num_of_wires, (int)(SA_prob*100));
    }
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
            fprintf(costFile, "%d ", costs[i*dim_y+j]);
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
                int sign = start_y < end_y ? 1 : -1;
                for(int j = 0; j < abs(end_y-start_y)+1; j++){
                    if(i>0 && j==0) continue;
                    fprintf(outFile, "%d %d ", start_x, start_y+j*(sign));
                }
                start_y=end_y;
            }else{
                int sign = start_x < end_x ? 1 : -1;
                for(int j = 0; j < abs(end_x-start_x)+1; j++){
                    if(i>0 && j==0) continue;
                    fprintf(outFile, "%d %d ", start_x+j*(sign),start_y);
                }
                start_x=end_x;

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

