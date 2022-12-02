#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <curand_kernel.h>
#include <chrono>
#include <cub/device/device_radix_sort.cuh>
#include <locale>
#include <iomanip>
#include <curand.h>

// The number of variables to sort
#define N 1'000'000'000

// The number of variables to sort, but the size of it in bytes
#define N_BYTES (N * sizeof(int))

// Random seed for the random number generator
// If set to 0, the seed will be randomly generated
// every time
#define SEED 0

// Max number of threads in the block I want to use (in x direction)
#define BLOCK_SIZE 1024

// Number of indices to assign a random variable to in one thread when setting up
#define BATCH_SIZE 20000

// Utility timing macros.
// This one starts a timer called tsn, with 'n' being whatever thing you pass in
// e.g. START_TIMER(3) => ts3 or START_TIMER(_three) => ts_three
#define START_TIMER(n) \
    auto ts##n = std::chrono::steady_clock::now();

// Stops the timer and prints the elapsed time in ms,
// prefixed with the message passed in
#define STOP_TIMER(n) \
    auto te##n = std::chrono::steady_clock::now(); \
    auto timer_##n = std::chrono::duration_cast<std::chrono::milliseconds>(te##n - ts##n).count();

void sorting_test();

// Sets each index in the array 'xs' to a random number between INT_MIN to INT_MAX
// prefixed with __global__ to indicate it runs on the *device* (the GPU)
__global__
void setupNums(uint64_t seed, int* xs);

// Just a utility function for testing purposes
// Code should be self-explanatory
void printGpuMemoryUsageStatistics();

// If you don't understand this, just quit CS now.
// I'm serious.
// You have no future if that's the case. Don't waste your time.
__global__
void isSorted(const int* arr, bool *is_sorted);

// Struct representation of data for the below function
typedef struct {
    uint64_t sum;
    double   mean;
    double   median;
    int      mode;
    int      mode_count;
    uint64_t range;
    double   stdev;
} IntSummaryStatistics;

// Utility method to find the above statistics in the struct for the
// sorted array. Does not work with an unsorted array. I just made it
// for fun because why not.
IntSummaryStatistics intSummaryStatistics(const int *arr, size_t size);

// Generates a 64-bit random number but only fills in the first
// 32 bits because I'm lazy, and it's enough for now
uint64_t genRandomSeed();

// I will start by saying that I hate C++'s method of printing to stdout.
// That being said, it is fairly extensible, even if it is a travesty.
// This does some C++ magic to make numbers printed to std.out to be
// grouped by commas by thousands
class comma_numpunct : public std::numpunct<char>
{
protected:
    std::string do_grouping() const override;
    char do_thousands_sep() const override;
};