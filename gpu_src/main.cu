#include "main.cuh"

#define PRINT_STATS

int main()
{
    sorting_test();
}

void sorting_test()
{
    std::srand(time(NULL));

    // This weird stuff makes numbers printed in stdout using cout
    // go from '10000000.123456' to '10,000,000.1234' (somehow)
    // I don't get it myself, but whatever. It works, so it works.
    std::locale comma_locale(std::locale(), new comma_numpunct());
    std::cout.imbue(comma_locale);
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(4);

    printGpuMemoryUsageStatistics();

    START_TIMER(overall)

    // Initialize 'host' array, aka the array that can be seen by the CPU
    // Prefixed with 'h_' to indicate it's on the host
    // Code run on the CPU can't access 'device memory', AKA memory on the GPU
    // and vice versa.
    // It needs to be copied back and forth via the PCI-e bus
    int* h_nums = new int[N];

    // Initialize device arrays
    // Prefixed with 'd_' because you can probably guess why
    // Needs to be allocated with cudaMalloc so it's allocated on the GPU
    // and not the CPU
    // TODO: Maybe use cudaMallocManaged for d_nums & test speed differences
    int* d_nums, *d_nums_alt;
    cudaMalloc(&d_nums, N_BYTES);
    cudaMalloc(&d_nums_alt, N_BYTES);

    START_TIMER(arrayinit)

    // Initialize states for the random number generator
    // Need to have one for each thread that will be used
    uint64_t seed = (SEED) ? SEED : genRandomSeed();

    // Calculates the number of thread blocks that need to be generated depending
    // on how large the thread block can be
    // For example, 10 blocks for 10*1024 threads.
    int num_blocks = 1 + ((N / BATCH_SIZE) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Runs setupNums on the GPU
    // <<< Number of threads blocks, number of threads per block>>>
    // The dimensions can be 3d, but keeping it 1d for this
    // The function is called once in thread that is run
    setupNums<<< num_blocks, BLOCK_SIZE >>>(seed, d_nums);
    cudaDeviceSynchronize();

    STOP_TIMER(arrayinit)
    START_TIMER(sortingwithsetup)

    // Double buffer to save on required device storage space, ~O(N + P) vs ~O(2N + P)
    cub::DoubleBuffer<int> d_nums_buffer(d_nums, d_nums_alt);

    // Initialize auxiliary space needed for the sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Figure out how much aux. space needed and allocate it
    // Just running the sort with d_temp_storage being null will calculate the number of
    // bytes needed and put it in temp_storage_bytes for us
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_nums_buffer, N);

    // We can then manually allocate the auxiliary space on the device ourselves
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Waits for the kernal function to finish
    // Otherwise it runs asynchronously to the CPU code until another GPU
    // function is called
    // I normally wouldn't need this because cub::DeviceRadixSort::SortKeys will wait
    // for cudaMalloc to finish, butttt I need cudaMalloc to finish BEFORE timer3 starts
    cudaDeviceSynchronize();

    // Finally sort array
    START_TIMER(sorting)
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_nums_buffer, N);

    printGpuMemoryUsageStatistics();

    // Stops the above code so timer3 can stop right after the sort is done, and not before or after
    cudaDeviceSynchronize();
    STOP_TIMER(sorting)

    // Copy sorted device array back to host
    // since we can't see the device memory from the host
    // without first copying it over
    cudaMemcpy(h_nums, d_nums_buffer.Current(), N_BYTES, cudaMemcpyDeviceToHost);
    STOP_TIMER(sortingwithsetup)

    // Frees the device memory we don't need anymore
    cudaFree(d_nums_buffer.Alternate());
    cudaFree(d_temp_storage);

    STOP_TIMER(overall)

    puts("\n-----------------------------------------------------------------");
    std::cout << "- " << N << " elements sorted in:" << std::endl;
    printf(
        "- \n"
        "- JUST sorting:                              %5lldms\n"
        "- Sorting + aux. allocations & such:         %5lldms\n"
        "- \n"
        "- Overall time (not including verification): %5lldms\n"
        "- Array initialization:                      %5lldms\n"
        "-----------------------------------------------------------------\n",
        timer_sorting, timer_sortingwithsetup, timer_overall, timer_arrayinit
    );

    // Some checks and verification
    const int NUMS_TO_PRINT_FOR_VERIFICATION = 5;

    for (int i = 0; i < NUMS_TO_PRINT_FOR_VERIFICATION; i++)
    {
        if (i == 0)
        {
            puts("");
        }

        printf("| %d <- [%d]\n", h_nums[i], i);
    }

    for (int i = N - NUMS_TO_PRINT_FOR_VERIFICATION; i < N; i++)
    {
        if (i == 0)
        {
            puts("");
        }

        printf("| %d <- [%d]\n", h_nums[i], i);
    }

    bool *is_sorted = nullptr;
    cudaMallocManaged(&is_sorted, sizeof(bool));

    isSorted<<< num_blocks, BLOCK_SIZE >>>(d_nums_buffer.Current(), is_sorted);
    cudaDeviceSynchronize();

    if (*is_sorted)
    {
        std::cout << "\nIt wprked!\n";
    }
    else
    {
        std::cout << "\nflip.\n";
    }

#ifdef PRINT_STATS
    // TODO: Parallelize
    START_TIMER(stats)
    auto stats = intSummaryStatistics(h_nums, N);
    STOP_TIMER(stats)

    auto printStat = [](std::string msg, auto stat) {
        if (stat == (int64_t)stat)
        {
            std::cout << msg            << std::setw(22) << std::right << (int64_t)stat << std::endl;
        }
        else
        {
            std::cout << msg << "     " << std::setw(22) << std::right << stat          << std::endl;
        }
    };

    puts("\n-----------------------------------------------------------------");
    std::cout << "- Random stats (time taken: " << timer_stats << "ms)" << std::endl;
         puts( "-");
    printStat( "- Sum:        ", stats.sum        );
    printStat( "- Mean:       ", stats.mean       );
    printStat( "- Median:     ", stats.median     );
    printStat( "- Mode:       ", stats.mode       );
    printStat( "- Mode cnt:   ", stats.mode_count );
    printStat( "- Range:      ", stats.range      );
    printStat( "  - max:      ", UINT_MAX         );
    printStat( "- Std. dev:   ", stats.stdev      );
    puts("-----------------------------------------------------------------");
#endif

    cudaFree(d_nums_buffer.Current());
    delete h_nums;
}

// Sets each index in the array 'xs' to a random number between INT_MIN to INT_MAX
// prefixed with __global__ to indicate it runs on the *device* (the GPU)
__global__
void setupNums(uint64_t seed, int* xs)
{
    // Gets the index of the current thread
    // I'm oversimplifying here, but threads are grouped into 3D blocks of threads,
    // but here I chose to just use the x dimension for simplicity

    // Block 1          Block 2          Block 3                              |
    // ---------------  ---------------  ---------------    this is a thread: |
    // | 0 1 2 3 4 5 |  | 0 1 2 3 4 5 |  | 0 1 2 3 4 5 |                      v
    // | | | | | | | |  | | | | | | | |  | | | | | | | |
    // | v v v v v v |  | v v v v v v |  | v v v v v v |    each block has width
    // ---------------  ---------------  ---------------    of 6 threads here

    // So, thread #0 of block #2 would be 0 + (2 * 6), which is index 12
    uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Like many C-like or functional constructs, the data is separate from the things that manipulate it.
    // Because of this, we need to manually initialize it and pass it into the RNG ourselves.
    curandState rand_state;
    curand_init(seed + idx, idx, 0, &rand_state);

    // I'm dividing the setting of each index into batches of 20000 indices per thread
    // to save on memory space and threads
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        // Checking to make sure that the array index is within bounds
        // It might be out of bounds if we have extra threads
        if (i + idx * BATCH_SIZE < N)
        {
            // curand_uniform_double generates a uniform distribution of doubles
            // so like each number is basically equally likely to be picked
           xs[i + idx * BATCH_SIZE] = (int)(INT_MAX * curand_uniform_double(&rand_state));
        }
    }
}

// Just a utility function for testing purposes
// Code should be self-explanatory
void printGpuMemoryUsageStatistics()
{
    static bool have_printed_newline = false;

    if (!have_printed_newline)
    {
        puts("");
        have_printed_newline = true;
    }

    size_t free_db;
    size_t total_db;

    cudaMemGetInfo(&free_db, &total_db);

    size_t used_db = total_db - free_db;

    printf(
//        "GPU memory usage: used = %2f MB, free = %2f MB, total = %f MB\n",
        "GPU memory usage: used = %2f MB\n",
        used_db  / (1024.0 * 1024.0)
//        free_db  / (1024.0 * 1024.0),
//        total_db / (1024.0 * 1024.0)
    );
}

// just read the function name and figure it out yourself
__global__
void isSorted(const int* xs, bool *is_sorted)
{
    int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx == 0)
    {
        *is_sorted = true;
    }

    __syncthreads();

    int64_t real_index;

    for (int i = 1; i < BATCH_SIZE && *is_sorted; i++)
    {
        real_index = i + idx * BATCH_SIZE;

        if (real_index < N && xs[real_index - 1] > xs[real_index])
        {
            *is_sorted = false;
        }
    }
}

// Utility method to find the above statistics in the struct for the
// sorted array. Does not work with an unsorted array. I just made it
// for fun because why not.
IntSummaryStatistics intSummaryStatistics(const int *arr, size_t size)
{
    IntSummaryStatistics stats = {};

    int max_mode_count = 0;
    int curr_mode_count = 0;

    uint64_t sum_squared = 0;

    for (int i = 0; i < size; i++)
    {
        stats.sum += arr[i];

        double delta = arr[i] - stats.mean;

        stats.mean += delta / (i + 1);

        sum_squared += delta * (arr[i] - stats.mean);

        if (i < 1) continue;

        curr_mode_count++;

        if (arr[i] != arr[i-1] || i == size - 1)
        {
            if (curr_mode_count > max_mode_count)
            {
                stats.mode = arr[i-1];              // Covers edge case when most freq is last, and makes sure the last
                stats.mode_count = curr_mode_count + (arr[i] == arr[i-1] && i == size - 1); // item isn't different too
            }
            max_mode_count = std::max(curr_mode_count, max_mode_count);
            curr_mode_count = 0;
        }
    }

    double variance = sum_squared / (double) size;

    stats.stdev = sqrt(variance);

    if (size % 2 == 0)
    {
        stats.median = (arr[(size-1)/2] + arr[size/2]) / 2.0;
    }
    else
    {
        stats.median = arr[size/2];
    }

    // Range is range.
    stats.range = (int64_t) arr[size-1] - arr[0];

    return stats;
}

// Generates a 64-bit random number but only fills in the first
// 32 bits because I'm lazy, and it's enough for now
uint64_t genRandomSeed()
{
    uint8_t  r1 = std::rand() & 255;
    uint16_t r2 = std::rand() & 255;
    uint32_t r3 = std::rand() & 255;
    uint32_t r4 = std::rand() & 255;
    return r4 << 24 | r3 << 16 | r2 << 8 | r1 << 0;
}

// It works
char comma_numpunct::do_thousands_sep() const
{
    return ',';
}

// Somehow. Don't ask me.
std::string comma_numpunct::do_grouping() const
{
    return "\03";
}
