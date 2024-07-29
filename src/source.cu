#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include "vector3d.cu"
#include "neuron.cu"
#include "axon.cu"
#include "dendrite.cu"
#include "synapse.cu"
#include "glia.cu"
#include "neural_network.cu"

const int NUM_NEURONS = 1000;
const int NUM_GLIA = 500;
const float DT = 0.05f;
const int SIMULATION_STEPS = 1000;

__global__ void init_rand_states(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

int main() {
    NeuralNetwork* d_network;
    cudaMalloc(&d_network, sizeof(NeuralNetwork));

    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, (NUM_NEURONS + NUM_GLIA) * sizeof(curandState));
    init_rand_states<<<(NUM_NEURONS + NUM_GLIA + 255) / 256, 256>>>(d_rand_states, time(NULL));

    launch_initialize_neural_network(d_network, NUM_NEURONS, NUM_GLIA, d_rand_states);

    for (int step = 0; step < SIMULATION_STEPS; ++step) {
        launch_update_neural_network(d_network, DT, d_rand_states);

        if (step % 100 == 0) {
            std::cout << "Simulation step: " << step << "/" << SIMULATION_STEPS << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cudaFree(d_network);
    cudaFree(d_rand_states);

    std::cout << "Simulation completed." << std::endl;
    return 0;
}