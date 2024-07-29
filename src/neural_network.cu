#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include "vector3d.cu"
#include "neuron.cu"
#include "glia.cu"
#include "synapse.cu"

class NeuralNetwork {
public:
    thrust::device_vector<Neuron> neurons;
    thrust::device_vector<Glia> glia;

    __host__ __device__ NeuralNetwork(int num_neurons, int num_glia, curandState* rand_state)
        : neurons(num_neurons), glia(num_glia) {
        initialize_neurons(rand_state);
        initialize_glia(rand_state);
        initialize_neuron_growth(rand_state);
    }

    __device__ void update(float dt, curandState* rand_state) {
        update_neurons(dt, rand_state);
        update_glia(dt, rand_state);
        update_synapses(dt, rand_state);
        prune_connections();
    }

    __device__ void stimulate_neuron(int index) {
        if (index >= 0 && index < neurons.size()) {
            neurons[index].stimulate(1.0f);
            propagate_signal(&neurons[index]);
        }
    }

    __device__ void reward(float amount = 1.0f) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].dopamine_level += amount * neurons[i].dopamine_receptors;
        }
    }

    __device__ void punish(float amount = 1.0f) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].serotonin_level += amount * neurons[i].serotonin_receptors;
        }
    }

    struct ExtracellularSpace {
        Vector3D position;
        float volume;
        float ion_concentrations[4];
        float neurotransmitter_concentrations[5];
        float pH;
        float temperature;
    };
    thrust::device_vector<ExtracellularSpace> extracellular_spaces;
    
    struct BloodVessel {
        Vector3D start_position;
        Vector3D end_position;
        float diameter;
        float blood_flow_rate;
        float oxygen_concentration;
        float glucose_concentration;
    };
    thrust::device_vector<BloodVessel> blood_vessels;
    
    __device__ void update(float dt, curandState* rand_state);
    __device__ void update_extracellular_space(float dt);
    __device__ void simulate_blood_flow(float dt);
    __device__ void handle_neuron_glia_interactions(float dt);

private:
    thrust::device_vector<Neuron> neurons;
    thrust::device_vector<Glia> glia;
    thrust::device_vector<Synapse> synapses;

    __device__ void initialize_neurons(curandState* rand_state) {
        for (int i = 0; i < neurons.size(); ++i) {
            float x = curand_uniform(rand_state) * 1150.0f + 25.0f;
            float y = curand_uniform(rand_state) * 750.0f + 25.0f;
            float z = curand_uniform(rand_state) * 750.0f + 25.0f;
            neurons[i].setPosition(Vector3D(x, y, z));
        }
    }

    __device__ void initialize_glia(curandState* rand_state) {
        for (int i = 0; i < glia.size(); ++i) {
            float x = curand_uniform(rand_state) * 1150.0f + 25.0f;
            float y = curand_uniform(rand_state) * 750.0f + 25.0f;
            float z = curand_uniform(rand_state) * 750.0f + 25.0f;
            glia[i] = Glia(Vector3D(x, y, z), rand_state);
        }
    }

    __device__ void initialize_neuron_growth(curandState* rand_state) {
        for (int i = 0; i < neurons.size(); ++i) {
            Vector3D axon_direction(
                curand_normal(rand_state),
                curand_normal(rand_state),
                curand_normal(rand_state)
            );
            neurons[i].grow_axon(axon_direction.normalize());

            int num_dendrites = curand_uniform(rand_state) * 3.0f + 2.0f;
            for (int j = 0; j < num_dendrites; ++j) {
                Vector3D dendrite_direction(
                    curand_normal(rand_state),
                    curand_normal(rand_state),
                    curand_normal(rand_state)
                );
                neurons[i].grow_dendrite(dendrite_direction.normalize());
            }
        }
    }

    __device__ void update_neurons(float dt, curandState* rand_state) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].update(dt, rand_state);
        }
    }

    __device__ void update_glia(float dt, curandState* rand_state) {
        for (int i = 0; i < glia.size(); ++i) {
            thrust::device_vector<float> neuron_data(MAX_NEARBY_NEURONS * 10);
            int num_nearby_neurons = get_nearby_neurons(glia[i].getPosition(), 50.0f, neuron_data.data());
            glia[i].update(dt, neuron_data.data(), num_nearby_neurons, rand_state);
        }
    }

    __device__ void update_synapses(float dt, curandState* rand_state) {
        for (int i = 0; i < synapses.size(); ++i) {
            Neuron* pre_neuron = synapses[i].get_pre_neuron();
            Neuron* post_neuron = synapses[i].get_post_neuron();
            synapses[i].update(dt, pre_neuron->time, pre_neuron->is_spiking,
                               pre_neuron->dopamine_level, pre_neuron->serotonin_level,
                               post_neuron->voltage, rand_state);
        }
    }

    __device__ void propagate_signal(Neuron* neuron, int depth = 0) {
        if (depth > 5) return;

        for (int i = 0; i < synapses.size(); ++i) {
            if (synapses[i].get_pre_neuron() == neuron) {
                float signal_strength = synapses[i].getWeight() * neuron->getActivation();
                Neuron* post_neuron = synapses[i].get_post_neuron();
                
                post_neuron->receive_signal(signal_strength, neuron->getType());
                propagate_signal(post_neuron, depth + 1);
            }
        }
    }

    __device__ void prune_connections() {
        thrust::device_vector<Synapse> new_synapses;
        for (int i = 0; i < synapses.size(); ++i) {
            if (synapses[i].getWeight() > 0.1f) {
                new_synapses.push_back(synapses[i]);
            }
        }
        synapses = new_synapses;
    }

    __device__ int get_nearby_neurons(const Vector3D& position, float radius, float* neuron_data) {
        int count = 0;
        for (int i = 0; i < neurons.size() && count < MAX_NEARBY_NEURONS; ++i) {
            if ((neurons[i].getPosition().subtract(position)).magnitude() < radius) {
                int offset = count * 10;
                neuron_data[offset + 0] = neurons[i].glutamate;
                neuron_data[offset + 1] = neurons[i].potassium_concentration;
                neuron_data[offset + 2] = neurons[i].lactate_concentration;
                neuron_data[offset + 3] = neurons[i].axons[0].z;
                neuron_data[offset + 4] = neurons[i].inflammation_level;
                neuron_data[offset + 5] = neurons[i].cellular_debris;
                neuron_data[offset + 6] = neurons[i].csf_volume;
                neuron_data[offset + 7] = neurons[i].gliotransmitter_concentration;
                neuron_data[offset + 8] = neurons[i].getVoltage();
                neuron_data[offset + 9] = neurons[i].getActivation();
                count++;
            }
        }
        return count;
    }
};

__global__ void initialize_neural_network_kernel(NeuralNetwork* network, int num_neurons, int num_glia, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        new (network) NeuralNetwork(num_neurons, num_glia, &rand_states[idx]);
    }
}

__global__ void update_neural_network_kernel(NeuralNetwork* network, float dt, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        network->update(dt, &rand_states[idx]);
    }
}

__global__ void stimulate_neuron_kernel(NeuralNetwork* network, int neuron_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        network->stimulate_neuron(neuron_index);
    }
}

__global__ void reward_network_kernel(NeuralNetwork* network, float amount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        network->reward(amount);
    }
}

__global__ void punish_network_kernel(NeuralNetwork* network, float amount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        network->punish(amount);
    }
}

void launch_initialize_neural_network(NeuralNetwork* d_network, int num_neurons, int num_glia, curandState* d_rand_states) {
    initialize_neural_network_kernel<<<1, 1>>>(d_network, num_neurons, num_glia, d_rand_states);
    cudaDeviceSynchronize();
}

void launch_update_neural_network(NeuralNetwork* d_network, float dt, curandState* d_rand_states) {
    update_neural_network_kernel<<<1, 1>>>(d_network, dt, d_rand_states);
    cudaDeviceSynchronize();
}

void launch_stimulate_neuron(NeuralNetwork* d_network, int neuron_index) {
    stimulate_neuron_kernel<<<1, 1>>>(d_network, neuron_index);
    cudaDeviceSynchronize();
}

void launch_reward_network(NeuralNetwork* d_network, float amount) {
    reward_network_kernel<<<1, 1>>>(d_network, amount);
    cudaDeviceSynchronize();
}

void launch_punish_network(NeuralNetwork* d_network, float amount) {
    punish_network_kernel<<<1, 1>>>(d_network, amount);
    cudaDeviceSynchronize();
}