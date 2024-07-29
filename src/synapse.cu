#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "vector3d.cu"

class Synapse {
public:
    __host__ __device__ Synapse(curandState* rand_state) : 
        weight(0.5f),
        max_weight(1.0f),
        min_weight(0.1f),
        last_spike_time(-INFINITY),
        is_excitatory(curand_uniform(rand_state) < 0.8f),
        resources(1.0f),
        utilization(0.3f),
        tau_rec(800.0f),
        tau_facil(0.0f),
        vesicle_count(1000),
        vesicle_release_probability(0.3f),
        vesicle_recycling_rate(10.0f),
        calcium_concentration(0.0f),
        baseline_calcium(0.1f),
        calcium_decay_rate(0.1f),
        ip3_concentration(0.0f),
        ip3_production_rate(0.1f),
        ip3_decay_rate(0.2f)
    {
        for (int i = 0; i < NUM_RECEPTOR_TYPES; ++i) {
            receptor_counts[i] = 100 + curand_uniform(rand_state) * 100;
            receptor_conductances[i] = 0.0f;
        }
        
        dopamine_sensitivity = 0.5f + 0.5f * curand_uniform(rand_state);
        serotonin_sensitivity = 0.5f + 0.5f * curand_uniform(rand_state);
        
        psd_size = 100.0f + curand_uniform(rand_state) * 100.0f;
        camkii_activation = 0.0f;
        pka_activation = 0.0f;
        
        rab3a_activation = 1.0f;
        rim1_activation = 1.0f;
        
        astrocyte_calcium = baseline_calcium;
        astrocyte_ip3 = 0.0f;
        gliotransmitter_release_probability = 0.1f;
    }

    __device__ void update(float dt, float pre_neuron_time, bool pre_neuron_is_spiking, 
                           float pre_neuron_dopamine_level, float pre_neuron_serotonin_level, 
                           float post_neuron_voltage, curandState* rand_state) {
        updateCalciumDynamics(dt, pre_neuron_is_spiking);
        updateReceptorTrafficking(dt);
        updateVesiclePoolDynamics(dt, pre_neuron_is_spiking);
        updateAstrocyteDynamics(dt, pre_neuron_is_spiking);
        updatePostsynapticDensity(dt);
        updatePresynapticProteins(dt, pre_neuron_is_spiking);
        
        if (pre_neuron_is_spiking) {
            float t_elapsed = pre_neuron_time - last_spike_time;
            resources = 1.0f - (1.0f - resources) * __expf(-t_elapsed / tau_rec);
            utilization += utilization * (1.0f - utilization) * __expf(-t_elapsed / tau_facil);
            
            float release_probability = resources * utilization * rab3a_activation * rim1_activation;
            float neurotransmitter_release = release_probability * vesicle_count;
            resources -= release_probability;
            
            last_spike_time = pre_neuron_time;
        }

        updateSynapticPlasticity(dt, pre_neuron_time, post_neuron_voltage);
        updateNeuromodulation(dt, pre_neuron_dopamine_level, pre_neuron_serotonin_level);
    }

    __device__ void updateCalciumDynamics(float dt, bool pre_neuron_is_spiking) {
        if (pre_neuron_is_spiking) {
            calcium_concentration += 0.1f;
        }
        calcium_concentration += (baseline_calcium - calcium_concentration) * calcium_decay_rate * dt;
        
        float ip3_effect = ip3_concentration / (ip3_concentration + 1.0f);
        calcium_concentration += ip3_effect * 0.05f * dt;
    }

    __device__ void updateReceptorTrafficking(float dt) {
        for (int i = 0; i < NUM_RECEPTOR_TYPES; ++i) {
            float insertion_rate = 1.0f + 0.5f * camkii_activation;
            float removal_rate = 1.0f - 0.5f * pka_activation;
            
            receptor_counts[i] += (insertion_rate - removal_rate) * dt;
            receptor_counts[i] = max(0.0f, min(receptor_counts[i], 1000.0f));
        }
    }

    __device__ void updateVesiclePoolDynamics(float dt, bool pre_neuron_is_spiking) {
        if (pre_neuron_is_spiking) {
            int released_vesicles = floorf(vesicle_count * vesicle_release_probability);
            vesicle_count -= released_vesicles;
        }
        
        float recycling_rate = vesicle_recycling_rate * (1.0f + 0.5f * rab3a_activation);
        vesicle_count += recycling_rate * dt;
        vesicle_count = min(vesicle_count, 1000.0f);
    }

    __device__ void updateAstrocyteDynamics(float dt, bool pre_neuron_is_spiking) {
        if (pre_neuron_is_spiking) {
            astrocyte_ip3 += ip3_production_rate;
        }
        astrocyte_ip3 -= ip3_decay_rate * astrocyte_ip3 * dt;
        
        float ip3_effect = astrocyte_ip3 / (astrocyte_ip3 + 1.0f);
        astrocyte_calcium += (ip3_effect * 0.1f - (astrocyte_calcium - baseline_calcium) * 0.1f) * dt;
        
        if (curand_uniform(rand_state) < gliotransmitter_release_probability * astrocyte_calcium) {
            calcium_concentration += 0.05f;
        }
    }

    __device__ void updatePostsynapticDensity(float dt) {
        float camkii_activation_rate = 0.1f * calcium_concentration * calcium_concentration;
        camkii_activation += (camkii_activation_rate - 0.1f * camkii_activation) * dt;
        camkii_activation = max(0.0f, min(camkii_activation, 1.0f));
        
        float pka_activation_rate = 0.05f * calcium_concentration;
        pka_activation += (pka_activation_rate - 0.1f * pka_activation) * dt;
        pka_activation = max(0.0f, min(pka_activation, 1.0f));
        
        psd_size += (camkii_activation - pka_activation) * dt;
        psd_size = max(50.0f, min(psd_size, 300.0f));
    }

    __device__ void updatePresynapticProteins(float dt, bool pre_neuron_is_spiking) {
        if (pre_neuron_is_spiking) {
            rab3a_activation -= 0.1f;
            rim1_activation -= 0.1f;
        }
        
        rab3a_activation += (1.0f - rab3a_activation) * 0.1f * dt;
        rim1_activation += (1.0f - rim1_activation) * 0.1f * dt;
        
        rab3a_activation = max(0.0f, min(rab3a_activation, 1.0f));
        rim1_activation = max(0.0f, min(rim1_activation, 1.0f));
    }

    __device__ void updateSynapticPlasticity(float dt, float pre_neuron_time, float post_neuron_voltage) {
        float time_diff = pre_neuron_time - last_spike_time;
        
        if (fabsf(time_diff) < 20.0f) {
            float learning_rate = 0.001f * camkii_activation;
            float delta_w = learning_rate * __expf(-fabsf(time_diff) / 10.0f) * copysignf(1.0f, time_diff);
            weight += delta_w;
        }
        
        float target_weight = 0.5f;
        weight += (target_weight - weight) * 0.001f * dt;
        
        float bcm_threshold = 0.5f + 0.5f * pka_activation;
        float bcm_change = (post_neuron_voltage - RESTING_POTENTIAL) * (post_neuron_voltage - bcm_threshold);
        weight += 0.0001f * bcm_change * dt;
        
        weight = max(min_weight, min(max_weight, weight));
    }

    __device__ void updateNeuromodulation(float dt, float dopamine_level, float serotonin_level) {
        float dopamine_effect = dopamine_level * dopamine_sensitivity;
        float serotonin_effect = serotonin_level * serotonin_sensitivity;
        
        weight += (dopamine_effect - serotonin_effect) * 0.01f * dt;
        weight = max(min_weight, min(max_weight, weight));
        
        vesicle_release_probability += dopamine_effect * 0.01f * dt;
        vesicle_release_probability = max(0.1f, min(vesicle_release_probability, 0.9f));
    }

    __device__ float current(float post_voltage) {
        float total_current = 0.0f;
        for (int i = 0; i < NUM_RECEPTOR_TYPES; ++i) {
            float conductance = receptor_counts[i] * receptor_conductances[i];
            float reversal_potential = receptor_reversal_potentials[i];
            total_current += conductance * (post_voltage - reversal_potential);
        }
        return total_current * weight;
    }

    __device__ float getWeight() const { return weight; }
    __device__ void setWeight(float w) { weight = max(min_weight, min(max_weight, w)); }
    __device__ bool isExcitatory() const { return is_excitatory; }

private:
    static const int NUM_RECEPTOR_TYPES = 4;
    static constexpr float RESTING_POTENTIAL = -70.0f; 

    float weight;
    float max_weight;
    float min_weight;
    float last_spike_time;
    bool is_excitatory;
    float resources;
    float utilization;
    float tau_rec;
    float tau_facil;
    float vesicle_count;
    float vesicle_release_probability;
    float vesicle_recycling_rate;
    float receptor_counts[NUM_RECEPTOR_TYPES];
    float receptor_conductances[NUM_RECEPTOR_TYPES];
    float receptor_reversal_potentials[NUM_RECEPTOR_TYPES];
    float calcium_concentration;
    float baseline_calcium;
    float calcium_decay_rate;
    float ip3_concentration;
    float ip3_production_rate;
    float ip3_decay_rate;
    float dopamine_sensitivity;
    float serotonin_sensitivity;
    float psd_size;
    float camkii_activation;
    float pka_activation;
    float rab3a_activation;
    float rim1_activation;
    float astrocyte_calcium;
    float astrocyte_ip3;
    float gliotransmitter_release_probability;
};

__global__ void update_synapses_kernel(Synapse* synapses, int num_synapses, float dt, 
                                       float* pre_neuron_times, bool* pre_neuron_is_spiking, 
                                       float* pre_neuron_dopamine_levels, float* pre_neuron_serotonin_levels, 
                                       float* post_neuron_voltages, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_synapses) {
        synapses[idx].update(dt, pre_neuron_times[idx], pre_neuron_is_spiking[idx],
                             pre_neuron_dopamine_levels[idx], pre_neuron_serotonin_levels[idx],
                             post_neuron_voltages[idx], &rand_states[idx]);
    }
}

void launch_synapse_update(Synapse* d_synapses, int num_synapses, float dt, 
                           float* d_pre_neuron_times, bool* d_pre_neuron_is_spiking, 
                           float* d_pre_neuron_dopamine_levels, float* d_pre_neuron_serotonin_levels, 
                           float* d_post_neuron_voltages, curandState* d_rand_states) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_synapses + threads_per_block - 1) / threads_per_block;
    update_synapses_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_synapses, num_synapses, dt, d_pre_neuron_times, d_pre_neuron_is_spiking,
        d_pre_neuron_dopamine_levels, d_pre_neuron_serotonin_levels,
        d_post_neuron_voltages, d_rand_states);
    cudaDeviceSynchronize();
}