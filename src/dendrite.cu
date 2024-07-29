#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include "vector3d.cu"

#define RESTING_POTENTIAL -70.0f  
#define THRESHOLD_POTENTIAL -55.0f  
#define CALCIUM_REST 0.0001f  
#define CALCIUM_THRESHOLD 0.0002f  
#define MAX_SYNAPSES 1000
#define MAX_BRANCHES 5

class Spine {
public:
    __device__ Spine() : position(), size(1.0f), ampa_receptors(50), nmda_receptors(20) {}

    __device__ void updateReceptors(float activity_level, curandState* rand_state) {
        float change = activity_level * 0.1f * curand_normal(rand_state);
        ampa_receptors = max(0.0f, ampa_receptors + change);
        nmda_receptors = max(0.0f, nmda_receptors + change * 0.5f);
        
        size += (activity_level - 0.5f) * 0.01f;
        size = max(0.5f, min(size, 2.0f));
    }

    Vector3D position;
    float size;
    float ampa_receptors;
    float nmda_receptors;
};

class Dendrite {
public:
    __host__ __device__ Dendrite() : length(0), growth_rate(5), max_segment_length(15), max_length(150),
                                     membrane_potential(RESTING_POTENTIAL), calcium_concentration(CALCIUM_REST) {}

    __host__ __device__ Dendrite(const Vector3D& neuron_position, const Vector3D& initial_direction)
        : length(0), growth_rate(5), max_segment_length(15), max_length(150),
          membrane_potential(RESTING_POTENTIAL), calcium_concentration(CALCIUM_REST) {
        tip = neuron_position;
        segments.push_back(tip);
        direction = initial_direction.normalize();
    }

    __device__ void grow(float dt, const float* growth_factors, int num_factors, float stimulation_level, curandState* rand_state) {
        if (length >= max_length) return;

        float growth_amount = 0;
        for (int i = 0; i < num_factors; i++) {
            growth_amount += growth_factors[i] * growth_rate * dt;
        }
        growth_amount *= (1.0f + (calcium_concentration - CALCIUM_REST) / CALCIUM_REST);

        length += growth_amount;

        Vector3D random_factor(
            curand_normal(rand_state),
            curand_normal(rand_state),
            curand_normal(rand_state)
        );
        random_factor = random_factor.multiply(0.05f * (2.0f - stimulation_level));

        Vector3D calcium_gradient(
            curand_normal(rand_state),
            curand_normal(rand_state),
            curand_normal(rand_state)
        );
        calcium_gradient = calcium_gradient.normalize().multiply((calcium_concentration - CALCIUM_REST) / CALCIUM_REST);

        if (curand_uniform(rand_state) < 0.03f) {
            direction = direction.rotate(
                curand_uniform(rand_state) * 2 * M_PI,
                Vector3D(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)).normalize()
            );
        }

        Vector3D growth_direction = direction.add(random_factor).add(calcium_gradient).normalize();
        Vector3D growth_vector = growth_direction.multiply(growth_amount);
        tip = tip.add(growth_vector);

        if (tip.subtract(segments.back()).magnitude() > max_segment_length) {
            segments.push_back(tip);
        }

        if (curand_uniform(rand_state) < 0.005f * (calcium_concentration / CALCIUM_REST) && branches.size() < MAX_BRANCHES) {
            Vector3D branch_direction = direction.rotate(
                curand_uniform(rand_state) * 2 * M_PI,
                Vector3D(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)).normalize()
            );
            branches.push_back(Dendrite(tip, branch_direction));
        }

        for (Dendrite& branch : branches) {
            branch.grow(dt, growth_factors, num_factors, stimulation_level, rand_state);
        }

        updateSpineDensity(dt, stimulation_level);
        for (Spine& spine : spines) {
            spine.updateReceptors(stimulation_level, rand_state);
        }

        updateElectrophysiology(dt, stimulation_level);

        updateMetabolism(dt);
    }

    __device__ void updateSpineDensity(float dt, float stimulation_level) {
        if (length > 50 && spines.size() < MAX_SYNAPSES) {
            float spine_growth_probability = 0.01f * stimulation_level * (calcium_concentration / CALCIUM_REST);
            if (curand_uniform(rand_state) < spine_growth_probability) {
                Spine new_spine;
                new_spine.position = tip.add(Vector3D(curand_normal(rand_state), curand_normal(rand_state), curand_normal(rand_state)).normalize().multiply(2.0f));
                spines.push_back(new_spine);
            }
        }
    }

    __device__ void updateElectrophysiology(float dt, float stimulation_level) {
        float input_current = stimulation_level * 10.0f; 
        membrane_potential += dt * (-0.1f * (membrane_potential - RESTING_POTENTIAL) + input_current);

        if (membrane_potential > THRESHOLD_POTENTIAL) {
            calcium_concentration += 0.0001f * dt;
        } else {
            calcium_concentration += (CALCIUM_REST - calcium_concentration) * 0.1f * dt;
        }
        calcium_concentration = max(CALCIUM_REST, min(calcium_concentration, 0.001f));
    }

    __device__ void updateMetabolism(float dt) {
        energy -= (0.1f * fabsf(membrane_potential - RESTING_POTENTIAL) + 0.01f * length) * dt;
        energy = max(0.0f, min(energy, 100.0f));

        growth_rate = 5.0f * (energy / 100.0f);
    }

    __device__ bool check_synapse_formation(const Vector3D& axon_tip, const float* growth_factors, int num_factors, curandState* rand_state) {
        float distance = tip.subtract(axon_tip).magnitude();
        if (distance < 5.0f && spines.size() < MAX_SYNAPSES) {
            float formation_probability = 0.1f * (spines.size() / float(MAX_SYNAPSES)) * (calcium_concentration / CALCIUM_REST);
            for (int i = 0; i < num_factors; i++) {
                formation_probability *= (1 + growth_factors[i]);
            }
            return curand_uniform(rand_state) < formation_probability;
        }
        return false;
    }

    __device__ float computeSignalPropagationTime() {
        float base_speed = 0.1f;
        float speed = base_speed * (1.0f + 0.5f * (energy / 100.0f));
        return length / (speed * 1000);
    }

    __host__ __device__ const Vector3D& get_tip() const { return tip; }
    __host__ __device__ float get_length() const { return length; }
    __host__ __device__ const thrust::device_vector<Vector3D>& get_segments() const { return segments; }
    __host__ __device__ size_t get_spine_count() const { return spines.size(); }
    __host__ __device__ float get_membrane_potential() const { return membrane_potential; }
    __host__ __device__ float get_calcium_concentration() const { return calcium_concentration; }

private:
    Vector3D direction;
    Vector3D tip;
    float length;
    float growth_rate;
    float max_segment_length;
    float max_length;
    thrust::device_vector<Vector3D> segments;
    thrust::device_vector<Dendrite> branches;
    thrust::device_vector<Spine> spines;

    float membrane_potential;
    float calcium_concentration;
    float energy;

    curandState* rand_state;
};

__global__ void grow_dendrites_kernel(Dendrite* dendrites, int num_dendrites, float dt, float* growth_factors, int num_factors, float* stimulation_levels, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_dendrites) {
        dendrites[idx].grow(dt, &growth_factors[idx * num_factors], num_factors, stimulation_levels[idx], &rand_states[idx]);
    }
}

void launch_dendrite_growth(Dendrite* d_dendrites, int num_dendrites, float dt, float* d_growth_factors, int num_factors, float* d_stimulation_levels, curandState* d_rand_states) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_dendrites + threads_per_block - 1) / threads_per_block;
    grow_dendrites_kernel<<<blocks_per_grid, threads_per_block>>>(d_dendrites, num_dendrites, dt, d_growth_factors, num_factors, d_stimulation_levels, d_rand_states);
    cudaDeviceSynchronize();
}

__global__ void check_dendrite_synapse_formation_kernel(Dendrite* dendrites, Vector3D* axon_tips, int num_dendrites, int num_axons, float* growth_factors, int num_factors, curandState* rand_states, bool* synapse_formed) {
    int dendrite_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int axon_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dendrite_idx < num_dendrites && axon_idx < num_axons) {
        bool formed = dendrites[dendrite_idx].check_synapse_formation(axon_tips[axon_idx], &growth_factors[dendrite_idx * num_factors], num_factors, &rand_states[dendrite_idx]);
        synapse_formed[dendrite_idx * num_axons + axon_idx] = formed;
    }
}

void launch_dendrite_synapse_formation_check(Dendrite* d_dendrites, Vector3D* d_axon_tips, int num_dendrites, int num_axons, float* d_growth_factors, int num_factors, curandState* d_rand_states, bool* d_synapse_formed) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_dendrites + threads_per_block.x - 1) / threads_per_block.x,
        (num_axons + threads_per_block.y - 1) / threads_per_block.y
    );
    check_dendrite_synapse_formation_kernel<<<blocks_per_grid, threads_per_block>>>(d_dendrites, d_axon_tips, num_dendrites, num_axons, d_growth_factors, num_factors, d_rand_states, d_synapse_formed);
    cudaDeviceSynchronize();
}