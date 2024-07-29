#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include "vector3d.cu"

class Axon {
public:
    __host__ __device__ Axon() : length(0), growth_rate(5), max_segment_length(20), max_length(1000) {}

    __host__ __device__ Axon(const Vector3D& neuron_position, const Vector3D& initial_direction)
        : length(0), growth_rate(5), max_segment_length(20), max_length(1000) {
        growth_cone = neuron_position;
        segments.push_back(growth_cone);
        direction = initial_direction.normalize();
    }

    __device__ void grow(float dt, const float* growth_factors, int num_factors, float stimulation_level, curandState* rand_state) {
        if (length >= max_length) return;

        float growth_amount = 0;
        for (int i = 0; i < num_factors; i++) {
            growth_amount += growth_factors[i] * growth_rate * dt;
        }

        length += growth_amount;

        Vector3D random_factor(
            curand_normal(rand_state),
            curand_normal(rand_state),
            curand_normal(rand_state)
        );
        random_factor = random_factor.multiply(0.1f * (2.0f - stimulation_level));

        if (curand_uniform(rand_state) < 0.05f) {
            direction = direction.rotate(
                curand_uniform(rand_state) * 2 * M_PI,
                curand_uniform(rand_state) * M_PI
            );
        }

        Vector3D growth_direction = direction.add(random_factor).normalize();
        Vector3D growth_vector = growth_direction.multiply(growth_amount);
        growth_cone = growth_cone.add(growth_vector);

        if (growth_cone.subtract(segments.back()).magnitude() > max_segment_length) {
            segments.push_back(growth_cone);
        }

        if (curand_uniform(rand_state) < 0.01f && branches.size() < 5) {
            Vector3D branch_direction = direction.rotate(
                curand_uniform(rand_state) * 2 * M_PI,
                curand_uniform(rand_state) * M_PI / 2
            );
            branches.push_back(Axon(growth_cone, branch_direction));
        }

        for (Axon& branch : branches) {
            branch.grow(dt, growth_factors, num_factors, stimulation_level, rand_state);
        }

        updateMyelination(dt, stimulation_level);
    }

    __device__ void updateMyelination(float dt, float stimulation_level) {
        if (length > 100 && myelination < 1.0f) {
            myelination += (0.01f * stimulation_level * dt);
            myelination = min(myelination, 1.0f);
        }
    }

    __device__ bool check_synapse_formation(const Vector3D& dendrite_tip, const float* growth_factors, int num_factors, curandState* rand_state) {
        float distance = growth_cone.subtract(dendrite_tip).magnitude();
        if (distance < 5.0f) {
            float formation_probability = 0.1f;
            for (int i = 0; i < num_factors; i++) {
                formation_probability *= (1 + growth_factors[i]);
            }
            return curand_uniform(rand_state) < formation_probability;
        }
        return false;
    }

    __device__ float computeSignalPropagationTime() {
        float base_speed = 0.1f; 
        float myelinated_speed = 100.0f; 
        float speed = base_speed + (myelinated_speed - base_speed) * myelination;
        return length / (speed * 1000); 
    }

    __host__ __device__ const Vector3D& get_growth_cone() const { return growth_cone; }
    __host__ __device__ float get_length() const { return length; }
    __host__ __device__ const thrust::device_vector<Vector3D>& get_segments() const { return segments; }
    __host__ __device__ float get_myelination() const { return myelination; }

private:
    Vector3D direction;
    Vector3D growth_cone;
    float length;
    float growth_rate;
    float max_segment_length;
    float max_length;
    float myelination;
    thrust::device_vector<Vector3D> segments;
    thrust::device_vector<Axon> branches;
};

__global__ void grow_axons_kernel(Axon* axons, int num_axons, float dt, float* growth_factors, int num_factors, float* stimulation_levels, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_axons) {
        axons[idx].grow(dt, &growth_factors[idx * num_factors], num_factors, stimulation_levels[idx], &rand_states[idx]);
    }
}

void launch_axon_growth(Axon* d_axons, int num_axons, float dt, float* d_growth_factors, int num_factors, float* d_stimulation_levels, curandState* d_rand_states) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_axons + threads_per_block - 1) / threads_per_block;
    grow_axons_kernel<<<blocks_per_grid, threads_per_block>>>(d_axons, num_axons, dt, d_growth_factors, num_factors, d_stimulation_levels, d_rand_states);
    cudaDeviceSynchronize();
}

__global__ void check_synapse_formation_kernel(Axon* axons, Vector3D* dendrite_tips, int num_axons, int num_dendrites, float* growth_factors, int num_factors, curandState* rand_states, bool* synapse_formed) {
    int axon_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dendrite_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (axon_idx < num_axons && dendrite_idx < num_dendrites) {
        bool formed = axons[axon_idx].check_synapse_formation(dendrite_tips[dendrite_idx], &growth_factors[axon_idx * num_factors], num_factors, &rand_states[axon_idx]);
        synapse_formed[axon_idx * num_dendrites + dendrite_idx] = formed;
    }
}

void launch_synapse_formation_check(Axon* d_axons, Vector3D* d_dendrite_tips, int num_axons, int num_dendrites, float* d_growth_factors, int num_factors, curandState* d_rand_states, bool* d_synapse_formed) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_axons + threads_per_block.x - 1) / threads_per_block.x,
        (num_dendrites + threads_per_block.y - 1) / threads_per_block.y
    );
    check_synapse_formation_kernel<<<blocks_per_grid, threads_per_block>>>(d_axons, d_dendrite_tips, num_axons, num_dendrites, d_growth_factors, num_factors, d_rand_states, d_synapse_formed);
    cudaDeviceSynchronize();
}