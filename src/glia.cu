#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "vector3d.cu"

#define MAX_PROCESSES 10
#define MAX_NEARBY_NEURONS 100
#define MAX_BRANCHES 50

class Glia {
public:
    enum GliaType { ASTROCYTE, OLIGODENDROCYTE, MICROGLIA, EPENDYMAL };

    struct Process {
        Vector3D tip;
        float length;
        float thickness;
    };

    __host__ __device__ Glia() : position(), type(ASTROCYTE), size(10.0f) {
        initializeCommonProperties();
    }

    __host__ __device__ Glia(const Vector3D& pos, curandState* rand_state) : 
        position(pos),
        type(determineGliaType(rand_state)),
        size(10.0f)
    {
        initializeCommonProperties();
        initializeTypeSpecificProperties(rand_state);
        initializeMorphology(rand_state);
    }

    __device__ void update(float dt, float* neuron_data, int num_nearby_neurons, curandState* rand_state) {
        updateCalciumDynamics(dt);
        updateMetabolism(dt);
        updateMorphology(dt, rand_state);

        switch (type) {
            case ASTROCYTE:
                updateAstrocyte(dt, neuron_data, num_nearby_neurons, rand_state);
                break;
            case OLIGODENDROCYTE:
                updateOligodendrocyte(dt, neuron_data, num_nearby_neurons);
                break;
            case MICROGLIA:
                updateMicroglia(dt, neuron_data, num_nearby_neurons, rand_state);
                break;
            case EPENDYMAL:
                updateEpendymal(dt, neuron_data, num_nearby_neurons);
                break;
        }
    }

    __device__ int getNumProcesses() const { return num_processes; }
    __device__ Vector3D getProcessTip(int index) const {
        if (index >= 0 && index < num_processes) {
            return processes[index].tip;
        }
        return Vector3D();
    }

    __device__ void communicateWithNearbyGlia(Glia* nearby_glia, int num_nearby_glia, float dt) {
        for (int i = 0; i < num_nearby_glia; i++) {
            float distance = (nearby_glia[i].position - position).magnitude();
            if (distance < 50.0f) {
                switch (type) {
                    case ASTROCYTE:
                        if (calcium_concentration > 0.5f) {
                            nearby_glia[i].triggerCalciumWave(dt * 0.5f);
                        }
                        break;
                    case OLIGODENDROCYTE:
                        myelination_rate = max(myelination_rate, nearby_glia[i].myelination_rate * 0.9f);
                        break;
                    case MICROGLIA:
                        phagocytosis_rate = max(phagocytosis_rate, nearby_glia[i].phagocytosis_rate * 0.8f);
                        break;
                    case EPENDYMAL:
                        csf_production_rate = (csf_production_rate + nearby_glia[i].csf_production_rate) * 0.5f;
                        break;
                }
            }
        }
    }

    Vector3D soma_position;
    thrust::device_vector<Vector3D> processes;
    
    __device__ void regulate_extracellular_environment(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    __device__ void support_myelination(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    __device__ void modulate_synaptic_transmission(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    __device__ void perform_immune_functions(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    
    __device__ void release_gliotransmitters(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    __device__ void respond_to_neurotransmitters(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    
    __device__ void provide_metabolic_support(Neuron* nearby_neurons, int num_nearby_neurons, float dt);
    
    __host__ __device__ const Vector3D& getPosition() const { return position; }
    __host__ __device__ float getSize() const { return size; }
    __host__ __device__ float4 getColor() const { return color; }
    __host__ __device__ GliaType getType() const { return type; }

private:
    Vector3D position;
    GliaType type;
    float size;
    float4 color;
    Process processes[MAX_PROCESSES];
    int num_processes;

    float calcium_concentration;
    float ip3_concentration;
    float atp_concentration;
    float glucose_concentration;
    float lactate_concentration;
    float glutamate_concentration;
    float potassium_concentration;

    float glutamate_uptake_rate;
    float potassium_buffering_rate;
    float myelination_rate;
    float phagocytosis_rate;
    float csf_production_rate;

    __host__ __device__ void initializeCommonProperties() {
        calcium_concentration = 0.1f;
        ip3_concentration = 0.0f;
        atp_concentration = 10.0f;
        glucose_concentration = 5.0f;
        lactate_concentration = 1.0f;
        glutamate_concentration = 0.0f;
        potassium_concentration = 5.0f;
        num_processes = 0;
    }

    __device__ GliaType determineGliaType(curandState* rand_state) {
        float r = curand_uniform(rand_state);
        if (r < 0.4f) return ASTROCYTE;
        else if (r < 0.7f) return OLIGODENDROCYTE;
        else if (r < 0.9f) return MICROGLIA;
        else return EPENDYMAL;
    }

    __device__ void initializeTypeSpecificProperties(curandState* rand_state) {
        switch (type) {
            case ASTROCYTE:
                glutamate_uptake_rate = 0.1f + 0.05f * curand_uniform(rand_state);
                potassium_buffering_rate = 0.05f + 0.02f * curand_uniform(rand_state);
                color = make_float4(1.0f, 1.0f, 0.0f, 0.3f);
                break;
            case OLIGODENDROCYTE:
                myelination_rate = 0.01f + 0.005f * curand_uniform(rand_state);
                color = make_float4(0.0f, 1.0f, 1.0f, 0.3f);
                break;
            case MICROGLIA:
                phagocytosis_rate = 0.02f + 0.01f * curand_uniform(rand_state);
                color = make_float4(1.0f, 0.0f, 1.0f, 0.3f);
                break;
            case EPENDYMAL:
                csf_production_rate = 0.005f + 0.002f * curand_uniform(rand_state);
                color = make_float4(0.5f, 0.5f, 1.0f, 0.3f);
                break;
        }
    }

    __device__ void initializeMorphology(curandState* rand_state) {
        int max_processes = (type == ASTROCYTE) ? MAX_PROCESSES : (type == OLIGODENDROCYTE) ? MAX_BRANCHES : 5;
        num_processes = max_processes;
        for (int i = 0; i < num_processes; ++i) {
            processes[i].tip = position.add(Vector3D(
                curand_normal(rand_state),
                curand_normal(rand_state),
                curand_normal(rand_state)
            ).normalize().multiply(10.0f));
            processes[i].length = 10.0f;
            processes[i].thickness = 0.5f + 0.5f * curand_uniform(rand_state);
        }
    }

    __device__ void updateCalciumDynamics(float dt) {
        float ip3_effect = ip3_concentration / (ip3_concentration + 1.0f);
        calcium_concentration += (ip3_effect * 0.5f - 0.1f * calcium_concentration) * dt;
        calcium_concentration = max(0.1f, min(calcium_concentration, 2.0f));
        ip3_concentration *= expf(-0.2f * dt);
    }

    __device__ void updateMetabolism(float dt) {
        glucose_concentration -= 0.1f * dt;
        atp_concentration += 0.2f * glucose_concentration * dt;
        lactate_concentration += 0.05f * glucose_concentration * dt;

        glucose_concentration = max(0.1f, glucose_concentration);
        atp_concentration = max(0.1f, min(atp_concentration, 20.0f));
        lactate_concentration = max(0.1f, min(lactate_concentration, 5.0f));
    }

    __device__ void updateMorphology(float dt, curandState* rand_state) {
        for (int i = 0; i < num_processes; ++i) {
            if (curand_uniform(rand_state) < 0.1f * dt) {
                processes[i].length += 0.1f * curand_normal(rand_state);
                processes[i].thickness += 0.01f * curand_normal(rand_state);
                
                processes[i].length = max(1.0f, min(processes[i].length, 100.0f));
                processes[i].thickness = max(0.1f, min(processes[i].thickness, 2.0f));

                Vector3D growth_direction = Vector3D(
                    curand_normal(rand_state),
                    curand_normal(rand_state),
                    curand_normal(rand_state)
                ).normalize();
                processes[i].tip = processes[i].tip.add(growth_direction.multiply(0.1f));
            }
        }
    }

    __device__ void updateAstrocyte(float dt, float* neuron_data, int num_nearby_neurons, curandState* rand_state) {
        for (int i = 0; i < num_nearby_neurons; ++i) {
            float* neuron = &neuron_data[i * 10];
            float neuron_glutamate = neuron[0];
            float neuron_potassium = neuron[1];
            float neuron_lactate = neuron[2];

            float glutamate_uptake = min(neuron_glutamate, glutamate_uptake_rate * dt);
            neuron[0] -= glutamate_uptake;
            glutamate_concentration += glutamate_uptake;

            float potassium_buffering = (neuron_potassium - 5.0f) * potassium_buffering_rate * dt;
            neuron[1] -= potassium_buffering;
            potassium_concentration += potassium_buffering;

            if (atp_concentration > 5.0f && neuron_lactate < 1.0f) {
                float lactate_transfer = min(lactate_concentration, 0.1f * dt);
                lactate_concentration -= lactate_transfer;
                neuron[2] += lactate_transfer;
            }

            if (glutamate_uptake > 0.05f) {
                triggerCalciumWave(dt);
            }
        }

        if (calcium_concentration > 0.5f && curand_uniform(rand_state) < 0.1f * dt) {
            releaseGliotransmitter(neuron_data, num_nearby_neurons);
        }
    }

    __device__ void updateOligodendrocyte(float dt, float* neuron_data, int num_nearby_neurons) {
        for (int i = 0; i < num_nearby_neurons; ++i) {
            float* neuron = &neuron_data[i * 10];
            float& axon_myelination = neuron[3];

            if (atp_concentration > 1.0f) {
                float myelination_amount = myelination_rate * dt;
                axon_myelination += myelination_amount;
                axon_myelination = min(axon_myelination, 1.0f);
                atp_concentration -= myelination_amount;
            }

            if (lactate_concentration > 0.5f && neuron[2] < 1.0f) {
                float lactate_transfer = min(lactate_concentration, 0.05f * dt);
                lactate_concentration -= lactate_transfer;
                neuron[2] += lactate_transfer;
            }
        }
    }

    __device__ void updateMicroglia(float dt, float* neuron_data, int num_nearby_neurons, curandState* rand_state) {
        float inflammation_level = 0.0f;
        for (int i = 0; i < num_nearby_neurons; ++i) {
            float* neuron = &neuron_data[i * 10];
            inflammation_level += neuron[4];
        }
        inflammation_level /= num_nearby_neurons;

        if (inflammation_level > 0.5f) {
            phagocytosis_rate *= 2.0f;
            for (int i = 0; i < num_nearby_neurons; ++i) {
                float* neuron = &neuron_data[i * 10];
                float debris = neuron[5];
                float phagocytosed = min(debris, phagocytosis_rate * dt);
                neuron[5] -= phagocytosed;
                
                neuron[4] += 0.1f * dt;
            }
        } else {
            phagocytosis_rate = max(phagocytosis_rate * 0.9f, 0.02f);
        }

        if (curand_uniform(rand_state) < 0.1f * dt) {
            position = position.add(Vector3D(
                curand_normal(rand_state),
                curand_normal(rand_state),
                curand_normal(rand_state)
            ).normalize().multiply(0.1f));
        }
    }

    __device__ void updateEpendymal(float dt, float* neuron_data, int num_nearby_neurons) {
        float csf_produced = csf_production_rate * dt;
        
        for (int i = 0; i < num_nearby_neurons; ++i) {
            float* neuron = &neuron_data[i * 10];
            neuron[6] += csf_produced / num_nearby_neurons;
        }

        if (atp_concentration > 0.5f) {
            atp_concentration -= 0.1f * dt;
        }
    }

    __device__ void triggerCalciumWave(float dt) {
        ip3_concentration += 0.5f * dt;
        ip3_concentration = min(ip3_concentration, 10.0f);
    }

    __device__ void releaseGliotransmitter(float* neuron_data, int num_nearby_neurons) {
        for (int i = 0; i < num_nearby_neurons; ++i) {
            float* neuron = &neuron_data[i * 10];
            neuron[7] += 0.1f;
        }
        calcium_concentration *= 0.9f; 
    }
};

__global__ void update_glia_kernel(Glia* glia, int num_glia, float dt, float* neuron_data, int* num_nearby_neurons, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_glia) {
        glia[idx].update(dt, &neuron_data[idx * MAX_NEARBY_NEURONS * 10], num_nearby_neurons[idx], &rand_states[idx]);
    }
}

void launch_glia_update(Glia* d_glia, int num_glia, float dt, float* d_neuron_data, int* d_num_nearby_neurons, curandState* d_rand_states) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_glia + threads_per_block - 1) / threads_per_block;
    update_glia_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_glia, num_glia, dt, d_neuron_data, d_num_nearby_neurons, d_rand_states);
    cudaDeviceSynchronize();
}

__global__ void glia_extracellular_interaction_kernel(Glia* glia, int num_glia, float* extracellular_space, int3 space_dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_glia) {
        Glia& cell = glia[idx];
        Vector3D pos = cell.getPosition();
        int x = static_cast<int>(pos.x);
        int y = static_cast<int>(pos.y);
        int z = static_cast<int>(pos.z);

        if (x >= 0 && x < space_dims.x && y >= 0 && y < space_dims.y && z >= 0 && z < space_dims.z) {
            int space_idx = z * space_dims.x * space_dims.y + y * space_dims.x + x;

            switch (cell.getType()) {
                case Glia::ASTROCYTE:
                    atomicAdd(&extracellular_space[space_idx * 4 + 0], -0.1f);  
                    atomicAdd(&extracellular_space[space_idx * 4 + 1], -0.05f); 
                    break;
                case Glia::OLIGODENDROCYTE:
                    atomicAdd(&extracellular_space[space_idx * 4 + 2], 0.02f);  
                    break;
                case Glia::MICROGLIA:
                    atomicAdd(&extracellular_space[space_idx * 4 + 3], -0.05f); 
                    break;
                case Glia::EPENDYMAL:
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dz = -1; dz <= 1; dz++) {
                                int nx = x + dx, ny = y + dy, nz = z + dz;
                                if (nx >= 0 && nx < space_dims.x && ny >= 0 && ny < space_dims.y && nz >= 0 && nz < space_dims.z) {
                                    int nidx = nz * space_dims.x * space_dims.y + ny * space_dims.x + nx;
                                    atomicAdd(&extracellular_space[nidx * 4 + 2], 0.01f);
                                }
                            }
                        }
                    }
                    break;
            }
        }
    }
}

void launch_glia_extracellular_interaction(Glia* d_glia, int num_glia, float* d_extracellular_space, int3 space_dims) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_glia + threads_per_block - 1) / threads_per_block;
    glia_extracellular_interaction_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_glia, num_glia, d_extracellular_space, space_dims);
    cudaDeviceSynchronize();
}

struct GliaVisualizationData {
    Vector3D position;
    float size;
    float4 color;
    Vector3D processes[MAX_PROCESSES];
    int num_processes;
};

__global__ void prepare_glia_visualization_kernel(Glia* glia, int num_glia, GliaVisualizationData* viz_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_glia) {
        Glia& cell = glia[idx];
        viz_data[idx].position = cell.getPosition();
        viz_data[idx].size = cell.getSize();
        viz_data[idx].color = cell.getColor();
        viz_data[idx].num_processes = cell.getNumProcesses();
        for (int i = 0; i < viz_data[idx].num_processes; i++) {
            viz_data[idx].processes[i] = cell.getProcessTip(i);
        }
    }
}

void launch_prepare_glia_visualization(Glia* d_glia, int num_glia, GliaVisualizationData* d_viz_data) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_glia + threads_per_block - 1) / threads_per_block;
    prepare_glia_visualization_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_glia, num_glia, d_viz_data);
    cudaDeviceSynchronize();
}

__global__ void inter_glial_communication_kernel(Glia* glia, int num_glia, int* nearby_glia_indices, int* num_nearby_glia, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_glia) {
        int start_idx = idx * MAX_NEARBY_GLIA;
        int count = num_nearby_glia[idx];
        Glia* nearby_glia = (Glia*)malloc(count * sizeof(Glia));
        for (int i = 0; i < count; i++) {
            nearby_glia[i] = glia[nearby_glia_indices[start_idx + i]];
        }
        glia[idx].communicateWithNearbyGlia(nearby_glia, count, dt);
        free(nearby_glia);
    }
}

void launch_inter_glial_communication(Glia* d_glia, int num_glia, int* d_nearby_glia_indices, int* d_num_nearby_glia, float dt) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_glia + threads_per_block - 1) / threads_per_block;
    inter_glial_communication_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_glia, num_glia, d_nearby_glia_indices, d_num_nearby_glia, dt);
    cudaDeviceSynchronize();
}