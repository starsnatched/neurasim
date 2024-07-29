#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include "vector3d.cu"

#define MAX_SYNAPSES 1000
#define MAX_AXONS 10
#define MAX_DENDRITES 20

class Neuron {
public:
    Vector3D soma_position;
    thrust::device_vector<Vector3D> dendritic_tree;
    thrust::device_vector<Vector3D> axon_segments;
    float membrane_capacitance;
    float leak_conductance;

    __host__ __device__ Neuron() : 
        position(),
        type(curand_uniform(&local_rand_state) < 0.8f ? EXCITATORY : INHIBITORY),
        voltage(-70.0f),
        threshold(-55.0f),
        is_spiking(false),
        refractory_period(2.0f),
        refractory_timer(0.0f),
        time(0.0f),
        received_signals(0.0f),
        target_voltage(-70.0f),
        voltage_recovery_rate(0.1f),
        target_activation(0.0f),
        activation_recovery_rate(0.05f),
        target_stimulation_level(0.0f),
        stimulation_recovery_rate(0.01f),
        atp(1000000.0f),
        atp_consumption_rate(100.0f),
        atp_production_rate(110.0f),
        channel_open_probability(0.5f),
        vesicle_release_probability(0.3f),
        network_activity(0.0f),
        a(0.02f),
        b(0.2f),
        c(-65.0f),
        d(2.0f),
        u(b * voltage),
        g_Na(120.0f),
        g_K(36.0f),
        g_L(0.3f),
        E_Na(55.0f),
        E_K(-77.0f),
        E_L(-54.0f),
        m(0.05f),
        h(0.54f),
        n(0.34f),
        capacitance(1.0f),
        activation(0.0f),
        flash_duration(0.0f),
        stimulation_level(0.0f),
        stimulation_decay_rate(0.1f),
        activation_threshold(0.5f),
        bdnf_level(1.0f),
        growth_factor(1.0f),
        branch_probability(0.05f),
        protein_synthesis_rate(0.1f),
        gene_expression_level(1.0f),
        metabolic_rate(1.0f),
        calcium_concentration(0.0001f),
        potassium_concentration(5.0f),
        sodium_concentration(15.0f),
        resting_potential(-70.0f),
        absolute_refractory_period(1.0f),
        relative_refractory_period(5.0f),
        neurotransmitter_release_probability(0.5f),
        neurotransmitter_reuptake_rate(0.1f),
        dopamine_receptors(0.5f + 0.5f * curand_uniform(&local_rand_state)),
        serotonin_receptors(0.5f + 0.5f * curand_uniform(&local_rand_state)),
        dopamine_level(0.0f),
        serotonin_level(0.0f)
    {
        initializeNeurotransmitters();
        initializeIonChannels();
        initializeSignalingPathways();
        initializeMetabolism();
        initializeGeneExpression();
        initializePlasticity();
        initializeNeuromodulation();
        initializeGlialInteraction();
    }

    struct IonChannel {
        float activation;
        float inactivation;
        float conductance;
        Vector3D location;
    };
    thrust::device_vector<IonChannel> sodium_channels;
    thrust::device_vector<IonChannel> potassium_channels;
    thrust::device_vector<IonChannel> calcium_channels;
    
    struct SignalingMolecule {
        float concentration;
        float diffusion_rate;
    };
    SignalingMolecule calcium;
    SignalingMolecule IP3;
    SignalingMolecule cAMP;
    
    struct Protein {
        float concentration;
        float synthesis_rate;
        float degradation_rate;
        Vector3D current_location;
    };
    thrust::device_vector<Protein> synthesized_proteins;
    
    struct Synapse {
        Vector3D location;
        float strength;
        float plasticity_rate;
        thrust::device_vector<float> neurotransmitter_vesicles;
        float release_probability;
        float reuptake_rate;
    };
    thrust::device_vector<Synapse> synapses;
    
    struct NeuromodulatorReceptor {
        float density;
        float sensitivity;
        Vector3D location;
    };
    thrust::device_vector<NeuromodulatorReceptor> dopamine_receptors;
    thrust::device_vector<NeuromodulatorReceptor> serotonin_receptors;
    
    struct Gene {
        float expression_level;
        float transcription_rate;
        float translation_rate;
        float mRNA_degradation_rate;
        float protein_degradation_rate;
    };
    thrust::device_vector<Gene> genes;
    
    struct EpigeneticMark {
        float methylation_level;
        float acetylation_level;
        int target_gene_index;
    };
    thrust::device_vector<EpigeneticMark> epigenetic_marks;
    
    struct Mitochondrion {
        Vector3D location;
        float atp_production_rate;
        float efficiency;
    };
    thrust::device_vector<Mitochondrion> mitochondria;

    __device__ void update(float dt, curandState* rand_state) {
        if (refractory_timer > 0.0f) {
            updateRefractory(dt);
            return;
        }

        if (atp <= 0.0f) {
            return;
        }

        updateVoltage(dt);
        updateActivation(dt);
        updateStimulationLevel(dt);
        updateATP(dt);
        updateIonChannels(dt);
        updateCalciumDynamics(dt);
        updateSignalingPathways(dt);
        updateSynapticPlasticity(dt);
        updateGeneExpression(dt);
        updateMetabolism(dt);
        updateNeuromodulation(dt);
        updateGlialInteraction(dt);

        if (voltage >= threshold && !is_spiking && refractory_timer <= 0.0f) {
            spike();
        }

        time += dt;
        updateNeurotransmitters(dt);
        updateBranchingAndGrowth(dt, rand_state);
    }

    __device__ void stimulate(float amount) {
        stimulation_level += amount;
        stimulation_level = min(stimulation_level, 0.9f);
        voltage += amount * 30.0f;

        if (!is_spiking && refractory_timer <= 0.0f) {
            spike();
        }

        flash_duration = 1.0f;
    }

    __device__ void receive_signal(float strength, NeuronType signal_type) {
        if (atp < 500.0f || refractory_timer > 0.0f) {
            return;
        }

        if (signal_type == EXCITATORY) {
            activation += strength;
        } else if (signal_type == INHIBITORY) {
            activation -= strength;
        }

        activation = max(0.0f, min(activation, 1.0f));
        stimulation_level += fabsf(strength) * 0.1f;
        stimulation_level = max(0.0f, min(stimulation_level, 1.0f));

        voltage += strength * 5.0f;
        atp -= 500.0f;

        if (voltage >= threshold) {
            spike();
        }
    }

    __device__ void update(float dt, curandState* rand_state);
    __device__ void propagate_action_potential();
    __device__ void update_ion_channels(float dt);
    __device__ void update_intracellular_signaling(float dt);
    __device__ void update_protein_synthesis(float dt);
    __device__ void update_synapses(float dt);
    __device__ void update_gene_expression(float dt);
    __device__ void update_epigenetics(float dt);
    __device__ void update_metabolism(float dt);
    __device__ const Vector3D& getPosition() const { return position; }
    __device__ void setPosition(const Vector3D& pos) { position = pos; }
    __device__ NeuronType getType() const { return type; }
    __device__ float getVoltage() const { return voltage; }
    __device__ bool isSpiking() const { return is_spiking; }
    __device__ float getActivation() const { return activation; }
    __device__ float getStimulationLevel() const { return stimulation_level; }

private:
    enum NeuronType { EXCITATORY, INHIBITORY };

    Vector3D position;
    NeuronType type;
    float voltage;
    float threshold;
    bool is_spiking;
    float refractory_period;
    float refractory_timer;
    float time;
    float received_signals;
    float target_voltage;
    float voltage_recovery_rate;
    float target_activation;
    float activation_recovery_rate;
    float target_stimulation_level;
    float stimulation_recovery_rate;
    float atp;
    float atp_consumption_rate;
    float atp_production_rate;
    float channel_open_probability;
    float vesicle_release_probability;
    float network_activity;
    float a, b, c, d, u;
    float g_Na, g_K, g_L, E_Na, E_K, E_L;
    float m, h, n;
    float capacitance;
    float activation;
    float flash_duration;
    float stimulation_level;
    float stimulation_decay_rate;
    float activation_threshold;
    float bdnf_level;
    float growth_factor;
    float branch_probability;
    float protein_synthesis_rate;
    float gene_expression_level;
    float metabolic_rate;
    float calcium_concentration;
    float potassium_concentration;
    float sodium_concentration;
    float resting_potential;
    float absolute_refractory_period;
    float relative_refractory_period;
    float neurotransmitter_release_probability;
    float neurotransmitter_reuptake_rate;
    float dopamine_receptors;
    float serotonin_receptors;
    float dopamine_level;
    float serotonin_level;

    thrust::device_vector<Vector3D> axons;
    thrust::device_vector<Vector3D> dendrites;
    thrust::device_vector<int> synapses;

    struct Neurotransmitter {
        float vesicles;
        float recycling_pool;
        float readily_releasable_pool;
        float synthesis_rate;
    };
    Neurotransmitter neurotransmitters[6];

    struct IonChannel {
        float activation;
        float inactivation;
        float density;
        float conductance;
    };
    IonChannel ion_channels[10];

    struct SignalingPathway {
        float activation;
    };
    SignalingPathway signaling_pathways[4];

    struct Metabolism {
        float atp;
        float adp;
        float amp;
        float glucose;
        float lactate;
        float oxygen;
        float mitochondria;
        float creatine_phosphate;
        float creatine;
        float nad;
        float nadh;
        float fadh2;
    };
    Metabolism metabolism;

    struct GeneExpression {
        float mrna;
        float protein;
        float methylation;
        float acetylation;
        float phosphorylation;
    };
    GeneExpression gene_expression[5];

    struct Plasticity {
        float a_plus;
        float a_minus;
        float tau_plus;
        float tau_minus;
        bool nearest_neighbor;
        float target_activity;
        float adjustment_rate;
        bool sliding_threshold;
        float threshold;
        float rate;
        float bcm_sliding_modification_threshold;
    };
    Plasticity plasticity;

    struct Neuromodulation {
        float dopamine_sensitivity;
        float serotonin_sensitivity;
        float norepinephrine_sensitivity;
        float acetylcholine_sensitivity;
    };
    Neuromodulation neuromodulation;

    struct GlialInteraction {
        float astrocyte_coverage;
        float oligodendrocyte_coverage;
        int microglia_state;
    };
    GlialInteraction glial_interaction;

    curandState local_rand_state;

    __device__ void initializeNeurotransmitters() {
        const char* nt_names[] = {"glutamate", "gaba", "dopamine", "serotonin", "norepinephrine", "acetylcholine"};
        for (int i = 0; i < 6; ++i) {
            neurotransmitters[i] = {1000.0f, 100.0f, 10.0f, 0.1f};
        }
    }

    __device__ void initializeIonChannels() {
        const char* channel_names[] = {"nav1_1", "nav1_6", "kv1", "kv3", "kv7", "cav2_1", "cav3_1", "hcn1", "hcn2", "kir"};
        for (int i = 0; i < 10; ++i) {
            ion_channels[i] = {0.0f, 1.0f, 10.0f, 20.0f};
        }
    }

    __device__ void initializeSignalingPathways() {
        const char* pathway_names[] = {"camkii", "pka", "mapk", "pi3k"};
        for (int i = 0; i < 4; ++i) {
            signaling_pathways[i] = {0.0f};
        }
    }

    __device__ void initializeMetabolism() {
        metabolism = {1000000.0f, 100000.0f, 10000.0f, 500000.0f, 100000.0f, 1000000.0f, 1000.0f, 500000.0f, 100000.0f, 100000.0f, 10000.0f, 10000.0f};
    }

    __device__ void initializeGeneExpression() {
        const char* gene_names[] = {"bdnf", "creb", "arc", "fos", "egr1"};
        for (int i = 0; i < 5; ++i) {
            gene_expression[i] = {100.0f, 1000.0f, 0.1f, 0.1f, 0.1f};
        }
    }

    __device__ void initializePlasticity() {
        plasticity = {0.1f, 0.12f, 20.0f, 20.0f, true, 0.1f, 0.01f, true, 0.5f, 0.001f, 0.5f};
    }

    __device__ void initializeNeuromodulation() {
        neuromodulation = {1.0f, 1.0f, 1.0f, 1.0f};
    }

    __device__ void initializeGlialInteraction() {
        glial_interaction = {0.5f, 0.3f, 0};
    }

    __device__ void updateRefractory(float dt) {
        refractory_timer -= dt;
        if (refractory_timer <= relative_refractory_period) {
            voltage = resting_potential + (voltage - resting_potential) * (1.0f - (refractory_timer / relative_refractory_period));
        } else {
            voltage = resting_potential;
        }
    }

    __device__ void updateVoltage(float dt) {
        voltage += received_signals * 5.0f;
        received_signals = 0.0f;

        float dv = (0.04f * voltage + 5.0f) * voltage + 140.0f - u;
        float du = a * (b * voltage - u);

        voltage += dv * dt;
        u += du * dt;

        voltage += (-0.1f * (voltage - resting_potential)) * dt;
    }

    __device__ void updateActivation(float dt) {
        activation *= expf(-dt);
    }

    __device__ void updateStimulationLevel(float dt) {
        stimulation_level *= expf(-stimulation_decay_rate * dt);
        bdnf_level = 1.0f + 0.5f * stimulation_level;
        growth_factor = 1.0f + 0.2f * stimulation_level;
    }

    __device__ void updateATP(float dt) {
        atp -= atp_consumption_rate * dt;
        atp += atp_production_rate * dt;
    }

    __device__ void updateIonChannels(float dt) {
        for (int i = 0; i < 10; ++i) {
            ion_channels[i].activation += (channel_open_probability - ion_channels[i].activation) * 0.1f * dt;
        }
    }

    __device__ void updateCalciumDynamics(float dt) {
        if (is_spiking) {
            calcium_concentration += 0.0001f;
        }
        calcium_concentration *= expf(-0.1f * dt);
        calcium_concentration = max(0.0001f, min(0.001f, calcium_concentration));
    }

    __device__ void updateSignalingPathways(float dt) {
        for (int i = 0; i < 4; ++i) {
            if (calcium_concentration > 0.0002f) {
                signaling_pathways[i].activation += 0.1f * dt;
            } else {
                signaling_pathways[i].activation *= expf(-dt / 1000.0f);
            }
            signaling_pathways[i].activation = min(1.0f, signaling_pathways[i].activation);
        }
    }

    __device__ void updateSynapticPlasticity(float dt) {
        for (int i = 0; i < synapses.size(); ++i) {
            float& weight = synapses[i];
            float target_weight = 0.5f;
            weight += (target_weight - weight) * plasticity.adjustment_rate * dt;
            weight = max(0.0f, min(1.0f, weight));
        }
    }

    __device__ void updateGeneExpression(float dt) {
        for (int i = 0; i < 5; ++i) {
            if (is_spiking) {
                gene_expression[i].mrna += 1.0f * (1.0f - gene_expression[i].methylation) * (1.0f + gene_expression[i].acetylation);
            }
            gene_expression[i].protein += gene_expression[i].mrna * 0.1f * dt;
            gene_expression[i].mrna *= expf(-dt / 3600.0f);
            gene_expression[i].protein *= expf(-dt / 86400.0f);

            if (signaling_pathways[0].activation > 0.5f) {
                gene_expression[i].acetylation += 0.01f * dt;
            } else {
                gene_expression[i].acetylation -= 0.005f * dt;
            }
            gene_expression[i].acetylation = max(0.0f, min(1.0f, gene_expression[i].acetylation));

            if (signaling_pathways[2].activation > 0.5f) { 
                gene_expression[i].methylation -= 0.01f * dt;
            } else {
                gene_expression[i].methylation += 0.005f * dt;
            }
            gene_expression[i].methylation = max(0.0f, min(1.0f, gene_expression[i].methylation));
        }
    }

    __device__ void updateMetabolism(float dt) {
        float glycolysis_rate = min(metabolism.glucose, metabolism.adp / 2.0f, metabolism.nad / 2.0f) * 0.1f * dt;
        metabolism.glucose -= glycolysis_rate;
        metabolism.adp -= 2.0f * glycolysis_rate;
        metabolism.nad -= 2.0f * glycolysis_rate;
        metabolism.atp += 2.0f * glycolysis_rate;
        metabolism.nadh += 2.0f * glycolysis_rate;
        metabolism.lactate += 2.0f * glycolysis_rate;

        float oxphos_rate = min(metabolism.nadh, metabolism.fadh2, metabolism.adp / 3.0f, metabolism.oxygen / 2.0f) * 0.1f * dt;
        metabolism.nadh -= oxphos_rate;
        metabolism.fadh2 -= oxphos_rate;
        metabolism.adp -= 3.0f * oxphos_rate;
        metabolism.oxygen -= 2.0f * oxphos_rate;
        metabolism.atp += 15.0f * oxphos_rate;
        metabolism.nad += oxphos_rate;

        float creatine_shuttle_rate = (metabolism.creatine_phosphate * metabolism.adp - metabolism.creatine * metabolism.atp / 10.0f) * 0.01f * dt;
        metabolism.creatine_phosphate -= creatine_shuttle_rate;
        metabolism.adp -= creatine_shuttle_rate;
        metabolism.creatine += creatine_shuttle_rate;
        metabolism.atp += creatine_shuttle_rate;
    }

    __device__ void updateNeuromodulation(float dt) {
        dopamine_level *= expf(-dt / 10.0f);
        serotonin_level *= expf(-dt / 20.0f);

        threshold -= dopamine_level * dopamine_receptors * 5.0f;
        threshold = max(-65.0f, threshold);

        threshold += serotonin_level * serotonin_receptors * 3.0f;
        threshold = min(-45.0f, threshold);
    }

    __device__ void updateGlialInteraction(float dt) {
        neurotransmitters[0].recycling_pool *= 1.0f - 0.1f * glial_interaction.astrocyte_coverage * dt;

        for (int i = 0; i < axons.size(); ++i) {
            float& myelination = axons[i].z; 
            myelination += 0.001f * glial_interaction.oligodendrocyte_coverage * dt;
            myelination = min(1.0f, myelination);
        }

        if (glial_interaction.microglia_state == 1) {
            atp -= 10.0f * dt;
            gene_expression[0].protein *= 0.99f; 
        }
    }

    __device__ void updateNeurotransmitters(float dt) {
        for (int i = 0; i < 6; ++i) {
            neurotransmitters[i].vesicles += neurotransmitters[i].synthesis_rate * dt;
            if (is_spiking) {
                float release = min(neurotransmitters[i].readily_releasable_pool, 5.0f);
                neurotransmitters[i].readily_releasable_pool -= release;
                neurotransmitters[i].recycling_pool += release * 0.8f;
            }
            float replenishment = min(neurotransmitters[i].recycling_pool, (10.0f - neurotransmitters[i].readily_releasable_pool) * 0.1f);
            neurotransmitters[i].recycling_pool -= replenishment;
            neurotransmitters[i].readily_releasable_pool += replenishment;
        }
    }

    __device__ void updateBranchingAndGrowth(float dt, curandState* rand_state) {
        if (curand_uniform(rand_state) < branch_probability * dt) {
            if (curand_uniform(rand_state) < 0.5f && axons.size() < MAX_AXONS) {
                Vector3D new_direction(
                    curand_normal(rand_state),
                    curand_normal(rand_state),
                    curand_normal(rand_state)
                );
                axons.push_back(new_direction.normalize());
            } else if (dendrites.size() < MAX_DENDRITES) {
                Vector3D new_direction(
                    curand_normal(rand_state),
                    curand_normal(rand_state),
                    curand_normal(rand_state)
                );
                dendrites.push_back(new_direction.normalize());
            }
        }

        for (int i = 0; i < axons.size(); ++i) {
            axons[i] = axons[i].add(axons[i].multiply(growth_factor * dt));
        }
        for (int i = 0; i < dendrites.size(); ++i) {
            dendrites[i] = dendrites[i].add(dendrites[i].multiply(growth_factor * 0.5f * dt));
        }
    }

    __device__ void spike() {
        is_spiking = true;
        voltage = 40.0f;
        refractory_timer = refractory_period;
        activation = 1.0f;
        releaseNeurotransmitters();
        flash_duration = 1.0f;
        voltage = c;
    }

    __device__ void releaseNeurotransmitters() {
        if (curand_uniform(&local_rand_state) < vesicle_release_probability) {
            if (type == EXCITATORY) {
                neurotransmitters[0].readily_releasable_pool -= 1.0f; 
            } else if (type == INHIBITORY) {
                neurotransmitters[1].readily_releasable_pool -= 1.0f; 
            }
            atp -= 1000.0f;
        }
        neurotransmitters[2].readily_releasable_pool -= 0.1f; 
        neurotransmitters[3].readily_releasable_pool -= 0.05f;
    }
};

__global__ void update_neurons_kernel(Neuron* neurons, int num_neurons, float dt, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        neurons[idx].update(dt, &rand_states[idx]);
    }
}

void launch_neuron_update(Neuron* d_neurons, int num_neurons, float dt, curandState* d_rand_states) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;
    update_neurons_kernel<<<blocks_per_grid, threads_per_block>>>(d_neurons, num_neurons, dt, d_rand_states);
    cudaDeviceSynchronize();
}