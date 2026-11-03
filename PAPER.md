# GraftBrain: Closed-Loop Embodied Reinforcement Learning via Multi-Connectome Cross-Wiring and Bio-Inspired Circuit Modules

**Kai Piper**  
*Independent Research / Laboratory for Connectomic Intelligence*  
`kai9987kai`

---

## Abstract

Recent breakthroughs in transmission electron microscopy (TEM) have produced high-resolution, synapse-level connectomes of the adult and larval fruit fly (*Drosophila melanogaster*), including the full female adult brain (FlyWire FAFB v783), the Janelia Hemibrain, the Male Adult Nerve Cord (MANC), the Female Adult Nerve Cord (FANC), MaleCNS, the BANC central nervous system, the optic lobe, and the complete larval nervous system. However, existing computational models treat these datasets either as isolated static topological graphs or as detached open-loop firing simulations. Here, we present **GraftBrain**, a unified, closed-loop embodied computational framework that fuses all eight primary *Drosophila* connectome datasets into a single cross-wired multi-regional network. We introduce a multi-site structural grafting methodology comprising: (1) anatomical interface sites mirroring biological connectives (descending brain pathways to motor nerve cord centers, ascending feedback, and optic-to-central projections), (2) homologous cell-type coupling preserving shared cell-type nomenclature, and (3) reciprocal distributed central bridges. 

The unified network consists of 12,000 neurons, 252,447 frozen biological synapses, and 95,920 plastic graft synapses across 177 distinct sites. We embed this connectome within a differentiable neural architecture augmented with three bio-computational modules: a 16-wedge Central Complex (CX) ring attractor compass providing path-integrated heading representations, a sparse Kenyon Cell (KC) Mushroom Body (MB) associative circuit with dopaminergic (DAN) credit assignment, and homeostatic intrinsic plasticity regulating network stability. We optimize the system end-to-end using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) and three-factor Hebbian learning in a 3D virtual environment across nine sensorimotor, fine-motor manipulation, and cognitive memory tasks. By replacing standard sparse matrix multiplication with vectorized `index_add` operations, GraftBrain achieves full closed-loop real-time execution (~14 steps/second on consumer CPU). We demonstrate that bio-inspired ring attractor and mushroom body modules substantially accelerate goal-directed navigation and delayed associative recall, providing a foundation for scalable, embodied connectome-constrained artificial intelligence.

---

## 1. Introduction

A central objective of computational neuroscience and artificial intelligence is to understand how biological connectivity gives rise to adaptive behavior. For decades, connectionist models have relied on idealized topologies—such as multilayer perceptrons, convolutional layers, and self-attention transformers—optimizing weights from unconstrained random initializations (LeCun et al., 2015; Vaswani et al., 2017). Concurrently, connectomics has undergone a technological revolution: advances in automated serial-section electron microscopy, volumetric computer vision, and distributed human proofreading have produced complete, synapse-level wiring diagrams of the nervous system of *Drosophila melanogaster* (Dorkenwald et al., 2024; Schlegel et al., 2024; Scheffer et al., 2020; Takemura et al., 2023).

Despite the availability of these wiring diagrams, significant computational barriers remain:
1. **Compartmentalized Disconnection**: Public datasets exist in isolation—such as the brain-only FlyWire dataset, the ventral nerve cord (MANC/FANC) datasets, and the optic lobe connectome. In a living organism, behavior arises from continuous, bidirectional loop dynamics traversing optic neuropils, central decision centers, descending premotor tracts through the neck connective, and ascending proprioceptive feedback from the nerve cord.
2. **Open-Loop Detachment**: The vast majority of whole-brain connectome simulations (e.g., Shiu et al., 2024) evaluate neural activity in response to artificial current pulses delivered in open-loop settings, detached from physical embodiment, environmental feedback, or closed-loop behavioral closed loops.
3. **Plasticity vs. Structure**: Connectome data provide a snapshot of fixed structural connectivity, but biological intelligence relies on synaptic plasticity, neuromodulation, and ongoing credit assignment.

To address these challenges, we introduce **GraftBrain**, an end-to-end differentiable computational platform that bridges multi-connectome reconstructions into a functioning, embodied agent. We formulate connectome fusion as a **multi-site structural grafting problem**, introducing targeted anatomical interfaces, homologous cell-type bridges, and reciprocal central links between separate biological reconstructions. 

Furthermore, we couple the grafted core with specialized biological circuits—specifically the **Central Complex (CX) ring attractor** (Seelig & Jayaraman, 2015; Hulse et al., 2021) and the **Mushroom Body (MB)** associative learning network (Aso et al., 2014)—and train the entire system closed-loop in a 3D physics environment.

---

## 2. Related Work

### 2.1 Whole-Brain Drosophila Connectomics
The completion of the FlyWire FAFB whole-brain connectome (Dorkenwald et al., 2024; Schlegel et al., 2024) and the Janelia Hemibrain (Scheffer et al., 2020) provides an unprecedented catalog of over 139,000 neurons and 50 million synaptic connections, annotated by cell type and neurotransmitter predictions. Shiu et al. (2024) constructed a leaky integrate-and-fire (LIF) whole-brain simulation based on FlyWire, demonstrating that resting-state dynamics and localized sensory activations spread through identifiable anatomical pathways. In sensory processing, Lappalainen et al. (2024) demonstrated that connectome-constrained recurrent neural networks predict neural responses in the fly visual system (T4/T5 motion detection pathways) with greater biological accuracy than unconstrained artificial models.

### 2.2 Central Complex Ring Attractors and Vector Navigation
In *Drosophila*, spatial navigation and heading estimation are computed in the central complex, specifically within the ellipsoid body (EB) and protocerebral bridge (PB). Experimental and theoretical studies have established that E-PG (Ellipsoid body-Protocerebral bridge-Gall) compass neurons form a continuous ring attractor network (Kim et al., 2017; Turner-Evans et al., 2020). An localized bump of neural activity represents the insect's current head direction in azimuthal space. This bump is continuously updated via two complementary mechanisms: self-motion integration (angular velocity signals delivered by P-EN neurons) and visual landmark anchoring (delivered by ring neurons).

### 2.3 Mushroom Body Plasticity and Neuromodulation
Associative learning and cognitive memory in insects reside primarily in the mushroom body (Cognigni et al., 2018). High-dimensional sensory inputs from projection neurons are converted into a sparse, decorrelated representation across thousands of Kenyon Cells (KCs) via claw morphology and winner-take-all feedback inhibition (Litwin-Kumar et al., 2017). Dopaminergic neurons (DANs) compartmentalize the mushroom body lobes into distinct functional units that signal reward (e.g., PAM cluster) or punishment/aversive states (e.g., PPL1 cluster). DAN activation alters the synaptic weight between active KCs and Mushroom Body Output Neurons (MBONs), directly driving approach or avoidance behavior.

---

## 3. Multi-Connectome Ingestion and Grafting Architecture

### 3.1 Mathematical Formulation of Connectome Datasets
Each connectome dataset $k \in \{1, \dots, K\}$ is represented as a directed, signed, attributed graph:
$$\mathcal{G}_k = (\mathcal{V}_k, \mathcal{E}_k, \mathbf{W}_k, \mathbf{S}_k, \mathbf{T}_k, \mathbf{R}_k)$$
where:
* $\mathcal{V}_k$ is the set of $N_k$ reconstructed neurons.
* $\mathcal{E}_k \subseteq \mathcal{V}_k \times \mathcal{V}_k$ is the set of directed synaptic connections $(i, j)$ from presynaptic neuron $i$ to postsynaptic neuron $j$.
* $\mathbf{W}_k \in \mathbb{R}_{\ge 0}^{|\mathcal{E}_k|}$ represents synaptic weight, measured as the integer count of reconstructed electron-microscopy T-bars / polyadic synaptic contacts.
* $\mathbf{S}_k \in \{-1, +1\}^{|\mathcal{E}_k|}$ represents synaptic sign determined by predicted neurotransmitter identity:
  $$\mathbf{S}_k(i, j) = \begin{cases} -1 & \text{if } \text{NT}(i) \in \{\text{GABA}, \text{Glutamate}\} \\ +1 & \text{if } \text{NT}(i) \in \{\text{Acetylcholine}\} \end{cases}$$
  *(In the insect central nervous system, glutamate gates inhibitory glutamate-gated chloride channels [GluCl], acting primarily as an inhibitory neurotransmitter).*
* $\mathbf{T}_k$ is the cell-type classification string (e.g., `E-PG`, `P-EN`, `L1-L5`, `DNp01`).
* $\mathbf{R}_k \in \{\text{sensory}, \text{visual}, \text{central}, \text{descending}, \text{ascending}, \text{motor}\}$ denotes the macroscopic functional role.

To maintain real-time computational throughput on CPU hardware, large-scale releases are sub-sampled to an active core of $N_{\text{budget}} \le 1,500$ neurons per dataset by ranking neurons according to total synaptic degree:
$$\deg(i) = \sum_{j} \mathbf{W}(i, j) + \sum_{j} \mathbf{W}(j, i)$$

The eight supported datasets integrated into the unified graph are summarized in Table 1.

| Dataset Name | Anatomical Region | Specimen Sex / Stage | Structural Modality | License |
|---|---|---|---|---|
| **FlyWire (FAFB v783)** | Whole Brain | Adult Female | Serial-section TEM | CC-BY-NC-4.0 |
| **Hemibrain v1.2.1** | Central Brain | Adult Female | FIB-SEM | CC-BY-4.0 |
| **MaleCNS v1.0** | Brain + Nerve Cord | Adult Male | FIB-SEM | CC-BY-4.0 |
| **BANC** | Brain + Nerve Cord | Adult Female | Serial-section TEM | CC-BY-4.0 |
| **Optic Lobe** | Visual Neuropils | Adult Female | FIB-SEM | CC-BY-4.0 |
| **MANC v1.2.1** | Ventral Nerve Cord | Adult Male | FIB-SEM | CC-BY-4.0 |
| **FANC** | Ventral Nerve Cord | Adult Female | Serial-section TEM | Custom / Academic |
| **Larval Connectome** | Full CNS | 1st Instar Larva | Serial-section TEM | CC-BY-4.0 |

*Table 1: The eight canonical Drosophila connectome datasets integrated within GraftBrain.*

### 3.2 Multi-Site Grafting Formulation
Let $\mathcal{V} = \bigcup_{k=1}^K \mathcal{V}_k$ denote the unified population of $N = \sum_{k=1}^K N_k$ neurons, indexed globally using index offsets $O_k = \sum_{m=1}^{k-1} N_m$. Within-connectome connections form the frozen biological base matrix $\mathbf{W}_{\text{base}}$. Between-connectome connections form the plastic synthetic graft set $\mathcal{E}_{\text{graft}}$ with trainable weights $\mathbf{W}_{\text{graft}}$.

Grafting connections are established across three distinct structural mechanisms:

#### 1. Anatomical Interface Sites
Connections are instantiated to reflect the known macro-architecture of the insect nervous system:
* **Descending Pathways (Neck Connective)**: Brain descending neurons ($\mathbf{R}_a = \text{descending}$) project into nerve-cord premotor and motor circuits ($\mathbf{R}_b \in \{\text{motor}, \text{central}\}$):
  $$\mathcal{E}_{\text{desc}} = \{(u, v) \mid u \in \mathcal{V}_a^{\text{desc}}, v \in \mathcal{V}_b^{\text{motor}}\}$$
* **Ascending Pathways**: Nerve-cord ascending neurons ($\mathbf{R}_b = \text{ascending}$) project into central brain targets ($\mathbf{R}_a = \text{central}$), providing proprioceptive and motor-state feedback.
* **Optic Projection Pathways**: Optic lobe visual output neurons ($\mathbf{R}_a = \text{visual}$) project into central brain visual and anterior neuropils.

#### 2. Homologous Cell-Type Sites
Where distinct datasets share standardized anatomical cell-type nomenclature (e.g., mutual E-PG or Kenyon Cell annotations across FlyWire, Hemibrain, and MaleCNS), shared cell-type subsets $\mathcal{T}_{a, b} = \mathbf{T}_a \cap \mathbf{T}_b$ are linked via reciprocal homologous coupling:
$$\mathcal{E}_{\text{homolog}} = \{(u, v) \mid \mathbf{T}_a(u) = \mathbf{T}_b(v)\}$$

#### 3. Distributed Reciprocal Central Bridges
To allow distributed, non-bottlenecked cross-talk across distinct central neuropils, $K_{\text{bridge}} = 3$ independent, spatially separated clusters of central neurons ($\mathbf{R} = \text{central}$) are reciprocally cross-wired between each pair of connectomes.

The total grafted graph combines $N = 12,000$ neurons, $|\mathcal{E}_{\text{base}}| = 252,447$ fixed biological edges, and $|\mathcal{E}_{\text{graft}}| = 95,920$ plastic graft edges distributed across 177 distinct anatomical and synthetic sites.

---

## 4. Dynamical Rate Model and Biological Sub-Circuits

### 4.1 Vectorized Dynamical Rate Model
Neural activation is modeled as a continuous, bounded firing-rate dynamical system evaluated over $M = 4$ microsteps per simulation time-step $\Delta t = 0.1\text{ s}$. For neuron $i \in \{1, \dots, N\}$, the total synaptic current $x_i$ and updated firing rate $r_i \in [0, 1]$ are governed by:

$$x_i(t) = \sum_{j \in \mathcal{N}_{\text{base}}(i)} \mathbf{W}_{\text{base}}(j, i) r_j(t) + \sum_{k \in \mathcal{N}_{\text{graft}}(i)} \mathbf{W}_{\text{graft}}(k, i) r_k(t) + b_i + I_{\text{drive}, i}(t)$$

$$r_i(t + \delta\tau) = (1 - \alpha) r_i(t) + \alpha \cdot \text{ReLU}(\tanh(x_i(t))) + \sigma \xi_i(t)$$

where $\alpha = 0.35$ is the leak rate, $b_i$ is intrinsic baseline excitability, $\sigma = 0.01$ is background Gaussian membrane noise $\xi_i \sim \mathcal{N}(0, 1)$, and $I_{\text{drive}, i}$ represents injected sensory drive.

```
Algorithm 1: Vectorized Connectome Microstep via index_add
--------------------------------------------------------------------------------
Input: Presynaptic rates r in [0, 1]^N, drive I_drive in R^N, bias b in R^N
Output: Updated post-synaptic rates r' in [0, 1]^N

1: x <- zeros(N)
2: x <- x.index_add(dim=0, index=base_post, tensor=base_w * r[base_pre])
3: x <- x.index_add(dim=0, index=graft_post, tensor=graft_w * r[graft_pre])
4: x <- x + b + I_drive
5: r' <- (1 - alpha) * r + alpha * clamp(tanh(x), min=0.0)
6: if noise > 0 then r' <- r' + noise * randn(N)
7: r' <- clamp(r', min=0.0, max=1.0)
8: return r'
```

### 4.2 Homeostatic Plasticity Regulation
In recurrent networks of biological scale, unbalanced positive feedback loops can precipitate either quiescent extinction ($r \to 0$) or pathological saturation ($r \to 1$). To maintain the connectome within a responsive dynamic regime, we implement continuous homeostatic firing rate adaptation:
$$\bar{r}_i(t) = (1 - \beta_{\text{homeo}}) \bar{r}_i(t - 1) + \beta_{\text{homeo}} r_i(t)$$
$$b_i(t + 1) = \text{clamp}\left(b_i(t) + \eta_{\text{homeo}} \left(r^* - \bar{r}_i(t)\right), -0.5, +0.5\right)$$
where $\beta_{\text{homeo}} = 0.01$, $\eta_{\text{homeo}} = 0.0005$, and $r^* = 0.12$ is the target mean population firing rate.

### 4.3 Central Complex (CX) Ring Attractor Compass
To provide the agent with a robust internal coordinate frame for spatial navigation, we model the Ellipsoid Body (EB) as a continuous ring attractor comprising $K_{\text{wedge}} = 16$ discrete angular columns covering $\theta \in [-\pi, +\pi)$.

The internal heading angle $\theta_{\text{bump}}$ is updated dynamically via:
$$\theta_{\text{bump}}(t + 1) = \theta_{\text{bump}}(t) + \omega(t) \Delta t + \kappa_{\text{vis}} \Delta\theta_{\text{landmark}}$$
where $\omega(t)$ is the angular velocity from proprioception, and $\Delta\theta_{\text{landmark}}$ is the visual angular discrepancy between the heading bump and the brightest/nearest visual contrast detected by the retina.

The activation $c_k$ of compass wedge $k$ with preferred angle $\phi_k = -\pi + k \frac{2\pi}{16}$ follows a circular Gaussian tuning curve:
$$c_k = \frac{\exp\left( - \frac{\text{dist}_{S^1}(\phi_k, \theta_{\text{bump}})^2}{2 \sigma_{\text{cx}}^2} \right)}{\sum_{m=1}^{16} \exp\left( - \frac{\text{dist}_{S^1}(\phi_m, \theta_{\text{bump}})^2}{2 \sigma_{\text{cx}}^2} \right)}$$
with width $\sigma_{\text{cx}} = 0.45\text{ rad}$. The resulting 16-dimensional compass state $\mathbf{c} \in \mathbb{R}^{16}$ is injected directly into central brain and descending steering pathways.

### 4.4 Mushroom Body (MB) Associative Sub-Circuit
For associative memory and delayed-response cognitive tasks, we implement a modeled mushroom body:
1. **Sensory Claw Expansion**: Environmental cue vectors $\mathbf{u}_{\text{cue}} \in \mathbb{R}^4$ (encoding cue color and active task flags) are projected onto a population of $N_{\text{KC}} = 64$ Kenyon Cells via a fixed pseudo-random claw projection matrix $\mathbf{W}_{\text{claw}} \in \mathbb{R}^{64 \times 4}$.
2. **Winner-Take-All (WTA) Sparsity**: To mirror biological Kenyon cell coding ($~5-10\%$ sparsity; Litwin-Kumar et al., 2017), we enforce top-$k$ Winner-Take-All gating ($k = 4$, yielding exactly $6.25\%$ active KCs):
   $$a_i = \begin{cases} \text{ReLU}\left((\mathbf{W}_{\text{claw}} \mathbf{u}_{\text{cue}})_i\right) & \text{if } (\mathbf{W}_{\text{claw}} \mathbf{u}_{\text{cue}})_i \in \text{Top-}k \\ 0 & \text{otherwise} \end{cases}$$
3. **Mushroom Body Output Readout**: Plastic synaptic weights $\mathbf{W}_{\text{MBON}} \in \mathbb{R}^{2 \times 64}$ project KCs to two opponent output neurons: $\text{MBON}_+$ (approach) and $\text{MBON}_-$ (avoidance):
   $$\mathbf{y}_{\text{MBON}} = \mathbf{W}_{\text{MBON}} \mathbf{a}$$
4. **Dopaminergic Neuromodulation**: Synaptic eligibility traces $\mathbf{E}_{\text{MB}} \in \mathbb{R}^{2 \times 64}$ track paired pre- and post-synaptic co-activation:
   $$\mathbf{E}_{\text{MB}}(t) = 0.85 \mathbf{E}_{\text{MB}}(t - 1) + \tanh(\mathbf{y}_{\text{MBON}}(t)) \mathbf{a}(t)^T$$
   When primary rewards or temporal-difference errors $\delta_{\text{TD}}$ occur, simulated dopaminergic neurons (DANs) modulate the weights:
   $$\mathbf{W}_{\text{MBON}} \leftarrow \text{clamp}\left(\mathbf{W}_{\text{MBON}} + \eta_{\text{DAN}} \begin{bmatrix} +\delta_{\text{TD}} \\ -\delta_{\text{TD}} \end{bmatrix} \odot \mathbf{E}_{\text{MB}}, -1.0, +1.0\right)$$

---

## 5. Differentiable Neural Interface and Agent Embodiment

### 5.1 Multimodal Sensory Processing
The agent receives a heterogeneous sensory vector $\mathbf{o} \in \mathbb{R}^{146}$:
* **Ray-Cast Retina**: 33 rays distributed over a $120^\circ$ horizontal field of view ($[-60^\circ, +60^\circ]$) with range $d_{\max} = 9.0\text{ m}$. Each ray returns 4 channels: normalized depth $1 - d/d_{\max}$ and normalized RGB surface reflectance ($33 \times 4 = 132$ dimensions).
* **Proprioception**: 8 channels encoding arm reach ($[0.25, 0.9]\text{ m}$), arm height ($[0.0, 0.75]\text{ m}$), gripper aperture ($[0.0, 1.0]$), held object flag, linear velocity $v$, angular velocity $\omega$, and heading components $(\sin\theta, \cos\theta)$.
* **Task Context**: One-hot task indicator vector ($\mathbb{R}^9$), cue vector ($\mathbb{R}^4$), previous reward, and normalized episode progress.
* **Camera Visual Stream**: Live webcam stream (or synthetic stimulus) downsampled to $2 \times 48 \times 64$ (grayscale luminance + motion energy), processed via a 2-stage convolutional network into a 32-dimensional embedding $\mathbf{z}_{\text{cam}} \in \mathbb{R}^{32}$.

### 5.2 Sensorimotor Encoder and Decoder
The artificial neural interface connects the external world to the biological connectome:
* **Encoder MLP**: Maps the combined input $[\mathbf{o}, \mathbf{z}_{\text{cam}}, \mathbf{c}_{\text{compass}}, \mathbf{y}_{\text{MBON}}] \in \mathbb{R}^{200}$ through a 256-unit hidden layer to drive $N_{\text{sensory}} = 256$ designated sensory and visual neurons in the connectome.
* **Decoder MLP**: Reads the firing rates of $N_{\text{motor}} = 128$ motor/descending neurons, $N_{\text{probe}} = 256$ distributed central probe neurons, and the 2 MBON channels to parameterize a Gaussian policy distribution $\pi_\theta(\mathbf{a} \mid \mathbf{s}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma})$ over 5 continuous motor actuators:
  $$\mathbf{a} = [\text{drive } v, \text{turn } \omega, \Delta\text{reach}, \Delta\text{height}, \text{grip aperture}] \in [-1, 1]^5$$
  and estimates state value $V_\phi(\mathbf{s})$.

---

## 6. Optimization and Learning Architecture

GraftBrain trains both the artificial neural interface and the plastic graft synapses $\mathbf{W}_{\text{graft}}$ through a hybrid optimization scheme combining **PPO-GAE** with **three-factor Hebbian plasticity**.

### 6.1 Proximal Policy Optimization with GAE
For a rollout trajectory of length $T = 32$, Generalized Advantage Estimation (Schulman et al., 2015) computes advantage estimates $\hat{A}_t$:
$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}$$
where $\gamma = 0.98$ and $\lambda = 0.95$.

The policy parameters and graft weights are optimized via the clipped surrogate objective:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\hat{A}_t, \, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
where $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ with clipping threshold $\epsilon = 0.2$. The complete loss function combines policy, value, and entropy terms:
$$L_{\text{total}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \frac{1}{2} \text{SmoothL1}\left(V_\phi(s_t), \hat{R}_t\right) - c_2 \mathcal{H}\left(\pi_\theta(\cdot \mid s_t)\right)$$
with value coefficient $c_1 = 0.5$ and entropy coefficient $c_2 = 0.01$. Gradients are back-propagated through the unrolled recurrent microsteps of the connectome.

### 6.2 Three-Factor Hebbian Plasticity on Graft Synapses
In parallel with gradient descent, plastic graft synapses update continuously at every time step according to a biological three-factor Hebbian learning rule:
$$\mathbf{E}_{\text{graft}}(t) = 0.90 \mathbf{E}_{\text{graft}}(t - 1) + \mathbf{r}_{\text{post}}(t) \odot \mathbf{r}_{\text{pre}}(t)$$
$$\Delta \mathbf{W}_{\text{graft}} = \eta_{\text{heb}} \cdot \text{clamp}(\delta_{\text{TD}}, -2, +2) \cdot \left(\mathbf{E}_{\text{graft}} - \bar{\mathbf{E}}_{\text{graft}}\right)$$
where $\delta_{\text{TD}}$ acts as a global broadcast neuromodulatory signal (representing volume-transmitted octopamine/dopamine).

### 6.3 Intrinsic Curiosity Bonus (Random Network Distillation)
To facilitate autonomous exploration in sparse-reward environments (e.g., complex mazes and object manipulation), an intrinsic reward bonus $r_{\text{int}}$ is generated using Random Network Distillation (Burda et al., 2018):
$$r_{\text{int}}(s_t) = \min\left(0.2, \, 0.5 \cdot \|\hat{f}_\psi(s_t) - f(s_t)\|_2^2\right)$$
$$r_{\text{total}}(s_t) = r_{\text{ext}}(s_t) + \beta_{\text{int}} r_{\text{int}}(s_t)$$
where $f$ is a fixed, randomly initialized target network, $\hat{f}_\psi$ is an online predictor network trained via mean-squared error, and $\beta_{\text{int}} = 0.05$.

---

## 7. Experimental Setup and Cognitive Benchmark Suite

The agent is evaluated within a custom continuous-physics 3D simulation environment featuring planar rigid-body translation, vertical object manipulation, and ray-intersection acoustics/optics. The test suite comprises nine challenging tasks:

```
+-----------------------------------------------------------------------------------------+
|                                    GRAFTBRAIN TASK SUITE                                |
+-----------------------+----------------------+--------------------+---------------------+
| Task Name             | Domain               | Description        | Success Criterion   |
+-----------------------+----------------------+--------------------+---------------------+
| 1. forage             | Spatial Navigation   | Open-field search  | dist(agent, goal)   |
|                       |                      | for random target  | < 0.55 m            |
| 2. maze               | Spatial Navigation   | Random DFS maze    | dist(agent, exit)   |
|                       |                      | traversal (6x6)    | < 0.60 m            |
| 3. push               | Object Manipulation  | Push heavy block   | dist(box, pad)      |
|                       |                      | to target pad      | < 0.35 m            |
| 4. grasp_place        | Fine Motor Control   | Grasp cube, lift,  | Object deposited    |
|                       |                      | deposit into bin   | in target bin       |
| 5. peg_insert         | Fine Motor Control   | Align peg, lift,   | Object inserted     |
|                       |                      | insert into hole   | in target hole      |
| 6. delayed_match      | Working Memory       | Cue -> 20-step     | Correct colored     |
|                       |                      | delay -> choice    | object selected     |
| 7. go_nogo            | Inhibitory Control   | Green: approach    | Reach / withhold    |
|                       |                      | Red: withhold 80st | correctly           |
| 8. button_sequence    | Working Memory       | Observe 3-color    | Press 3 buttons     |
|                       |                      | cue, press order   | in exact sequence   |
| 9. webcam_chase       | Closed-Loop Tracking | Chase orb steered  | Maintain tracking   |
|                       |                      | by live webcam     | proximity < 1.0 m   |
+-----------------------+----------------------+--------------------+---------------------+
```

*Table 2: Benchmark tasks evaluating spatial, motor, and cognitive capabilities.*

### 7.1 Solvability Verification via Scripted Oracle
To guarantee that low success rates reflect learning difficulty rather than kinematic or physical insolvability, a programmatic oracle controller was developed for each task. The oracle was evaluated over 10 random seeds per task ($N = 90$ total episodes). All 9 tasks achieved a **10/10 (100%) completion rate**, confirming physical and kinematic solvability.

---

## 8. Empirical Results and Performance Benchmarks

### 8.1 Computational Throughput and Kernel Vectorization
Recurrent simulation of biological graphs with hundreds of thousands of edges is conventionally bounded by sparse matrix-vector multiplication overhead. We profiled three matrix-vector backends on an 8-thread Intel/AMD CPU executing 4 microsteps per time-step across 12,000 neurons, 252,447 connectome edges, and 95,920 graft edges.

```
+-----------------------------------------------------------------------------------------+
|                        FORWARD + BACKWARD MICROSTEP LATENCY (N=12,000)                  |
+--------------------------------------+---------------------------+----------------------+
| Formulation                          | Latency per Step (ms)     | Speedup Factor       |
+--------------------------------------+---------------------------+----------------------+
| PyTorch Sparse COO (torch.sparse.mm) | 28.29 ms                  | 1.00x (Baseline)     |
| PyTorch Sparse CSR (torch.sparse.mm) | 40.36 ms                  | 0.70x                |
| Vectorized index_add (GraftBrain)    |  7.90 ms                  | 3.58x                |
+--------------------------------------+---------------------------+----------------------+
```

*Table 3: Latency comparison of computational backends for combined forward and backward microsteps.*

By implementing `index_add`, the cumulative time for 100 closed-loop environment steps (including simulation, ray-casting, PPO rollout buffer management, and autograd backward graph evaluation) dropped from **16.85 seconds** to **7.75 seconds**. This achieves an online training throughput of **~14 steps/second on CPU**, allowing consumer hardware to execute continuous closed-loop learning without requiring high-performance computing clusters.

### 8.2 Role of Biological Sub-Circuits in Task Performance
To evaluate the contribution of the integrated biological sub-circuits, ablation experiments were conducted:

```
+-----------------------------------------------------------------------------------------+
|                         ABLATION STUDY ACROSS REPRESENTATIVE TASKS                      |
+------------------------------------+------------------+-----------------+---------------+
| Architecture Configuration         | Maze (Success %) | Match (Success%)| Steps to Conv |
+------------------------------------+------------------+-----------------+---------------+
| Base Connectome + PPO Only         | 18.4%            | 31.2%           | ~45,000       |
| + Homeostatic Regulation           | 24.6%            | 33.5%           | ~38,000       |
| + Central Complex (CX) Compass     | 58.2%            | 34.0%           | ~22,000       |
| + Mushroom Body (MB) Circuit       | 59.1%            | 68.7%           | ~14,000       |
| Full System (+ RND Curiosity)      | 67.4%            | 72.1%           | ~11,000       |
+------------------------------------+------------------+-----------------+---------------+
```

*Table 4: Task completion rates and convergence speed across architectural ablations.*

1. **Central Complex Ring Attractor**: Inclusion of the 16-wedge E-PG compass increased maze exit navigation success from 24.6% to 58.2%. The persistent heading bump acts as an internal directional memory, preventing the agent from executing repetitive circular turns in featureless corridors.
2. **Mushroom Body Associative Memory**: Inclusion of the sparse Kenyon Cell expansion and dopaminergic MBON modulation increased delayed-match success from 34.0% to 68.7%. The sparse top-$k$ representation prevents catastrophic interference between sample cues and subsequent distractor visual stimuli during the 20-step delay period.
3. **Homeostatic Activity Scaling**: Stabilized average network firing rates across extended runs ($>50,000$ steps), preventing the silent-extinction failure mode observed in non-homeostatic recurrent networks.

---

## 9. Discussion and Limitations

### 9.1 Point-Neuron Abstraction vs. Biophysical Spiking
GraftBrain models neural activation using a bounded continuous firing-rate equation rather than multi-compartmental Hodgkin-Huxley or conductance-based Leaky Integrate-and-Fire (LIF) models. While rate equations abstract away spike-timing-dependent plasticity (STDP) and dendritic branch computations, they provide two decisive advantages:
1. Exact differentiability, enabling gradient backpropagation through recurrent connectome dynamics using standard autograd engines.
2. Computational efficiency, enabling closed-loop real-time simulation on commodity processors. Future extensions may integrate event-based surrogate gradient spiking layers (Neftci et al., 2019) to explore sub-millisecond temporal coincidence detection.

### 9.2 Biological Grafting vs. Unconstrained Synthetic Topologies
A frequent question in neuromorphic engineering is whether constraining an artificial agent with biological connectomes offers measurable advantages over unconstrained MLPs or transformers. Our findings indicate that biological connectomes provide an inductive bias: heavy-tailed degree distributions, modular neuropil clustering, and separation of sensory, central, and motor roles create a reservoir that resists catastrophic forgetting and stabilizes recurrent dynamics.

---

## 10. Conclusion

We have introduced **GraftBrain**, a unified, closed-loop platform that fuses eight premier *Drosophila* connectome datasets into an embodied, learning agent. By combining multi-site structural grafting with biological circuit modules—specifically the Central Complex ring attractor compass and Mushroom Body associative learning—GraftBrain bridges the divide between static neuroanatomical reconstructions and adaptive, goal-directed behavior. Operating in real-time on consumer hardware, GraftBrain provides a scalable, open foundation for investigating connectome-constrained computation, biological spatial cognition, and biologically grounded reinforcement learning.

---

## References

* Aso, Y., et al. (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife*, 3, e04577.
* Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *arXiv preprint arXiv:1810.12894*.
* Cognigni, P., Felsenberg, J., & Waddell, S. (2018). Do the right thing: neural network mechanisms of memory formation, expression and update in *Drosophila*. *Current Opinion in Neurobiology*, 52, 18–29.
* Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*, 634(8032), 124–138.
* Hulse, B. K., et al. (2021). A connectome of the *Drosophila* central complex reveals network motifs suitable for flexible navigation and goal-directed behavior. *eLife*, 10, e66039.
* Kim, S. S., Rouault, H., Druckmann, S., & Jayaraman, V. (2017). Ring attractor dynamics in the *Drosophila* central brain. *Science*, 356(6340), 849–853.
* Lappalainen, J. K., et al. (2024). Connectome-constrained networks predict neural activity in the *Drosophila* visual system. *Nature*, 634(8032), 166–174.
* LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
* Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H., & Abbott, L. F. (2017). Optimal degrees of synaptic connectivity for mushroom body function. *Neuron*, 93(5), 1153–1164.
* Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to neuromorphic hardware. *IEEE Signal Processing Magazine*, 36(6), 51–63.
* Scheffer, L. K., et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife*, 9, e57443.
* Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome mapping of *Drosophila*. *Nature*, 634(8032), 139–152.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
* Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
* Seelig, J. D., & Jayaraman, V. (2015). Neural dynamics for landmark orientation and angular path integration. *Nature*, 521(7551), 186–191.
* Shiu, P. K., et al. (2024). A leaky integrate-and-fire computational model of the adult *Drosophila* brain. *Nature*, 634(8032), 153–165.
* Takemura, S., et al. (2023). A connectome of the male *Drosophila* ventral nerve cord. *bioRxiv*, 2023-06.
* Turner-Evans, D. B., et al. (2020). The neuroanatomical ultrastructure and function of a heading direction circuit in *Drosophila*. *Neuron*, 108(1), 145–163.
* Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
