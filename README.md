# NLP
Design a novel Neuron-like Program (NLP) with dynamic synaptic plasticity and low computational overhead, which has the basic functions of biological neurons and provides core technical support for the deployment of deep large-scale SNNs and the implementation of brain like intelligence.
# Dependency
The major dependencies are list as below
 
 Name                               Version

numpy                                1.26.4

pytorch                               2.3.0
# Operating instructions
First, ensure that your PyTorch and NumPy versions meet the requirements.

Second, download the NLP code.

Third, run Main_initialization.py for initialization first, and then run Main_test.py.

Fourth, simply export the generated .csv data files:

1.MemPotential_data.csv

2.OutSpikes_data.csv

3.w_dict_data_0.csv

4.w_ELTPD_data.csv

5.w_LLTPD_data.csv

6.delta_E_LTP_data.csv

7.delta_E_LTD_data.csv

8.delta_L_LTP_data.csv

9.C_star_data.csv

10.delta_L_LTD_data.csv

11.N_star_data.csv

12.Ca_data.csv

# Neuron Model Establishment

Long-term potentiation (LTP) and long-term depression (LTD) are core forms of synaptic plasticity and key neural mechanisms for learning and memory. Based on biological
neurons, we construct the neuron model shown below(see Fig. 1). Processes such as phosphorylation and dephosphorylation of AMPA receptors and protein synthesis are optimized, with only the basic functions of signal transmission and synaptic update of biological neurons retained. According to the calcium ion concentration, synaptic update exhibits three states:LTP, LTD, and No Response. LTP can be divided into an early phase Early-LTP(E-LTP) and a late phase Late-LTP(L-LTP) based on the C** concentration, while LTD can be divided into an early phase Early-LTD(E-LTD) and a late phase Late-LTD(L-LTD) based on the N* concentration.

<img width="1512" height="848" alt="image" src="https://github.com/user-attachments/assets/716805f4-0e12-4c42-88ee-8e054dfb6fb6" /> 
  
                                                               Fig. 1
  The core factors of synaptic weight update are Ca2+ concentration ([Ca2+]), C∗∗ concentration ([C∗∗]), and N∗
concentration([N∗]). E-LTP and E-LTD induction are mainly related to Ca2+ concentration, L-LTP induction is mainly related to C∗∗ concentration, and L-LTD induction
is mainly related to N∗ concentration.

  When the neurotransmitters received by the postsynaptic neuron are sufficient to bring the membrane potential to the threshold, depolarization is initiated
and the intracellular Ca2+ concentration increases. Different Ca2+ concentrations induce three distinct synaptic update pathways.

  If [Ca2+] < 0.01 (biologically 50 - 100nmol/L), this corresponds to the resting basal level, with no firing response and no synaptic update.
  
  If 0.01 ≤ [Ca2+] < 1 (biologically 100 - 300nmol/L), LTD is induced in this concentration range. Each activation generates an increment of active CaN
denoted as ∆N∗. N∗ decays rapidly over time. When the accumulated concentration of N∗([N∗]) is below the threshold, i.e., [N∗] < [N∗]th, the E-LTD process
is activated. When the accumulated concentration of N∗ reaches or exceeds the threshold, i.e., [N∗] ≥ [N∗]th, the L-LTD process is triggered.

  If the calcium concentration [Ca2+] ≥ 1 (biologically 300 - 1000nmol/L), LTP is induced in this range. Each activation produces an increment of calcium-independent active CaMKII denoted as ∆C**. The magnitude of ∆C** is related to the inter-spike interval (biologically, an interval of approximately 10min is
optimal). C∗∗ decays slowly over time. When the accumulated concentration of C∗∗ ([C∗∗]) is below the threshold, i.e., [C∗∗] < [C∗∗]th, the E-LTP process is
activated. When the accumulated concentration of C∗∗ reaches or exceeds the threshold, i.e., [C∗∗] ≥ [C∗∗]th, the L-LTP process is initiated.

# Basic Functions of the Model

1. Leaky Integrate-and-Fire Function

When a neuron receives input signals, its membrane potential increases. Meanwhile, the membrane potential decays over time due to leakage. When the accumulated membrane potential reaches the threshold, the neuron fires and generates a spike, and the membrane potential is reset to 0 (see Fig. 2a).

<img width="1281" height="431" alt="image" src="https://github.com/user-attachments/assets/47c25ec6-a6e6-4983-acb2-a845cb991709" />
 
                                                               Fig. 2
2. Synaptic Weight Self-Update

Synaptic weights are updated according to the timing and frequency of neuronal firing, depending on the firing state of neurons, which is a progressive process. Synaptic weight update depends on the combined effect of early-phase LTP/LTD(ELTPD) and late-phase LTP/LTD (LLTPD)(see Fig. 2b).ELTPD is determined by the increment of E-LTP (∆E-LTP) and the increment of E-LTD (∆E-LTD). LLTPD is determined by the increment of L-LTP (∆L-LTP) and the increment of L-LTD (∆L-LTD)

3. Attention Mechanism

The attention function s can adjust the connection weights of local neurons. By regulating the weights, the signal transmission is either enhanced or weakened, which in turn affects synaptic weight updates. Even after attention fades, the previously modified synaptic weights are retained. Notably, this regulation is short-term and dynamic. When attention changes, the modulation effect on synaptic weights also changes.

4. Forgetting Function

In synaptic weight update, the decay process is fitted to the forgetting curve, and a synaptic weight model with forgetting characteristics is established.

# Neuron Connection Model

The model includes Neuron 1 (subscript:_1) and Neuron 2 (subscript: _2), which are connected via synapses (see Fig. 3). Input spikes are injected into Neuron 1. When the membrane potential of Neuron 1 reaches the threshold, depolarization and firing occur. The signal released by Neuron 1 is transmitted to Neuron 2 through the synapse. When the membrane potential of Neuron 2 reaches the threshold, Neuron 2 fires and outputs a signal.

<img width="1070" height="565" alt="image" src="https://github.com/user-attachments/assets/194ef541-9978-4b6f-8b61-1ad5bde556e7" />
 
                                                               Fig. 3
# Continuous Input Signal
1. Effect of Attention Function s

Variations in the membrane potential, firing frequency, synaptic weight (early LTP & LTD, late LTP & LTD), calcium concentration [Ca2+], C** concentration [C**], and N* concentration [N*] of Neuron2 under different attention functions are shown in Fig. 4 (see Fig. 4a, 4b and 4c). During this process, Neuron 1 maintains a fixed firing frequency and signal intensity, from which the conclusions can be drawn as follows:
<img width="1711" height="773" alt="image" src="https://github.com/user-attachments/assets/03d5da73-a2bf-49b3-bbb2-80708417d4e4" />
<img width="1695" height="586" alt="image" src="https://github.com/user-attachments/assets/15fbdde6-3dbb-46da-b151-2ce6e888df3d" />
<img width="1666" height="333" alt="image" src="https://github.com/user-attachments/assets/7d27302a-3ffd-4824-9942-1f6767246ce2" />

                                                               Fig. 4
(I). Neuron 2 receives the signal (neurotransmitter) from Neuron 1 through the synapse, and the membrane potential of Neuron 2 increases. Meanwhile, the membrane potential changes dynamically due to the leakage of the cell itself. When s is small (s=1), the membrane potential of Neuron 2 cannot reach the threshold, depolarization cannot be induced, and Neuron 2 never fires. When s increases moderately (s=3), after 2 - 3 signal accumulations, the membrane potential of Neuron 2 can reach the threshold and induce depolarization, and Neuron 2 fires. However, due to the low firing frequency, a high Ca2+ concentration cannot be accumulated, and LTD (E-LTD) is actually induced, resulting in a decrease in synaptic weight. When s is large (s=6), each firing of Neuron 1 can induce depolarization of Neuron 2, which can accumulate a high Ca2+ concentration and induce LTP (E-LTP), resulting in an increase in synaptic weight.

(II). The Ca2+ concentration is induced by the depolarization of Neuron 2 and decays at a certain rate. Therefore, when the depolarization amplitude of Neuron 2 is small and the period is longer than the decay period of Ca2+ concentration (s=3), the Ca2+ accumulation is low ([Ca2+] < 1), inducing LTD and thus reducing the synaptic weight. Only when the depolarization amplitude of Neuron 2 is large and the period is shorter than the decay period of Ca2+ concentration (s=6) can an effective high Ca2+ accumulation ([Ca2+] ≥ 1) be formed, inducing LTP and thus increasing the synaptic weight. LTP and LTD are dynamic processes, and the final change in synaptic weight is the combined result of both. In addition, the Ca2+ concentration is also the inducement for changes in C∗∗ and N∗.

(III). The accumulation of C∗∗ is related to the firing state of neurons (Ca2+ concentration) and the firing interval. C∗∗ is the key factor for inducing L-LTP.

(IV). N∗ is induced when neurons are at a low Ca2+ concentration and serves as the key factor for L-LTD induction. In Figure 4c, [N∗] still increases at s = 6 because there exists a range where the Ca2+ concentration is below 1, which triggers the generation of N∗.

(V). Synaptic weight update:When Neuron 2 does not fire (s = 1), no Ca2+ is generated, and the synaptic weight decays according to the forgetting curve. When Neuron 2 fires with small amplitude and low frequency (s = 3), the Ca2+ concentration accumulates at a low level ([Ca2+] < 1), which induces LTD. Specifically, when [N∗] < [N∗]th ([N∗]th = 0.15), E-LTD is induced; when [N∗] ≥ [N∗]th, L-LTD is induced, and the synaptic weight decreases. When Neuron 2 fires with large depolarization amplitude and high frequency (s = 6), an effective high Ca2+ concentration accumulation (([Ca2+] ≥ 1) is formed, which induces LTP. Specifically, when [C∗∗] < [C∗∗]th ([C∗∗]th = 0.11), E-LTP is induced; when [C∗∗] ≥ [C∗∗]th, L-LTP is induced, and the synaptic weight increases. E-LTP and E-LTD decay rapidly over time, whereas L-LTP and L-LTD decay slowly over time.

2. Leaky Integrate-and-Fire Function

Comparisons are performed under different input spike intervals ∆t_input (see 4b and 4d in Fig. 4), corresponding to different signal frequencies. The initial synaptic weight is 0.2, and the initial membrane potentials of both Neuron 1 and Neuron 2 are 0.

When ∆t_input = 0.1s (see Fig. 4b), the firing period of Neuron 1 is long(∼ 0.3s), the input signal to Neuron 2 is weak, and the firing period of Neuron 2 is long (∼ 1.1s). The Ca2+ accumulation is low ([Ca2+] < 1), and the synaptic weight between Neuron 2 and Neuron 1 decreases over time. The decrease in synaptic weight mainly comes from the reduction of ELTPD, while LLTPD remains almost unchanged. The decrease in ELTPD is due to ∆E-LTP < ∆E-LTD. Since the corresponding conditions are not satisfied ([C∗∗] < [C∗∗]th, [N∗] < [N∗]th), ∆L-LTP and ∆L-LTD are both 0, and LLTPD does not change. The peak value of [Ca2+]_2 of Neuron 2 decreases over time, [N∗]_2 increases over time, and [C∗∗]_2 remains unchanged.

When ∆t_input = 0.02s (see Fig. 4d), the firing period of Neuron 1 is short(∼ 0.06s), the input signal to Neuron 2 is strong, and the firing period of Neuron 2 is short (∼ 0.12s). An effective high Ca2+ accumulation ([Ca2+] ≥ 1) can be formed, the synaptic weight between Neuron 2 and Neuron 1 increases over time, and the firing period of Neuron 2 gradually decreases (0.06 s). The increase in synaptic weight mainly comes from the increase of ELTPD, while LLTPD remains almost unchanged. The increase in ELTPD is due to ∆E-LTP > ∆E-LTD. Since the corresponding conditions are not satisfied ([C∗∗] < [C∗∗]th, [N∗] < [N∗]th), ∆L-LTP and ∆L-LTD are both 0, and LLTPD does not change. The peak value of [Ca2+]_2 of Neuron 2 increases over time, [N∗]_2 first increases and then gradually decreases as the Ca2+ concentration rises, and [C∗∗]_2 remains unchanged.

# Time-spaced Input Signals
Biological neuron firing is a sparse signal; therefore, two sparse firing scenarios (single attention and repeated attention) are simulated to demonstrate the induction conditions of late-phase LTP, as shown in Fig. 5.

The first case simulates the single-attention process (see Fig. 5a): 100 input spikes (s = 1) → 600s interval → 100 input spikes (s = 3) → 600s interval → 100 input spikes (s = 1).

The second case simulates the multiple-attention process (see Fig. 5b): 100 input spikes (s = 1) → 600s interval → 100 input spikes (s = 3) → 600s interval → 100 input spikes (s = 3) → 600s interval → 100 input spikes (s = 3) → 600s interval → 100 input spikes (s = 1). The 600s intervals are used to simulate the induction of LLTPD.

<img width="1757" height="608" alt="image" src="https://github.com/user-attachments/assets/e4a2d787-e928-4659-95f7-4bfd708206fc" />
<img width="1754" height="438" alt="image" src="https://github.com/user-attachments/assets/03cfdae0-7dc4-4be5-91ff-b7217e5fac02" />
<img width="1749" height="441" alt="image" src="https://github.com/user-attachments/assets/8ee9facd-99f0-4189-bab2-b21442369a66" />
<img width="1747" height="440" alt="image" src="https://github.com/user-attachments/assets/0d974893-a771-41d0-8145-f66bde4f314d" />
<img width="1730" height="519" alt="image" src="https://github.com/user-attachments/assets/ef8acd07-981f-42aa-8b66-66cf4d3bbb79" />

                                                               Fig. 5
In the single-attention process, during the attention-enhancement phase (s=3), the increase in synaptic weight mainly comes from the increase in ELTPD, while LLTPD remains almost unchanged. The increase in ELTPD is due to ∆E-LTP > ∆E-LTD. During the subsequent 600s interval, ELTPD decays significantly, and the synaptic weight returns to its original level.

During repeated attention, two additional episodes of 100 input spikes (s = 3) are introduced compared with single attention. Although [C∗∗] decays partially during each 600s interval, [C∗∗] can eventually reach the threshold after multiple accumulations, thus inducing L-LTP. [C∗∗] is elicited in all three attention-enhancing phases (s = 3). However, the accumulated [C∗∗] in the first two phases does not reach the threshold and is insufficient to induce L-LTP. At the interaction between the second and third attention-enhancing phases, the accumulated [C∗∗] reaches the threshold ([C∗∗]_2 in Fig. 5b, [C∗∗]th = 0.1), inducing L-LTP (corresponding to an increasing peak in ∆L-LTP in Fig. 5b). The increase in synaptic weight induced by L-LTP decays slowly and can be maintained for a long time, which forms the basis of long-term memory.

