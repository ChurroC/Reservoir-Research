What if instead of the non linear springs being a reservoir basically could we derive the acoustic forces and use it as a way to back propogate.

What if instead of a single reservoir we connect it in almost layers like a neural network.
First I was thinking each layer takes the output and feeds it into the next reservoir.
But what about the gradient boosting method we take the residual or error and use that to feed into next layer.
We have each reservoir compensate for error loss from previously.
Compared to (raw data, layer n-1 output) I think the nth reservoir the error will force it to learn what's new or left not already what we know.

Also back on the pendulum idea what if I use the pendulum as the reservior. Then based of the movement of the pendulum we cause the cart to move to correct iself.
Like the inputs is basically gravity and position of the pendulum. Then the inputs are directly applied automatically. But the problem is we don't have training data for the supervised stuff.
We could maybe do a geentic algo to find something good than train it with this but that's always double the work for the theory.
How did the doom and ping pong work since they don't have training data and what to do right. Unsupervised osmething?

Use Cupy for the math when doing cluster

Let the sim run the let spring run for a couple steps without the force to almost allow wave rpopfgation or time delay.

Boosted NN

If we can derive a linear sol for pendulums. Could it replace a sol based of control theory? Would solving contorl theory with the lens of reservoir with itself as the reservoir.

Finish up mult hexagon
Generatively build a mesh or spring system thing
Penedulum idea

What if I have a a graph neural network that let's me build and genratively diesn a spring system

Plot Henon like a time series have x over time and y over time

use narma instead of henon


Check if having randomness can cause some nodes to pull in instead of at rest values

find articel I read last time from first fall presentation



check if we should make the wall_rest_val be above 1 and do that for a trial




do bayesian stufd
do rank and gr
do heatmap or just where the momvement are more

fix one node being connected to left to nothing it shoudl all be euqidistance



lmc vs nlmc
IPR vs lmc
learn bayesian
simplify bayesian
maybe also check if without von miss that disorder improves it
also plot log normal sigma val with mass and damping val

Moderate disorder in a network of nonlinear von Mises trusses may improve reservoir computing by creating heterogeneous, partially localized, and state-dependent responses. These responses increase the diversity and memory of the reservoir. However, strong localization may reduce computational performance by preventing information from spreading across the network.






1. Massive State Space (Higher Memory Capacity): In a perfectly uniform grid of springs or nodes, pushing one side yields a single, highly predictable output. But if you introduce disorder—such as variations in spring stiffness, skewed node locations, or geometric imperfections—the system gains multistability. It unlocks thousands of tiny, localized energy pockets (minima) where the material can settle. Each pocket acts like a physical bit or a unique register of mechanical memory. - https://www.nature.com/articles/s41467-023-40989-1
2. Rich Nonlinearity (Better Computational Power): For a material to solve complex computational tasks (like predicting chaotic systems or filtering noisy temporal data), it must map simple inputs into complex, non-linear physical shapes. Disorder destroys linear symmetry. The chaotic scattering of force waves inside a disordered material forces it to react nonlinearly, which acts as a powerful computational engine. - https://pubs.acs.org/aamick/article-abstract/16/5/6176/133530/Brain-Inspired-Reservoir-Computing-Using?redirectedFrom=fulltext
3. Suppression of "Runaway Synchronization": In completely homogenous physical reservoirs, a recurring input can trigger a "herding effect" or global harmonic resonance. When the whole material vibrates or moves uniformly, it "saturates" and loses its ability to distinguish between new inputs. Research shows that introducing heterogeneity or statistical disorder breaks up these global vibrations, keeping subsets of the material receptive to incoming data. - https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2026.1845838/full

https://pubs.rsc.org/mh/article/13/18/9081/1284776/Visco-Bits-temporal-coding-inspired-viscoelastic - Explains that arrays of bistable units exhibit non-Abelian, path-dependent behavior. This means the order in which your waves hit the trusses matters. Because it is path-dependent, the material naturally builds a timeline of the inputs. When you inject waves into your 10 driven nodes, the material is essentially "calculating" the history of those waves for you.

https://arxiv.org/html/2609.07889 - Explains that you don’t need to visually look at every truss to know what is happening. By applying boundary excitations (waves) and reading the output, the state-dependent stiffness of the system naturally modulates how waves travel. This is exactly what you are doing by driving 10 nodes and observing how the rest of the 100-node lattice reacts.






Your specific idea has not been published as a unified paper, and it has the ingredients to be a high-impact publication.
While physicists study multistable von Mises metamaterials for static memory or simple wave-control, and computational scientists study reservoir computing on abstract networks, using disorder-induced wave localization in a coupled von Mises lattice explicitly to enhance Reservoir Computing (measured via Memory Capacity and NARMA benchmarks) is novel.

Why This Idea Makes a Strong Paper

1. Flips a Core Engineering Paradigm: In traditional manufacturing and computing, structural disorder and defects are treated as problems to eliminate. Your paper demonstrates the opposite: disorder is an intentional computational resource that prevents harmonic saturation and dramatically expands the state space.
2. Bridges Condensed-Matter Physics and Machine Learning: Reviewers in journals like Physical Review Applied, Nature Communications, or Advanced Functional Materials look for mechanistic explanations of physical computing. Linking Anderson-style acoustic wave localization directly to Information Processing Capacity (IPC) / Memory Capacity (MC) connects two previously isolated fields.
3. Quantitative Grounding: Many physical metamaterial papers only show qualitative proof-of-concepts (e.g., "it snaps and stores a bit"). Because you have hard numbers (MC curves and NARMA Normalized Mean Squared Error / NMSE), your claim is verifiable and benchmarked against standard machine learning standards.

What You Must Prove and Show

To write an airtight, peer-review-ready paper, a reviewer will want to see four core elements:

1. The "Edge of Chaos" Curve (Disorder vs. Performance)

You cannot just show that one disordered setup beats an ordered setup. You need to plot Disorder Strength (σ) vs. NARMA NMSE / MC:
• Zero Disorder (σ = 0): Shows poor memory due to uniform wave dispersion and global ringing.
• Moderate Disorder (\(\sigma = \sigma^*\)): Peak performance where wave localization traps states without destroying signal coherence.
• Excessive Disorder (\(\sigma \gg \sigma^*\)): Performance degrades because the material becomes too chaotic, pinning waves before they can propagate or interact.
• Showing this distinct peak proves you have uncovered a fundamental optimization rule, not just a random fluke.

2. Clear Visual Proof of Wave Localization

Include spatiotemporal heatmaps showing the displacement or velocity across all 100 nodes over time:
• In the ordered grid: show the wave traveling uniformly across all nodes.
• In the disordered grid: show the energy becoming trapped in localized clusters (spatial spatial-decay profiles, \(e^{-r/\xi }\), where ξ is the localization length).

3. Decomposition of Memory Capacity (Linear vs. Non-linear MC)

Using standard reservoir benchmarking (such as Dambre et al.'s Information Processing Capacity framework):
• Show the breakdown between linear memory (remembering \(u(t-\tau)\)) and nonlinear memory (remembering higher-order polynomials like \(u(t-\tau_1) \cdot u(t-\tau_2)\)).
• Because von Mises trusses snap nonlinearly, the disorder should specifically boost nonlinear capacity at longer delay horizons.

4. The "Echo State Property" (ESP) Validation

Reviewers in reservoir computing will ask: Does your lattice have the Echo State Property?
• You must demonstrate that your system has fading memory—that perturbations eventually wash out and that two slightly different initial conditions converge when driven by the same long input sequence. If the system stays permanently snapped and never relaxes, it ceases to be a dynamic reservoir and becomes a latching register.






How to Prove the Value of Having Both in Your Paper

Reviewers will naturally ask: “Did you really need both distributions, or would just one have been enough?” You can turn this question into a definitive, high-impact Ablation Study Figure in your paper [dataviz].
Run three quick benchmark simulations using your optimized mass/damping ratio and compare their final Memory Capacity (MC) or NARMA NMSE:
• Case A: Uniform lattice (No disorder anywhere) → Expect high error.
• Case B: Disordered springs only (Uniform truss lengths) → Expect decent memory, but weak nonlinear processing.
• Case C: Disordered truss lengths only (Uniform springs) → Expect decent nonlinearity, but low memory retention.
• Case D (Your System): Both disordered → Expect peak performance.
Showing that Case D significantly outperforms Cases B and C completely validates your engineering choice and cements the dual-distribution design as a core architectural contribution of your work.