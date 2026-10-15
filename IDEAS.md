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


Anderson stuff
1. Calculate the Participation Ratio (PR) of Eigenmodes
2. Plot the Local Density of States (LDOS) - In quantum mechanics and acoustic metamaterials, the Local Density of States (LDOS) maps how many vibrational modes are available at a specific node \(i\) at a specific frequency \(\omega \).
• The Phenomenon: When you introduce your log-normal distribution for the springs and bistable truss lengths, you shatter the flat, uniform spectrum of the material into highly fragmented, localized energy landscapes.
• The Proof: By plotting a 2D colormap of Node ID vs. Frequency (\(\omega \)), your disordered lattice will exhibit highly localized "islands" of energy spikes. This proves that your 10 driven nodes are feeding data into a material that natively partitions frequencies into separate spatial zones. This acts as a physical spectrum analyzer, performing a real-time Fourier transform via structural geometry before your linear regression even reads the output.

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




1. Quantify the "Edge of Chaos" via the Maximal Lyapunov Exponent (MLE)

A physical reservoir computes best at the Edge of Chaos—the exact boundary between rigid stability (where the material doesn't change) and chaotic turbulence (where the material forgets everything instantly).
• How to prove it: Run two parallel simulations with identical log-normal disorder profiles. Start them with infinitesimally different initial input vectors (Δ x₀ = 10⁻⁶). Track how the Euclidean distance between the two 100-node state trajectories (\(\Delta x_t\)) diverges or converges over time. Calculate the Lyapunov Exponent: \(\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\Vert{}\Delta x_t\Vert{}}{\Vert{}\Delta x_0\Vert{}}\).
• The Result: Plot the MLE (λ) against your Log-Normal disorder strength (σ).
	• Ordered lattice: λ will be heavily negative (strongly stable, low computational flexibility).
	• Hyper-chaotic lattice: λ will be positive (chaotic, breaks the Echo State Property).
	• Optimized lattice: λ will hover exactly near 0 (the edge of chaos). Showing that your Bayesian Optimizer explicitly picks parameters that drive the MLE to zero is absolute gold-standard proof that disorder tunes the material's phase space for maximum information processing.





2. Map the Attractor Reconstructed Phase-Space Rank

Because you are feeding your system a NARMA signal and using linear regression to predict a future step, you are performing a physical version of Takens' Embedding Theorem. Your 100-node disordered lattice is taking a 1D time-series input and reconstructing its high-dimensional chaotic attractor inside the material's physical folds.
• How to prove it: Extract the 100-node output trajectory matrix during a NARMA run. Run a Principal Component Analysis (PCA) on this state space. Calculate the number of principal components required to explain 99% of the variance.
• The Result: An ordered lattice will show that just 2 or 3 principal components explain all the motion because the waves move uniformly. Your optimized, localized disordered lattice will require a vastly higher number of principal components to explain the variance. This proves that the disorder physically reconstructs a much cleaner, uncollapsed phase space of the NARMA attractor, which is exactly why the linear regression can effortlessly predict the future step.




3. Track Information Bottleneck and Mutual Information Decay

Reviewers love information theory. You can track exactly how information flows from your 10 driven nodes down into the 90 reservoir nodes by calculating Time-Delayed Mutual Information (MI) between the input signal U(t) and the individual state vectors of your nodes \(X_i(t+\tau)\).
• How to prove it: Calculate \(I(U(t); X_i(t+\tau))\) for various time delays τ across every node in the lattice.
• The Result:
	• Ordered Lattice: Mutual information drops off a cliff almost instantly across all nodes due to global scattering and constructive interference wiping the memory clean.
	• Disordered Lattice: The mutual information decay profile will show a power-law decay, with specific node clusters holding onto high mutual information for long delays (τ > 5). This provides rigorous information-theoretic proof that wave localization physically traps the Shannon information inside the spatial bounds of the material, creating local hardware storage registers.





Why it matters to you: This study is one of the very few that explicitly isolates vibration mode localization as a design methodology for Physical Reservoir Computing. They mathematically analyze how trapping structural waves increases features, which provides the exact peer-reviewed foundation you need to back up your claim that wave localization directly upgrades computation. - https://www.nms.me.kyoto-u.ac.jp/en/mode-localization-reservoir-computing/







styling papers

Paper A: “Visualizing the Loss Landscape of Neural Nets” (Li et al.)

• The Style: Deep Blue & Copper Metamaterial/Contour aesthetic.
• Why it's perfect for you: This paper is famous for making high-dimensional vector spaces look incredibly clean. They use highly polished 3D surface maps and 2D contour lines to track states. It uses a minimal layout with sharp contrast, using deep blues to represent low error and rich copper/reds to show high error.

Paper B: “Embodying Intelligence into Mechanical Metamaterials via Reservoir Computing” (He, Musgrave, et al., May 2026)

• The Style: Modern Engineering Blueprint & Flat Vector aesthetic.
• Why it's perfect for you: This paper is in your exact domain. Look at how they structure their figures: they combine the actual CAD/physics diagram of the metamaterial right next to flat, clear line graphs. The font in the plots matches the font of the paper's text exactly, and they stick to a strictly restricted, desaturated color palette (Slate Gray, Soft Teal, muted Red).












Yes, UMAP (Uniform Manifold Approximation and Projection) is a phenomenal tool for your work, and it will provide your paper with one of its most visually striking and scientifically deep figures.
In your 100-node lattice, your reservoir state matrix is 100-dimensional (100 distinct node displacements or velocities changing at every time step). Because humans cannot visualize a 100-dimensional space, UMAP's job is to crush those 100 dimensions down to a 2D or 3D coordinate space while preserving the complex, non-linear relationships between those states.
Here is exactly how you can use UMAP to provide definitive proof of your system's capabilities:

1. Visualize the "Attractor Reconstruction" (The Smoking Gun Proof)

Since you are driving your lattice with a NARMA signal, the input sequence is tracing a specific chaotic trajectory (an attractor) through time.
• How to use UMAP: Collect your 100 × N state matrix (where N is the number of time steps in your simulation). Pass this matrix into UMAP to project it down to 2D (X and Y). Color-code the plotted points by the true future NARMA value.
• What it proves:
	• Ordered Lattice UMAP: The projected points will look like a messy, tangled ball of yarn. This proves that the ordered lattice collapses the input data, making it impossible for a linear regression model to separate different future states.
	• Disordered Lattice UMAP: The points will untangle into a beautiful, smoothly unrolled geometric shape (the reconstructed attractor) where the colors transition perfectly from one side to the other. This visualizes exactly why your linear regression works so well—the disorder has successfully unrolled the high-dimensional phase space, making the future states linearly separable.

2. Map the "Topology of Energy Pockets" (Multistability)

You can use UMAP to map the structural configurations of your bistable von Mises units.
• How to use UMAP: Run your simulation across thousands of random input pulses, and save the final "settled" binary-like state of your 100 nodes (which trusses snapped up, and which snapped down). Pass these final state vectors into UMAP.
• What it proves: The UMAP plot will show distinct, isolated clusters of points. Each cluster represents a specific "energy pocket" or macro-stable memory configuration. You can use this to visually prove that your disordered lattice creates a vast, rich archipelago of unique memory clusters, whereas the ordered lattice only shows a couple of flat, uninteresting dots.

3. Track the "Trajectory of Information" Over Time

Instead of plotting static points, you can plot the UMAP coordinates sequentially and connect them with a line to show the path your material takes as a wave travels through it.
• How to use UMAP: Trace a single 50-step simulation run.
• What it proves: You can show that when a new NARMA input hit the 10 driven nodes, the system's state trajectory swiftly branches into a specific region of the UMAP space. This proves the Echo State Property visually: it shows how the material dynamically transitions between states based on temporal context without getting permanently locked in a single configuration.













PRESENTATION

Use esn equation

simplied for actual
cite it using name et al and publication and year

then talk about actual real ones

rnn
echo state reservoir
physical reservoir




Ideas more Chloe
Basically the bistable energy landscape but do it to the entire structure as multiple are conncted
Also for the spring displacment heat plot what if we also have another plot left of it that colors based of the strength of the spring so we can see that it gets forced into the localized due to that.