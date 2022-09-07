# "Deep Learning Applications on Gravitational Waves"

**MSc "Digital Media & Computational Intelligence", Department of Informatics, Aristotle University of Thessaloniki, September 2021**

Supervisor: Anastasios Tefas, *In collaboration with Virgo group at AUTh*


This is the code of the experiments during my master Thesis. 
This is the final step during the process of the creation of a surrogate model, built upon waveforms from [SEOBNRv4](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.044028/) model. During the series of experiments that were tried out, this is the idea where we divided the range of one of the input parameters in order to obtain better results. Two seperate trainings took place, one for the amplitude and one for the phase components of the complex waveform. The final errors measurement takes place, in the script where we combine our reconstructed waveforms from the models' predictions and obtain the final mismatch.

Inline equation: $equation$
Display equation: $$equation$$


**1) Creating a Surrogate Model**

- A training of $N$ waveforms was created, using SEOBNRv4 with PyCBC,  $\{ h_i(t; \lambda_i) \}_{i=1}^{N}$ where $\lambda = (q, x_1,x_2)$, $q=\frac{m1}{m2}$ is the mass   ratio $1\leq q \leq 8$,  $-1\leq x_1 \leq 1$ and  $-1\leq x_2 \leq 1$ are the spins.
- Greedy algorithm selects $m < N$ waveforms (and their $\lambda$ values), which create the reduced basis $\{e_i\}_{i=1}^{m}$ for a given tolerance.
- the EIM algorithm finds informative time points (empirical nodes $\{ T_i \}_{i=1}^{m}$) that can be used to reconstruct the whole waveform for arbitrary $\lambda$.

**2) Implementing Neural Networks**

- Following [Khan & Green](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.064015/), we use neural networks to map the input $\lambda$ to the coefficients from the empirical nodes $T_j$.  
- After testing to work with complex form of waveforms, two separate networks are used; one for the amplitude and one for the phase of the waveforms, for better results.
