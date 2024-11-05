# MC-cluster-algorithms
Efficient implementation of Monte Carlo Cluster Algorithms in Python.
In ising.py the Metropolis, Wolff and Swedsen-Wang (SW) algorithms are implemented.

In the animation below you can see the different Ising model algorithms near critical temperature. The Wolff and SW algortihms don't suffer from critical slowing down (An issue of the Metropolis Algorithm). This was seen by the much longer autocorrelation times of the Metropolis algorithms.

In simpler terms the Wolff and SW algortihms are good at changing the current configuration of the system. They generate different thermodynamic configurations much more efficiently as they traverse a larger portion of the state space.

![alt text](https://github.com/ranjit002/MCCA/blob/main/imgs/Ising_Model_crit_temp.mov)

