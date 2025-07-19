# Ising Model GUI

A simple Ising Model app writing in Python using PIL, Tkinter, Numba, and Numpy. Arrays and math are handled with Numpy, which is then provided a significant speedup by Numba. Tkinter creates a popup window in which the simulation runs in with sliders and buttons, and PIL creates the images displayed of the simulation.

![demo](images/demo.gif)

## Background

The Ising model is a simple spin model, where each site on a lattice can take on a single value (-1,+1). It is described by the following Hamiltonian:

```math
H = -J \sum_i\sigma_i\sigma_j + h\sum_i\sigma_i
```

where $J$ is the interaction strength between neighboring sites, $\sigma_i=\pm1$ is the value at site $i$, and $h$ is an external magnetic field applied parallel do the spin axis.

In two dimensions with no external magnetic field ($h=0$), the model exhibits a phase transition at $T_c = \frac{2J}{k \text{ln}(1+\sqrt{2}}) \approx (2.269185...)\frac{J}{k}$ where $k$ is the Boltzmann constant, which is commonly set to $k=1$. For $J>0$, the model is ferromagnetic, and below $T_c$ will converge to a fully-aligned state. For $J<0$, the model is anti-ferromagnetic and will instead converge to a fully anti-aligned state.

This simulation allows for simulation using the Metropolis-Hastings, Wolff, Swendsen-Wang, Glauber, and Kawasaki methods.

The Metropolis-Hastings algorithm is where "flips" are proposed to random sites on the lattice. A "flip" will invert the value on a given site $\sigma_i=\pm1\rightarrow\mp1$.
A flip will either be accepted or rejected based on a Boltzmann probability, $r<e^{-\Delta E/T}$, where $r$ is a random number drawn on $(0,1)$. Decreases in energy are always accepted, and increases in energy have a chance to be accepted.

The Glauber, or "Heat Bath", algorithm is very similar to the Mtropolis-Hastings one, with the exception that the probability acceptance criteria is different. This criteria is $r < \frac{1}{1+e^{\Delta E/T}}$

The Wolff algorithm works by picking a random site and then attempting to build a cluster made up of neighboring sites with the same spin value $\sigma_i$. Additions to the cluster are based on a probability $r < 1-e^{-2J/T}$. This algorithm is rejection-free, so the cluster is flipped every time.

The Swendsen-Wang algorithm is another cluster method, but instead of building only one cluster, it creates a number of clusters until every site in the lattice has been categorized into a cluster. Adding a site to a cluster is based on a probability $r < 1-e^{-2J/T}$. Each individual cluster is then either flipped or not, based on a probability $r<\frac{1}{2}$.

Finally, the Kawasaki interpretation of the Ising model is unique in that spin is conserved. This means that sites cannot be "flipped", but rather a spin-up site and neighboring spin-down site can switch places in the lattice.

## Usage

Simply run `python3 Ising_GUI.py` to open a Tkinter window and run the simulation. The slider bars for $T$ and $J$ are intuitive to use, and the simulation will update automatically in correspondance with them.

## Future Work

- Add external magnetic field functionality and slider

- Add lattice scale and size adjustments

- Add the ability to save data from simulation runs

## Acknowledgements

This work was inspired by [mattbierbaum's ising.js](https://github.com/mattbierbaum/ising.js/). When I was first learning about the Ising model, I thought that it was a very helpful tool for visualizing the behavior of the model. I wanted to take my own attempt at it because of that!
