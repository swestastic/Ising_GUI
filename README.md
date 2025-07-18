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

This simulation using the Metropolis-Hastings algorithm, where "flips" are proposed to random sites on the lattice. A "flip" will invert the value on a given site $\sigma_i=\pm1\rightarrow\mp1$.
A flip will either be accepted or rejected based on a Boltzmann probability, $r<e^{-\Delta E/T}$, where $r$ is a random number drawn on $(0,1)$. Decreases in energy are always accepted, and increases in energy have a chance to be accepted.

## Usage

Simply run `python3 Ising_GUI.py` to open a Tkinter window and run the simulation. The slider bars for $T$ and $J$ are intuitive to use, and the simulation will update automatically in correspondance with them.

## Future Work

- Add external magnetic field functionality and slider

- Add lattice scale and size adjustments

- Add the ability to save data from simulation runs

## Acknowledgements

This work was inspired by [mattbierbaum's ising.js](https://github.com/mattbierbaum/ising.js/). When I was first learning about the Ising model, I thought that it was a very helpful tool for visualizing the behavior of the model. I wanted to take my own attempt at it because of that!
