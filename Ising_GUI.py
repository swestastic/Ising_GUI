import tkinter as tk
import numpy as np
import random

from numba import jit, prange

from PIL import Image, ImageTk

scale = 8 # scaling factor for display
update_delay = 5 # milliseconds between updates

# parameters
L = 50 # lattice size (LxL)
T = 2.27 # temperature
J = 1 # coupling constant
Acceptance = 0 # initialize acceptance counter

# initialize spins randomly
spins = np.random.choice([-1, 1], size=(L, L))

# define a sweep function
@jit(nopython=True)
def sweep(spins,T,J,Acceptance):
    # Metropolis single spin flip algorithm. We first pick a random site (x,y) and then calculate the change 
    # in energy if we were to flip it (up->down or down->up). We then draw a number to see if the move is accepted.
    # If it is, then we update the value in the lattice and update the energy, magnetization, and acceptances. 
    flipped_sites = []
    L = spins.shape[0]
    for j in prange(L**2):
        x=np.random.randint(L) #get a random position to update in the lattice
        y=np.random.randint(L)

        dE = 2*J*spins[x,y]*(spins[(x-1)%L,y]+spins[(x+1)%L,y]+spins[x,(y-1)%L]+spins[x,(y+1)%L])

        if np.random.random() < np.exp(-dE/T):# Incrementing the energy and magnetization if the move is accepted
            spins[x,y]*=-1 # update the value in the lattice
            Acceptance += 1 # increment acceptance counter
            flipped_sites.append((x,y))

    return spins,Acceptance,flipped_sites

def spins_to_image(spins):
    L = spins.shape[0]
    # Create an RGB image: white = +1, black = -1
    rgb_array = np.zeros((L, L, 3), dtype=np.uint8)
    rgb_array[spins == 1] = [255, 255, 255]  # white
    rgb_array[spins == -1] = [0, 0, 0]       # black
    img = Image.fromarray(rgb_array, mode='RGB')
    img = img.resize((L * scale, L * scale), resample=Image.NEAREST)
    return img

def run_simulation():
    global spins, T, J, Acceptance, label_img, label
    spins, Acceptance, flipped_sites = sweep(spins, T, J, Acceptance)

    pil_img = spins_to_image(spins)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    root.after(5, run_simulation)

root = tk.Tk()
root.title("Ising Model GUI")

img = tk.PhotoImage(width=L, height=L)
label = tk.Label(root, image=img)
label.pack()

temp_slider = tk.Scale(root, from_=0.1, to=5.0, resolution=0.01, orient=tk.HORIZONTAL, label="Temperature T")
temp_slider.set(T)
temp_slider.pack()

def update_temp(val):
    global T
    T = float(val)
temp_slider.config(command=update_temp)

coupling_slider = tk.Scale(root, from_=-2.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, label="Coupling J")
coupling_slider.set(J)
coupling_slider.pack()

def update_coupling(val):
    global J
    J = float(val)
coupling_slider.config(command=update_coupling)

root.after(5, run_simulation)
root.mainloop()