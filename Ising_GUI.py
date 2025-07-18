# Look at using https://www.pyqtgraph.org/ for plotting instead of matplotlib
import tkinter as tk
from tkinter import ttk
import numpy as np
import random

from numba import jit, prange

from PIL import Image, ImageTk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

scale = 8 # scaling factor for display
simulation_update_delay = 5 # milliseconds between updates
plot_update_delay = 100  # ms
count = 0
plot_observable = "Magnetization"

# parameters
L = 50 # lattice size (LxL)
T = 2.27 # temperature
J = 1 # coupling constant
Acceptance = 0 # initialize acceptance counter
sweepcount = 1

# define functions to calculate energy and magnetization
@jit(nopython=True)
def Energy(spins,J):
  # Calculates the energy of a given lattice configuration. 
  TotalEnergy=0
  side = len(spins)
  for i in prange(side):
    for j in prange(side):
      TotalEnergy+= -J * (spins[i,j] * (spins[(i+1)%side,j] + spins[i,(j+1)%side]))
  return TotalEnergy

@jit(nopython=True)
def Mag(spins): 
  # Calculates the magnetization of a given lattice configuration.
  M = np.sum(spins)
  return M

# define a sweep function
@jit(nopython=True)
def sweep(spins, T, J, Acceptance, E, M, sweepcount):
    # Metropolis single spin flip algorithm. We first pick a random site (x,y) and then calculate the change 
    # in energy if we were to flip it (up->down or down->up). We then draw a number to see if the move is accepted.
    # If it is, then we update the value in the lattice and update the energy, magnetization, and acceptances. 
    flipped_sites = []
    L = spins.shape[0]
    sweepcount += L**2
    for j in prange(L**2):
        x=np.random.randint(L) #get a random position to update in the lattice
        y=np.random.randint(L)

        dE = 2*J*spins[x,y]*(spins[(x-1)%L,y]+spins[(x+1)%L,y]+spins[x,(y-1)%L]+spins[x,(y+1)%L])

        if np.random.random() < np.exp(-dE/T):# Incrementing the energy and magnetization if the move is accepted
            spins[x,y]*=-1 # update the value in the lattice
            Acceptance += 1 # increment acceptance counter
            flipped_sites.append((x,y))
            E += dE
            M += 2*spins[x,y]

    return spins, Acceptance, flipped_sites, E, M, sweepcount

def spins_to_image_init(spins):
    L = spins.shape[0]
    # Create an RGB image: white = +1, black = -1
    rgb_array = np.zeros((L, L, 3), dtype=np.uint8)
    rgb_array[spins == 1] = [255, 255, 255]  # white
    rgb_array[spins == -1] = [0, 0, 0]       # black
    img = Image.fromarray(rgb_array, mode='RGB')
    img = img.resize((L * scale, L * scale), resample=Image.NEAREST)
    return rgb_array

def spins_to_image(spins, flipped_sites, rgb_array):
    L = spins.shape[0]
    # Create an RGB image: white = +1, black = -1
    for x, y in flipped_sites:
        if spins[x, y] == 1:
            rgb_array[x, y] = [255, 255, 255]  # white
        else:
            rgb_array[x, y] = [0, 0, 0]        # black
    img = Image.fromarray(rgb_array, mode='RGB')
    img = img.resize((L * scale, L * scale), resample=Image.NEAREST)
    return img

def reset_for_parameter_change():
    global Acceptance, sweepcount, E, M
    Acceptance = 0
    sweepcount = 1
    E = Energy(spins,J)
    M = Mag(spins)

def update_temp(val):
    global T
    reset_for_parameter_change()
    T = float(val)
    temp_entry.delete(0, tk.END)
    temp_entry.insert(0, f"{T:.2f}")

def update_coupling(val):
    global J
    reset_for_parameter_change()
    J = float(val)
    coupling_entry.delete(0, tk.END)
    coupling_entry.insert(0, f"{J:.2f}")

def update_temp_entry(val):
    reset_for_parameter_change()
    try:
        T_val = float(val)
        if 0.1 <= T_val <= 5.0:
            temp_slider.set(T_val)
    except ValueError:
        pass

def update_coupling_entry(val):
    reset_for_parameter_change()
    try:
        J_val = float(val)
        if -2.0 <= J_val <= 2.0:
            coupling_slider.set(J_val)
    except ValueError:
        pass

def update_plot_choice(event):
    global data_buffer, line, plot_observable, sweepcount
    plot_observable = observable_dropdown.get()
    if plot_observable == "Energy":
        ax.set_ylabel("Energy / (L^2 J)")
        ax.set_ylim(-2, 2)
    elif plot_observable == "Magnetization":
        ax.set_ylabel("Magnetization (M/$L^2$)")
        ax.set_ylim(-1, 1)
    elif plot_observable == "Acceptance":
        ax.set_ylabel("Acceptance")
        ax.set_ylim(0, 1)
    data_buffer.clear()
    line.set_ydata([0]*100)
    ax.set_title(f"Live {plot_observable} Vs. Time")
    canvas.draw()

def run_simulation():
    # This is our main simulation loop, called every few milliseconds by Tkinter's after method. 
    # It performs a sweep of the lattice, updates the image and the plot.
    global spins, T, J, Acceptance, label_img, label, E, M, L, plot_observable, sweepcount
    global count

    spins, Acceptance, flipped_sites, E, M, sweepcount = sweep(spins, T, J, Acceptance, E, M, sweepcount)

    # update the image
    pil_img = spins_to_image(spins, flipped_sites, rgb_array)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    # update the plot after 8 run_simulation calls (~40ms)
    count = (count + 1) % 8
    if count == 0:
        if plot_observable == "Energy":
            data_buffer.append(E / L**2)
        elif plot_observable == "Magnetization":
            data_buffer.append(M / L**2)
        elif plot_observable == "Acceptance":
            data_buffer.append(Acceptance / sweepcount)
        line.set_ydata(list(data_buffer) + [0] * (100 - len(data_buffer)))
        canvas.draw()

    root.after(5, run_simulation)

# initialize spins randomly
spins = np.random.choice([-1, 1], size=(L, L))
E = Energy(spins,J)
M = Mag(spins)

# initialize the RGB image array
rgb_array = spins_to_image_init(spins)

## Set up the GUI
# Create the main window
root = tk.Tk()
root.title("Ising Model GUI")

# Create the image frame and set it to the left side of the window
image_frame = ttk.Frame(root)
image_frame.pack(side=tk.LEFT)

img = tk.PhotoImage(width=L, height=L)
label = ttk.Label(image_frame, image=img)
label.pack()

# Create the slider frame and set it to the right side of the window
slider_frame = ttk.Frame(root)
slider_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(slider_frame)
plot_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Create the matplotlib figure and axis for plotting
plt.style.use('fast')
fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Data buffer for live plot (e.g. tracking magnetization or acceptance)
data_buffer = deque(maxlen=100)
x_vals = list(range(100))
line, = ax.plot(x_vals, [0]*100)
ax.set_ylim(-1, 1)
ax.set_title(f"Live {plot_observable} Vs. Time")
ax.set_xlabel("Time")
ax.set_ylabel(f"{plot_observable}")
fig.tight_layout()

# Create the sliders and add them to the slider frame
temp_label = ttk.Label(slider_frame, text="Temperature (T):")
temp_label.grid(row=0, column=0, padx=5, pady=5)
temp_slider = ttk.Scale(slider_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, value=T)
temp_slider.grid(row=0, column=1, padx=5, pady=5)
temp_slider.config(command=update_temp)
temp_entry = ttk.Entry(slider_frame, width=5)
temp_entry.insert(0, str(T))  # set initial value
temp_entry.bind("<Return>", lambda event: update_temp_entry(temp_entry.get()))
temp_entry.grid(row=0, column=2, padx=5, pady=5)


coupling_label = ttk.Label(slider_frame, text="Coupling (J):")
coupling_label.grid(row=1, column=0, padx=5, pady=5)
coupling_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=J)
coupling_slider.grid(row=1, column=1, padx=5, pady=5)
coupling_slider.config(command=update_coupling)
coupling_entry = ttk.Entry(slider_frame, width=5)
coupling_entry.insert(0, str(J))  # set initial value
coupling_entry.bind("<Return>", lambda event: update_coupling_entry(coupling_entry.get()))
coupling_entry.grid(row=1, column=2, padx=5, pady=5)

observable_label = ttk.Label(slider_frame, text="Observable to Plot:")
observable_label.grid(row=3, column=0, padx=5, pady=5)

observable_dropdown = ttk.Combobox(slider_frame, values=["Magnetization", "Energy", "Acceptance"], state="readonly")
observable_dropdown.current(0)
observable_dropdown.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

observable_dropdown.bind("<<ComboboxSelected>>", update_plot_choice)

values_label = ttk.Label(slider_frame, text="Current Values:")
values_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
acceptance_label = ttk.Label(slider_frame, text=f"Acceptance: {Acceptance/sweepcount:.3f}")
acceptance_label.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

energy_label = ttk.Label(slider_frame, text=f"Energy / (L^2 J): {E / (L**2):.3f}")
energy_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5)
magnetization_label = ttk.Label(slider_frame, text=f"Magnetization (M/L^2): {M / (L**2):.3f}")
magnetization_label.grid(row=7, column=0, columnspan=3, padx=5, pady=5)
def update_observable_labels():
    energy_label.config(text=f"Energy / (L^2 J): {E / (L**2):.3f}")
    magnetization_label.config(text=f"Magnetization (M/L^2): {M / (L**2):.3f}")
    acceptance_label.config(text=f"Acceptance: {Acceptance/sweepcount:.3f}")
    root.after(50, update_observable_labels)


# run the window and simulation
root.after(50, update_observable_labels)
root.after(5, run_simulation)
root.mainloop()