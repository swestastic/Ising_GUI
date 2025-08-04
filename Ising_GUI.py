import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from numba import njit, prange, float64, int32, types
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import argparse
import time

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Ising Model Simulation GUI")
parser.add_argument("--cache", type=bool, default=False, help="Enable caching for faster simulations")
parser.add_argument("--fastmath", type=bool, default=True, help="Enable fast math optimizations")
parser.add_argument("--parallel", type=bool, default=True, help="Enable parallel execution")

# parameters
L = 64 # lattice size (LxL)
T = 2.26918531421 # temperature
J = 1.0 # coupling constant
h = 0.0 # external magnetic field

# initializations
count = 0 # counter for plot updates
Acceptance = 0 # initialize acceptance counter
sweepcount = 1 # initialize sweep counter
sweep_counter = 0 # counter for sweeps per second

scale = 512 // L # scaling factor for display

# Numba settings
FASTMATH = parser.parse_args().fastmath
PARALLEL = parser.parse_args().parallel
CACHE = parser.parse_args().cache

RUN_SIM = False

plot_observable = "Magnetization" # "Magnetization", "Energy", "Acceptance"
algorithm = "Metropolis" # "Metropolis", "Wolff", "Glauber", "Swendsen-Wang", "Kawasaki"

############################# Observable Calculation Functions #############################
# define functions to calculate energy and magnetization
@njit(float64(int32[:,:], float64, float64), 
    parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def Energy(spins,J, h):
  # Calculates the energy of a given lattice configuration. 
  TotalEnergy = 0
  side = len(spins)
  for i in prange(side):
    for j in prange(side):
      TotalEnergy += (spins[i,j] * (spins[(i+1)%side,j] + spins[i,(j+1)%side])) - h * spins[i,j]
  TotalEnergy *= -J
  return TotalEnergy

@njit(float64(int32[:,:]), fastmath=FASTMATH, cache=CACHE)
def Mag(spins): 
  # Calculates the magnetization of a given lattice configuration.
  M = np.sum(spins)
  return M

############################# Monte Carlo Algorithms #############################
@njit(types.Tuple((int32[:, :],int32,int32[:, :],float64,float64,int32))
    (int32[:, :], float64, float64, float64, float64, float64, int32, int32, int32),
    fastmath=FASTMATH, cache=CACHE)
def Metropolis(spins, T, J, h, E, M, L, Acceptance, sweepcount):
    # Metropolis single spin flip algorithm. We first pick a random site (x,y) and then calculate the change 
    # in energy if we were to flip it (up->down or down->up). We then draw a number to see if the move is accepted.
    # If it is, then we update the value in the lattice and update the energy, magnetization, and acceptances. 
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)  # Preallocate for flipped sites
    flip_count = 0
    sweepcount += L**2
    for j in range(L**2):
        x,y = np.random.randint(0,L,2) #get a random position to update in the lattice

        dE = 2 * spins[x, y] * (
            J * (
                spins[(x-1)%L, y] +
                spins[(x+1)%L, y] +
                spins[x, (y-1)%L] +
                spins[x, (y+1)%L]
            ) + h
        )

        if np.random.random() < np.exp(-dE/T):# Incrementing the energy and magnetization if the move is accepted
            spins[x,y]*=-1 # update the value in the lattice
            Acceptance += 1 # increment acceptance counter
            flipped_sites[flip_count] = (x,y)
            E += dE
            M += 2*spins[x,y]
            flip_count += 1

    return spins, Acceptance, flipped_sites[:flip_count], E, M, sweepcount

def flip_sublattice(spins, L):
    # Flips the spins of one sublattice (A or B) in a checkerboard pattern.
    for i in range(L):
        for j in range(L):
            if (i + j) % 2 == 0:  # flip sublattice A
                spins[i, j] *= -1
    return spins

@njit(fastmath=FASTMATH, cache=CACHE)
def Wolff(spins,T,J,L, h):
    attempted=[]
    x,y = np.random.randint(0,L,2)
    cluster = [(x,y)]
    prob = 1-np.exp(-2*abs(J)/T)
    if J < 0:
        spins = flip_sublattice(spins, L) 

    for i,j in cluster: # add nearest neighbors to the cluster if they are within the range
        north = i,(j+1)%L
        south = i,(j-1)%L
        east = (i+1)%L,j
        west = (i-1)%L,j
        neighbors = [north,south,east,west]
        
        for k in neighbors:
            if tuple((i,j,k[0],k[1])) not in attempted:
                attempted.append(tuple((i,j,k[0],k[1])))
                if k not in cluster and spins[i,j]==spins[k] and np.random.rand() < prob:
                    cluster.append(k)
    
    for x,y in cluster:
        spins[x,y] *= -1

    if J < 0:
        spins = flip_sublattice(spins, L)

    ClusterSize = len(cluster)
    return spins, cluster # here cluster = flipped_sites

@njit(fastmath=FASTMATH, cache=CACHE)
def SwendsenWang(spins, T, J, h, L):
    bonds = np.zeros((L, L, 4), dtype=np.uint8)  # 0: up, 1: down, 2: left, 3: right
    p = 1 - np.exp(-2 * abs(J) / T)

    if J < 0:
        spins = flip_sublattice(spins, L)

    # Build bonds
    for i in range(L):
        for j in range(L):
            if spins[i, j] == spins[(i+1)%L, j] and np.random.rand() < p:
                bonds[i, j, 0] = 1  # bond to down
            if spins[i, j] == spins[(i-1)%L, j] and np.random.rand() < p:
                bonds[i, j, 1] = 1  # bond to up
            if spins[i, j] == spins[i, (j-1)%L] and np.random.rand() < p:
                bonds[i, j, 2] = 1  # bond to left
            if spins[i, j] == spins[i, (j+1)%L] and np.random.rand() < p:
                bonds[i, j, 3] = 1  # bond to right

    visited = np.zeros((L, L), dtype=np.uint8)
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)
    flip_count = 0

    for i in range(L):
        for j in range(L):
            if visited[i, j] == 0:
                # Begin new cluster
                cluster = np.zeros((L**2, 2), dtype=np.int32)
                cluster[0, 0], cluster[0, 1] = i, j
                visited[i, j] = 1
                cluster_size = 1
                k = 0
                while k < cluster_size:
                    x, y = cluster[k]
                    neighbors = [((x+1)%L, y, 0),
                                 ((x-1)%L, y, 1),
                                 (x, (y-1)%L, 2),
                                 (x, (y+1)%L, 3)]
                    for a, b, d in neighbors:
                        if bonds[x, y, d] == 1 and visited[a, b] == 0:
                            visited[a, b] = 1
                            cluster[cluster_size, 0] = a
                            cluster[cluster_size, 1] = b
                            cluster_size += 1
                    k += 1
                # Flip with 50% probability
                if np.random.rand() < 0.5:
                    for m in range(cluster_size):
                        x, y = cluster[m]
                        spins[x, y] *= -1
                        flipped_sites[flip_count, 0] = x
                        flipped_sites[flip_count, 1] = y
                        flip_count += 1

    if J < 0:
        spins = flip_sublattice(spins, L)

    return spins, flipped_sites[:flip_count]

@njit(types.Tuple((types.Array(int32, 2, 'C'),types.Array(int32, 2, 'C')))
    (types.Array(int32, 2, 'C'),float64, float64, float64,int32),
    fastmath=FASTMATH, cache=CACHE)
def Kawasaki(spins, T, J, h, L):
    flipped_sites = np.zeros((2*L**2, 2), dtype=np.int32)
    flip_count = 0
    for i in range(L**2):
        x1, y1 = np.random.randint(0,L,2)
        neighbors = [((x1+1)%L,y1),((x1-1)%L,y1),(x1,(y1+1)%L),(x1,(y1-1)%L)]
        x2,y2 = neighbors[np.random.randint(0, 4)]
        if spins[x1,y1] != spins[x2,y2]:
            # calculate energy before swap
            E_before = -J * spins[x1, y1] * (
                            spins[(x1 - 1) % L, y1] +
                            spins[(x1 + 1) % L, y1] +
                            spins[x1, (y1 - 1) % L] +
                            spins[x1, (y1 + 1) % L]) - h * spins[x1, y1]

            E_before += -J * spins[x2, y2] * (
                            spins[(x2 - 1) % L, y2] +
                            spins[(x2 + 1) % L, y2] +
                            spins[x2, (y2 - 1) % L] +
                            spins[x2, (y2 + 1) % L]) - h * spins[x2, y2]

            # swap sites
            spins[x1, y1], spins[x2, y2] = spins[x2, y2], spins[x1, y1]

            # calculate energy after swap
            E_after = -J * spins[x1, y1] * (
                            spins[(x1 - 1) % L, y1] +
                            spins[(x1 + 1) % L, y1] +
                            spins[x1, (y1 - 1) % L] +
                            spins[x1, (y1 + 1) % L]) - h * spins[x1, y1]

            E_after += -J * spins[x2, y2] * (
                            spins[(x2 - 1) % L, y2] +
                            spins[(x2 + 1) % L, y2] +
                            spins[x2, (y2 - 1) % L] +
                            spins[x2, (y2 + 1) % L]) - h * spins[x2, y2]

            dE = E_after - E_before

            if dE <= 0 or np.random.random() < np.exp(-dE/T):
                flipped_sites[flip_count] = (x1,y1)
                flipped_sites[flip_count + 1] = (x2,y2)
                flip_count += 2
            else:
                spins[x1,y1], spins[x2,y2] = spins[x2,y2], spins[x1,y1] # swap back if not accepted
    return spins, flipped_sites[:flip_count]

@njit(types.Tuple((int32[:, :],int32,int32[:, :],float64,float64,int32))
    (int32[:, :], float64, float64, float64, float64, float64, int32, int32, int32),
    fastmath=FASTMATH, cache=CACHE)
def Glauber(spins, T, J, h, E, M, L, Acceptance, sweepcount):
    # Glauber algorithm. We first pick a random site (x,y) and then calculate the change
    # in energy if we were to flip it (up->down or down->up). We then draw a number to see if the move is accepted.
    # If it is, then we update the value in the lattice and update the energy, magnetization, and acceptances. 
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)
    flip_count = 0
    sweepcount += L**2
    for j in range(L**2):
        x,y = np.random.randint(0,L,2) #get a random position to update in the lattice

        dE = 2 * spins[x, y] * (
            J * (
                spins[(x-1)%L, y] +
                spins[(x+1)%L, y] +
                spins[x, (y-1)%L] +
                spins[x, (y+1)%L]
            ) + h
        )

        if np.random.random() < 1/(1+np.exp(dE/T)):# Incrementing the energy and magnetization if the move is accepted
            spins[x,y]*=-1 # update the value in the lattice
            Acceptance += 1 # increment acceptance counter
            flipped_sites[flip_count] = (x,y)
            flip_count += 1
            E += dE
            M += 2*spins[x,y]

    return spins, Acceptance, flipped_sites[:flip_count], E, M, sweepcount

@njit(types.Tuple((int32[:, :],int32[:, :]))
    (int32[:, :],float64,float64,float64,int32),
    fastmath=FASTMATH,cache=CACHE)
def HeatBath(spins, T, J, h, L):
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)  # Preallocate for flipped sites
    flip_count = 0

    for i in range(L**2):
        x,y = np.random.randint(0,L,2)

        # Periodic boundary neighbors
        neighbors = (
            spins[(x+1)%L, y] + spins[(x-1)%L, y] +
            spins[x, (y+1)%L] + spins[x, (y-1)%L]
        )

        # Energies for spin up/down including magnetic field
        E_up = -J * neighbors - h
        E_down = J * neighbors + h

        p_up = np.exp(-E_up/T)
        p_down = np.exp(-E_down/T)
        prob_spin_up = p_up / (p_up + p_down)

        # Update spin based on computed probability
        backup = spins[x, y]
        if np.random.rand() < prob_spin_up:
            spins[x, y] = 1
        else:
            spins[x, y] = -1
        if backup != spins[x, y]:
            flipped_sites[flip_count] = (x,y)  # No change in spin
            flip_count += 1
    return spins, flipped_sites[:flip_count]

########################### Binning Function #################

def bins(data,Nperbin,Nbins):
    # Each bin is a point in the array, values are continuously added to it
    # Ex. 'Ebins' would be a 1xNbins array containing values of E (each value is a sum of Nperbin values)
    # This function takes the average of each bin
    
    # Nbins = len(data), but we can also supply it as an argument

    if Nbins != len(data):
        print('Check array size')
        return

    Bin_avgs = data / Nperbin

    Bin_totalavg=np.mean(Bin_avgs) #calculates one total value

    #This is where we calculate the error bars
    ErrorBars=0
    for i in range(Nbins):
        ErrorBars+=(Bin_avgs[i]-Bin_totalavg)**2
    ErrorBars=np.sqrt(1/Nbins)*np.sqrt(1/(Nbins-1))*np.sqrt(ErrorBars)

    return Bin_totalavg,ErrorBars
    

############################# Image Generation #############################

def init_rgb_array(spins, L):
    # Create an RGB array for the image: white = +1, black = -1
    rgb_array = np.zeros((L, L, 3), dtype=np.uint8)
    rgb_array[spins == 1] = [255, 255, 255]  # white
    rgb_array[spins == -1] = [0, 0, 0]       # black
    return rgb_array

def update_spins_image(spins, flipped_sites, rgb_array, scale):
    for x, y in flipped_sites:
        rgb_array[x, y] = [255, 255, 255] if spins[x, y] == 1 else [0, 0, 0]

    # Scale using repeat
    scaled_array = np.repeat(np.repeat(rgb_array, scale, axis=0), scale, axis=1)

    return Image.fromarray(scaled_array, 'RGB')

def reset_for_parameter_change():
    global Acceptance, sweepcount, E, M, timer, sweep_counter
    Acceptance = 0
    sweepcount = 1
    E = Energy(spins,J,h)
    M = Mag(spins)
    timer = time.time()
    sweep_counter = 0

def update_temp(val):
    global T
    T = float(val)
    temp_entry.delete(0, tk.END)
    temp_entry.insert(0, f"{T:.2f}")
    reset_for_parameter_change()

def update_coupling(val):
    global J
    J = float(val)
    coupling_entry_main.delete(0, tk.END)
    coupling_entry_main.insert(0, f"{J:.2f}")
    reset_for_parameter_change()

def update_magneticfield(val):
    global h
    h = float(val)
    magneticfield_entry_main.delete(0, tk.END)
    magneticfield_entry_main.insert(0, f"{h:.2f}")
    reset_for_parameter_change()

def update_temp_entry(val):
    try:
        T_val = float(val)
        if 0.1 <= T_val <= 5.0:
            temp_slider.set(T_val)
    except ValueError:
        pass
    reset_for_parameter_change()

def update_coupling_entry(val):
    try:
        J_val = float(val)
        if -2.0 <= J_val <= 2.0:
            coupling_slider.set(J_val)
    except ValueError:
        pass
    reset_for_parameter_change()

def update_magneticfield_entry(val):
    try:
        h_val = float(val)
        if -2.0 <= h_val <= 2.0:
            magneticfield_slider.set(h_val)
    except ValueError:
        pass
    reset_for_parameter_change()

def update_plot_choice(event):
    global data_buffer, line, plot_observable, sweepcount
    plot_observable = observable_dropdown.get()
    if plot_observable == "Energy":
        ax.set_ylabel("Energy Per Site (E / $L^2$)")
        ax.set_ylim(-2, 2)
    elif plot_observable == "Magnetization":
        ax.set_ylabel("Magnetization Per Site (M / $L^2$)")
        ax.set_ylim(-1, 1)
    elif plot_observable == "Acceptance":
        ax.set_ylabel("Acceptance")
        ax.set_ylim(0, 1)
    data_buffer.clear()
    line.set_ydata([0]*100)
    ax.set_title(f"Live {plot_observable} Vs. Time")
    canvas.draw()

def update_observable_labels():
    global sweep_counter
    energy_label.config(text=f"Energy: {E / (L**2):>9.3f}")
    magnetization_label.config(text=f"Magnetization: {M / (L**2):>9.3f}")
    acceptance_label.config(text=f"Acceptance: {Acceptance/sweepcount:>9.3f}")
    timer_label.config(text=f"Sweeps/second: {sweep_counter / (time.time() - timer - 0.1):>9.3f}")
    root.after(50, update_observable_labels)

def update_algorithm_choice_main(event):
    global algorithm, magneticfield_entry, magneticfield_slider
    algorithm = algorithm_dropdown_main.get()
    if algorithm == "Kawasaki" or algorithm == "Wolff" or algorithm == "Swendsen-Wang":
        magneticfield_entry_main.insert(0, 0)  # set initial value
        magneticfield_slider.set(0)
        magneticfield_entry_main.config(state=tk.DISABLED)
        magneticfield_slider.config(state=tk.DISABLED)
    else:
        magneticfield_entry_main.config(state=tk.NORMAL)
        magneticfield_slider.config(state=tk.NORMAL)
    
    # reset_for_parameter_change()
    
    # reset_for_parameter_change()

def update_size_choice(event):
    global L, scale, spins, rgb_array, label_img, label, E, M
    L = int(size_dropdown.get())
    scale = 512 // L
    spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int32)
    rgb_array = init_rgb_array(spins, L)
    pil_img = update_spins_image(spins, [], rgb_array, scale)
    label_img = ImageTk.PhotoImage(pil_img)
    reset_for_parameter_change()

def update_size(L):
    global scale, spins, rgb_array, label_img, label, E, M, size_dropdown
    size_dropdown.set(str(L))
    scale = 512 // L
    spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int32)
    rgb_array = init_rgb_array(spins, L)
    pil_img = update_spins_image(spins, [], rgb_array, scale)
    label_img = ImageTk.PhotoImage(pil_img)
    reset_for_parameter_change()
    
def generate_data():
    global progressbar
    def input_onoff(CONFIG):
        temp_slider.config(state=CONFIG)
        temp_entry.config(state=CONFIG)
        coupling_slider.config(state=CONFIG)
        coupling_entry_main.config(state=CONFIG)
        coupling_entry_popup.config(state=CONFIG)
        magneticfield_slider.config(state=CONFIG)
        magneticfield_entry_main.config(state=CONFIG)
        magneticfield_entry_popup.config(state=CONFIG)
        warmup_entry.config(state=CONFIG)
        Ti_entry.config(state=CONFIG)
        Tf_entry.config(state=CONFIG)
        Ts_entry.config(state=CONFIG)
        L_entry.config(state=CONFIG)
        algorithm_dropdown_main.config(state=CONFIG)
        measurement_entry.config(state=CONFIG)
        bin_entry.config(state=CONFIG)
        size_dropdown.config(state=CONFIG)

    def run_data_generation():
        global T, Ti, Tf, Ts, L, J, h, algorithm, spins, E, M, Acceptance, sweepcount, scale, T_values, T_counter, RUN_SIM
        global warmup_sweeps, measurement_sweeps, pil_img, E_bins, M_bins, N_bins, Nperbin
        try:
            Ti = float(Ti_entry.get())
            Tf = float(Tf_entry.get())
            Ts = float(Ts_entry.get())
            L = int(L_entry.get())
            size_dropdown.set(str(L))
            J = float(coupling_entry_popup.get())
            h = float(magneticfield_entry_popup.get())
            warmup_sweeps = int(warmup_entry.get())
            measurement_sweeps = int(measurement_entry.get())
            N_bins = int(bin_entry.get())
            Nperbin = measurement_sweeps // N_bins
            algorithm = algorithm_dropdown_popup.get()

            E_bins = np.zeros(N_bins, dtype=np.float64) # energy bins
            M_bins = np.zeros(N_bins, dtype=np.float64) # magnetization bins 


        except ValueError:
            print("Invalid input values.")
            return
        
        # Validate inputs
        if L%2 != 0:
            print("Lattice size (L) must be even.")
            return
        if h != 0 and algorithm in ["Wolff", "Swendsen-Wang", "Kawasaki"]:
            print("External magnetic field (h) is not supported for Wolff, Swendsen-Wang, Kawasaki algorithms.")
            return
        if Tf <= 0 or Ti <= 0 or Ts <= 0:
            print("Final Temperature (Tf), Initial Temperature (Ti), and Temperature Step (Ts) must be greater than 0.")
            return
        if Ti < Tf:
            print("Initial Temperature (Ti) must be greater than or equal to Final Temperature (Tf).")
            return
        
        update_magneticfield_entry(h)
        update_magneticfield(h)
        update_coupling_entry(J)
        update_coupling(J)
        update_size(L)
        
        T_counter = 0
        T_values = np.arange(Ti, Tf - Ts, -Ts).round(3)
        T = T_values[T_counter]
        update_temp_entry(T)
        update_temp(T)

        E = Energy(spins, J, h)
        M = Mag(spins)
        Acceptance = 0
        sweepcount = 1

        RUN_SIM = True
        print(f"Running simulation from Ti={Ti} to Tf={Tf} with step Ts={Ts}, L={L}, J={J}, h={h}, algorithm={algorithm}")
        start_btn.config(text="Stop", command=stop_data_generation)
        progressbar["value"] = 0
        input_onoff(tk.DISABLED)

    def stop_data_generation():
        global RUN_SIM
        RUN_SIM = False
        input_onoff(tk.NORMAL)
        start_btn.config(text="Start", command=run_data_generation)
        print("Simulation stopped.")

    sim_win = tk.Toplevel(root)
    sim_win.title("Run a Simulation")
    sim_win.geometry("400x400")

    Ti_label = ttk.Label(sim_win, text="Initial Temperature (Ti):")
    Ti_label.grid(row=0, column=0, padx=5, pady=5)
    Ti_entry = ttk.Entry(sim_win, width=10)
    Ti_entry.insert(0, str(5.0))
    Ti_entry.grid(row=0, column=1, padx=5, pady=5)

    Tf_label = ttk.Label(sim_win, text="Final Temperature (Tf):")
    Tf_label.grid(row=1, column=0, padx=5, pady=5)
    Tf_entry = ttk.Entry(sim_win, width=10)
    Tf_entry.insert(0, str(1.0))
    Tf_entry.grid(row=1, column=1, padx=5, pady=5)

    Ts_label = ttk.Label(sim_win, text="Temperature Step (Ts):")
    Ts_label.grid(row=2, column=0, padx=5, pady=5)
    Ts_entry = ttk.Entry(sim_win, width=10)
    Ts_entry.insert(0, str(0.1))
    Ts_entry.grid(row=2, column=1, padx=5, pady=5)

    L_label = ttk.Label(sim_win, text="Lattice Size (L):")
    L_label.grid(row=3, column=0, padx=5, pady=5)
    L_entry = ttk.Entry(sim_win, width=10)
    L_entry.insert(0, str(10))
    L_entry.grid(row=3, column=1, padx=5, pady=5)

    algorithm_label = ttk.Label(sim_win, text="Algorithm:")
    algorithm_label.grid(row=4, column=0, padx=5, pady=5)
    algorithm_dropdown_popup = ttk.Combobox(sim_win, values=["Metropolis", "Wolff", "Glauber", "Swendsen-Wang", "Kawasaki", "HeatBath"])
    algorithm_dropdown_popup.current(0)
    algorithm_dropdown_popup.grid(row=4, column=1, padx=5, pady=5)
    algorithm_dropdown_popup.config(state="disabled")  # Disable dropdown for now

    coupling_label = ttk.Label(sim_win, text="Coupling Constant (J):")
    coupling_label.grid(row=5, column=0, padx=5, pady=5)
    coupling_entry_popup = ttk.Entry(sim_win, width=10)
    coupling_entry_popup.insert(0, str(1.0))
    coupling_entry_popup.grid(row=5, column=1, padx=5, pady=5)

    magneticfield_label = ttk.Label(sim_win, text="Magnetic Field (h):")
    magneticfield_label.grid(row=6, column=0, padx=5, pady=5)
    magneticfield_entry_popup = ttk.Entry(sim_win, width=10)
    magneticfield_entry_popup.insert(0, str(0.0))
    magneticfield_entry_popup.grid(row=6, column=1, padx=5, pady=5)

    warmup_label = ttk.Label(sim_win, text="Warmup Sweeps:")
    warmup_label.grid(row=7, column=0, padx=5, pady=5)
    warmup_entry = ttk.Entry(sim_win, width=10)
    warmup_entry.insert(0, str(1000))
    warmup_entry.grid(row=7, column=1, padx=5, pady=5)

    measurement_label = ttk.Label(sim_win, text="Measurement Sweeps:")
    measurement_label.grid(row=8, column=0, padx=5, pady=5)
    measurement_entry = ttk.Entry(sim_win, width=10)
    measurement_entry.insert(0, str(1000))
    measurement_entry.grid(row=8, column=1, padx=5, pady=5)

    bin_label = ttk.Label(sim_win, text="Number of Bins:")
    bin_label.grid(row=9, column=0, padx=5, pady=5)
    bin_entry = ttk.Entry(sim_win, width=10)
    bin_entry.insert(0, str(20))
    bin_entry.grid(row=9, column=1, padx=5, pady=5)

    start_btn = ttk.Button(sim_win, text="Start", command=run_data_generation)
    start_btn.grid(row=10, column=0, padx=5, pady=5)

    progressbar = ttk.Progressbar(sim_win, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progressbar.grid(row=10, column=1, padx=5, pady=5)

def update_plot(E, M, L, data_buffer):
    global root, line
    if plot_observable == "Energy":
        data_buffer.append(E / L**2)
    elif plot_observable == "Magnetization":
        data_buffer.append(M / L**2)
    elif plot_observable == "Acceptance":
        data_buffer.append(Acceptance / sweepcount)
    line.set_ydata(list(data_buffer) + [0] * (100 - len(data_buffer)))
    root.after_idle(canvas.draw)

def run_simulation():
    # This is our main simulation loop, called every few milliseconds by Tkinter's after method. 
    # It performs a sweep of the lattice, updates the image and the plot.
    global spins, T, J, Acceptance, label_img, label, E, M, L, plot_observable, sweepcount
    global count, algorithm, sweep_counter, warmup_sweeps, measurement_sweeps
    global E_bins, M_bins, RUN_SIM, T_counter

    if algorithm == "Metropolis":
        spins, Acceptance, flipped_sites, E, M, sweepcount = Metropolis(spins, T, J, h, E, M, L, Acceptance, sweepcount)
    elif algorithm == "Wolff":
        spins, flipped_sites = Wolff(spins, T, J, L, h)
        E = Energy(spins,J,h)
        M = Mag(spins)
    elif algorithm == "Glauber":
        spins, Acceptance, flipped_sites, E, M, sweepcount = Glauber(spins, T, J, h, E, M, L, Acceptance, sweepcount)
    elif algorithm == "Swendsen-Wang":
        spins, flipped_sites = SwendsenWang(spins, T, J, h, L)
        E = Energy(spins,J,h)
        M = Mag(spins)
    elif algorithm == "Kawasaki":
        spins, flipped_sites = Kawasaki(spins, T, J, h, L)
        E = Energy(spins,J,h)
    elif algorithm == "HeatBath":
        spins, flipped_sites = HeatBath(spins, T, J, h, L)
        E = Energy(spins,J,h)
        M = Mag(spins)
    if RUN_SIM:
        if warmup_sweeps <= sweep_counter < warmup_sweeps + measurement_sweeps:
            E_bins[sweep_counter % N_bins] += E
            M_bins[sweep_counter % N_bins] += M
        elif sweep_counter >= warmup_sweeps + measurement_sweeps:
            E_val, E_err = bins(E_bins, Nperbin, N_bins)
            M_val, M_err = bins(M_bins, Nperbin, N_bins)
            print(f"T: {T}, E: {E_val / L**2}, Error: {E_err / L**2}, M: {M_val / L**2}, Error: {M_err / L**2}")
            sweep_counter = 0
            progressbar.step(1 / len(T_values) * 100)
            T_counter += 1
            if T_counter >= len(T_values):
                RUN_SIM = False
                T_counter = 0
                print("Simulation complete.")
            else:
                T = T_values[T_counter]
                update_temp_entry(T)
                update_temp(T)
                reset_for_parameter_change()

                E = Energy(spins, J, h)
                M = Mag(spins)
                Acceptance = 0
                sweepcount = 1

                E_bins[:] = 0
                M_bins[:] = 0
    sweep_counter += 1

    # update the image
    pil_img = update_spins_image(spins, flipped_sites, rgb_array, scale)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    # update the plot after 2 run_simulation calls (~10ms)
    count = (count + 1) % 3
    if count == 0:
        update_plot(E, M, L, data_buffer)

    root.after(5, run_simulation)

# initialize spins randomly
spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int32)
E = Energy(spins,J,h)
M = Mag(spins)

# initialize the RGB image array
rgb_array = init_rgb_array(spins, L)

# initialize the timer
timer = time.time()

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
plot_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

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
ax.set_ylabel(f"{plot_observable} per Site")
fig.tight_layout()

# Create the sliders and add them to the slider frame
temp_label = ttk.Label(slider_frame, text="Temperature (T):")
temp_label.grid(row=1, column=0, padx=5, pady=5)
temp_slider = ttk.Scale(slider_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, value=T, length=250)
temp_slider.grid(row=1, column=1, padx=5, pady=5)
temp_slider.config(command=update_temp)
temp_entry = ttk.Entry(slider_frame, width=5)
temp_entry.insert(0, str(T))  # set initial value
temp_entry.bind("<Return>", lambda event: update_temp_entry(temp_entry.get()))
temp_entry.grid(row=1, column=2, padx=5, pady=5)


coupling_label = ttk.Label(slider_frame, text="Coupling (J):")
coupling_label.grid(row=2, column=0, padx=5, pady=5)
coupling_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=J, length=250)
coupling_slider.grid(row=2, column=1, padx=5, pady=5)
coupling_slider.config(command=update_coupling)
coupling_entry_main = ttk.Entry(slider_frame, width=5)
coupling_entry_main.insert(0, str(J))  # set initial value
coupling_entry_main.bind("<Return>", lambda event: update_coupling_entry(coupling_entry_main.get()))
coupling_entry_main.grid(row=2, column=2, padx=5, pady=5)

magneticfield_label = ttk.Label(slider_frame, text="Magnetic Field (h):")
magneticfield_label.grid(row=3, column=0, padx=5, pady=5)
magneticfield_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=h, length=250)
magneticfield_slider.grid(row=3, column=1, padx=5, pady=5)
magneticfield_slider.config(command=update_magneticfield)
magneticfield_entry_main = ttk.Entry(slider_frame, width=5)
magneticfield_entry_main.insert(0, str(h))  # set initial value
magneticfield_entry_main.bind("<Return>", lambda event: update_magneticfield_entry(magneticfield_entry_main.get()))
magneticfield_entry_main.grid(row=3, column=2, padx=5, pady=5)

observable_label = ttk.Label(slider_frame, text="Observable to Plot:")
observable_label.grid(row=4, column=0, padx=5, pady=5)
observable_dropdown = ttk.Combobox(slider_frame, values=["Magnetization", "Energy", "Acceptance"], state="readonly")
observable_dropdown.current(0)
observable_dropdown.grid(row=4, column=1, padx=5, pady=5)
observable_dropdown.bind("<<ComboboxSelected>>", update_plot_choice)

size_label = ttk.Label(slider_frame, text="Size (L):")
size_label.grid(row=6, column=0, padx=5, pady=5)
size_dropdown = ttk.Combobox(slider_frame, values=[4, 8, 16, 32, 64, 128, 256], state="readonly")
size_dropdown.current(4)
size_dropdown.grid(row=6, column=1, padx=5, pady=5)
size_dropdown.bind("<<ComboboxSelected>>", update_size_choice)

# Live update labels for observables/stats
acceptance_label = ttk.Label(slider_frame, text=f"Acceptance: {Acceptance/sweepcount:>7.3f}")
acceptance_label.grid(row=4, column=2, padx=5, pady=5)
acceptance_label.config(width=25)
energy_label = ttk.Label(slider_frame, text=f"Energy: {E / (L**2):>7.3f}")
energy_label.grid(row=5, column=2, padx=5, pady=5)
energy_label.config(width=25)
magnetization_label = ttk.Label(slider_frame, text=f"Magnetization: {M / (L**2):>7.3f}")
magnetization_label.grid(row=6, column=2, padx=5, pady=5)
magnetization_label.config(width=25)
timer_label = ttk.Label(slider_frame, text=f"Sweeps/second: {sweep_counter / (time.time() - timer - 0.1):>7.3f}")
timer_label.grid(row=7, column=2, padx=5, pady=5)
timer_label.config(width=25)

algorithm_label = ttk.Label(slider_frame, text="Algorithm:")
algorithm_label.grid(row=5, column=0, padx=5, pady=5)
algorithm_dropdown_main = ttk.Combobox(slider_frame, values=["Metropolis", "Wolff", "Glauber", "Swendsen-Wang", "Kawasaki", "HeatBath"], state="readonly")
algorithm_dropdown_main.current(0)
algorithm_dropdown_main.grid(row=5, column=1, padx=5, pady=5)

algorithm_dropdown_main.bind("<<ComboboxSelected>>", update_algorithm_choice_main)

generate_data_btn = ttk.Button(slider_frame, text="Generate Data", command=lambda: generate_data())
generate_data_btn.grid(row=7, column=1, padx=5, pady=5)

# # precompile numba functions
# This is only necessary for functions without numba signatures
# Since it's already implemented for Kawasaki, Glauber, and HeatBath we don't need to call them here.
if not CACHE:
    Wolff(spins, T, J, L, h)
    SwendsenWang(spins, T, J, h, L)

# Get a fresh timer so the initial sweeps/sec calculation isn't skewed
timer = time.time()
# run the window and simulation
root.after(50, update_observable_labels)
root.after(5, run_simulation)
root.mainloop()