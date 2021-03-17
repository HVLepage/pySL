# -*- coding: utf-8 -*-

# import tkinter module 
# from tkinter import * 
# from tkinter.ttk import *

import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm
import time
import tracemalloc
import humanize


################################################################################################################################
# Define commutation functions
################################################################################################################################

def commutator (A,B):
    return A@B - B@A

def anticommutator (A,B):
    return A@B + B@A


################################################################################################################################
# Simulation 
################################################################################################################################

def densitySolverQHO(Nx, T, dt, w, T1, T2s, initialState, DecoherenceSource):
    Nx = int(Nx)
    T = float(T)
    dt = float(dt)
    w = float(w)
    T1 = float(T1)
    T2s = float(T2s)


    PsiSize = (Nx)
    m = 1
    hbar = 1

    Xa = -40
    Xb = 40
    dx = (Xb-Xa)/(Nx-1)
    x=np.linspace(Xa,Xb,Nx)

    # Define ladder operators (This should change when not dealing with a QHO)
    a = np.zeros((PsiSize,PsiSize))
    rng = np.arange(PsiSize-1)
    a[rng, rng+1] = np.sqrt(rng+1)
    ad = np.transpose(a)

    #Init Hamiltonian
    H = np.zeros([PsiSize**2], dtype=np.complex64)

    # Define total hamiltonian
    H=np.reshape(H, [PsiSize,PsiSize])
    # Kinetic part of the Hamiltonian
    HK=np.reshape(H, [PsiSize,PsiSize])

    hbar2 = hbar ** 2 #hbar squared
    V = 0.5 * w**2 * x**2   

    hk = hbar2 / (2 * m * dx ** 2)

    for k in range(0, PsiSize):# Go over each row
        #fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
    
        HK[k][k] += 2 * hk # on-site kinetic part

        pmone= k -1
        if (pmone>=0 and pmone<Nx):
            HK[k][k-1] -= hk

        pmone= k+1
        if (pmone>=0 and pmone< Nx ):
            HK[k][k+1] -= hk

    # Add potential term to the Hamiltonian (can be time dependent)
    H = HK + np.diag(V)

    D, psi_eig = LA.eigh(H)

    psi0=psi_eig[:,0]
    psi1=psi_eig[:,1]

    # psi = (1/np.sqrt(2)) *(psi1+psi0)
    # psi = psi0

    xoffset = 10 # Wave packet distance from center
    re = (w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2) # + (w/np.pi)**(1/4)*np.exp(-w*(x+xoffset)**2/2) # Initial wave function
    psi = np.zeros(PsiSize, dtype=np.complex64)
    # psi += re


    if (initialState[0] == "0"):
        psi += psi_eig[:,0]
    elif (initialState[0] == "1"):
        psi += psi_eig[:,1]
    elif (initialState[0] == "2"):
        psi += psi_eig[:,2]
    elif (initialState[0] == "3"):
        psi += (w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2)
    elif (initialState[0] == "4"):
        psi += (w/np.pi)**(1/4)*np.exp(-4*w*(x)**2/2)

    rho=np.outer(psi,psi)
    rhodot = commutator(H,rho)/(hbar*1j)

    Nt=T/dt

    #rhofinal=expm(-1j *H *T/hbar)@rho@expm(1j *H *T/hbar)


    psi00 = np.zeros(int(Nt), dtype=np.complex64)
    psi01 = np.zeros(int(Nt), dtype=np.complex64)
    psi10 = np.zeros(int(Nt), dtype=np.complex64)
    psi11 = np.zeros(int(Nt), dtype=np.complex64)

    psi2 = np.zeros((int(Nt), int(PsiSize)))


    # Dissipation Lindblad term
    L1 = np.sqrt(1/T1) * a
    L1d = np.transpose(L1)

    # Dephasing Lindblad term
    L2 = np.sqrt(1/T2s)*(1/(hbar*w))*H
    L2d = np.transpose(L2)

    # Thermalization Lindblad term
    Temperature = 30
    kB = 0.1
    kBT = kB * Temperature
    L3 = np.sqrt(1/T1)*np.exp(- 0.5 * hbar*w/kBT ) * ad
    L3d = np.transpose(L3)

    if (DecoherenceSource[0] == "0"):
        for i in range(0,int(Nt)):
            rhodot = -1j*commutator(H,rho)/(hbar)
            rho += rhodot*dt
            psi2[i] = np.diag(rho).real

    elif (DecoherenceSource[0] == "1"):
        for i in range(0,int(Nt)):
            rhodot = -1j*commutator(H,rho)/(hbar)
            rhodot += L1@rho@L1d - 0.5 * anticommutator( L1d@L1, rho )
            rho += rhodot*dt
            psi2[i] = np.diag(rho).real

    elif (DecoherenceSource[0] == "2"):
        for i in range(0,int(Nt)):
            rhodot = -1j*commutator(H,rho)/(hbar)
            rhodot += L2@rho@L2d - 0.5 * anticommutator( L2d@L2, rho )
            rho += rhodot*dt
            psi2[i] = np.diag(rho).real

    elif (DecoherenceSource[0] == "3"):
        for i in range(0,int(Nt)):
            rhodot = -1j*commutator(H,rho)/(hbar)
            rhodot += L1@rho@L1d - 0.5 * anticommutator( L1d@L1, rho )
            rhodot += L2@rho@L2d - 0.5 * anticommutator( L2d@L2, rho )
            rho += rhodot*dt
            psi2[i] = np.diag(rho).real

    elif (DecoherenceSource[0] == "4"):
        for i in range(0,int(Nt)):
            rhodot = -1j*commutator(H,rho)/(hbar)
            rhodot += L1@rho@L1d - 0.5 * anticommutator( L1d@L1, rho )
            rhodot += L3@rho@L3d - 0.5 * anticommutator( L3d@L3, rho )
            rho += rhodot*dt
            psi2[i] = np.diag(rho).real


    DMGraph = np.zeros((PsiSize,PsiSize), dtype=np.complex64)
    for i in range(PsiSize):
        for j in range(PsiSize):
            DMGraph[i][j] = np.conjugate(psi_eig[:,i])@rho@psi_eig[:,j]

    return psi2, DMGraph




################################################################################################################################
# GUI
################################################################################################################################

root = tkinter.Tk()
root.wm_title("QHOrhoGUI in Tk")

# this will create a label widget 
l1 = tkinter.Label(root, text = "Nx") 
l2 = tkinter.Label(root, text = "T") 
l3 = tkinter.Label(root, text = "dt") 
l4 = tkinter.Label(root, text = "w") 
l5 = tkinter.Label(root, text = "Initial State") 
l6 = tkinter.Label(root, text = "Decoherence") 
l7 = tkinter.Label(root, text = "T1") 
l8 = tkinter.Label(root, text = "T2*") 

# grid method to arrange labels in respective 
# rows and columns as specified 
l1.grid(row = 0, column = 0, sticky = tkinter.W)
l2.grid(row = 1, column = 0, sticky = tkinter.W)
l3.grid(row = 2, column = 0, sticky = tkinter.W)
l4.grid(row = 3, column = 0, sticky = tkinter.W)
l5.grid(row = 0, column = 2, sticky = tkinter.W)
l6.grid(row = 1, column = 2, sticky = tkinter.W)
l7.grid(row = 2, column = 2, sticky = tkinter.W)
l8.grid(row = 3, column = 2, sticky = tkinter.W)

# entry widgets, used to take entry from user 
e1 = tkinter.Entry(root, width=5) 
e1.insert(tkinter.END, '21')
e2 = tkinter.Entry(root, width=5) 
e2.insert(tkinter.END, '2000')
e3 = tkinter.Entry(root, width=5) 
e3.insert(tkinter.END, '0.01')
e4 = tkinter.Entry(root, width=5) 
e4.insert(tkinter.END, '0.015')
# e5 = tkinter.Entry(root, width=5) 
# e6 = tkinter.Entry(root, width=5) 
e7 = tkinter.Entry(root, width=5) 
e7.insert(tkinter.END, '600')
e8 = tkinter.Entry(root, width=5) 
e8.insert(tkinter.END, '600')


# Create a Tkinter variable
tkvarinit = tkinter.StringVar(root)
# Dictionary with options
choicesinit = { '0 - Ground State','1 - 1st state','2 - 2nd state','3 - Coherent State','4 - Squeezed State'}
choicesinit = sorted(choicesinit)
tkvarinit.set('0') # set the default option

initMenu = tkinter.OptionMenu(root, tkvarinit, *choicesinit)
initMenu.config(width=4)
initMenu.grid(row = 0, column = 3)

# on change dropdown value
def init_dropdown(*args):
    print( tkvarinit.get() )
    printFigs()

# link function to change dropdown
tkvarinit.trace('w', init_dropdown)

# Create a Tkinter variable
tkvarsource = tkinter.StringVar(root)
# Dictionary with options
choicessource = { '0 - None','1 - Relaxation','2 - Dephasing','3 - Relaxation & Dissipation','4 - Thermalization'}
choicessource = sorted(choicessource)
tkvarsource.set('0') # set the default option

sourceMenu = tkinter.OptionMenu(root, tkvarsource, *choicessource)
sourceMenu.config(width=4)
sourceMenu.grid(row = 1, column = 3)

# on change dropdown value
def source_dropdown(*args):
    print( tkvarsource.get() )
    printFigs()

# link function to change dropdown
tkvarsource.trace('w', source_dropdown)

# this will arrange entry widgets 
e1.grid(row = 0, column = 1, pady = 2) 
e2.grid(row = 1, column = 1, pady = 2) 
e3.grid(row = 2, column = 1, pady = 2) 
e4.grid(row = 3, column = 1, pady = 2) 
# e5.grid(row = 0, column = 3, pady = 2) 
# e6.grid(row = 1, column = 3, pady = 2) 
e7.grid(row = 2, column = 3, pady = 2) 
e8.grid(row = 3, column = 3, pady = 2) 



def printFigs():
    psi2, DMGraph = densitySolverQHO(e1.get(), e2.get(), e3.get(), e4.get(), e7.get(), e8.get(), tkvarinit.get(), tkvarsource.get())

    # Figure trace
    fig_trace = Figure(figsize=(2, 3), dpi=100)
    t = np.arange(0, 3, .01)
    fig_trace.add_subplot(111).pcolormesh(psi2)

    canvas_trace = FigureCanvasTkAgg(fig_trace, master=root)  # A tk.DrawingArea.
    canvas_trace.draw()
    canvas_trace.get_tk_widget().grid(row=0,column=4,columnspan=2,rowspan=9)

    # Figure Energy
    fig_energy = Figure(figsize=(1, 1), dpi=100)
    t = np.arange(0, 3, .01)
    fig_energy.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    canvas_energy = FigureCanvasTkAgg(fig_energy, master=root)  # A tk.DrawingArea.
    canvas_energy.draw()
    canvas_energy.get_tk_widget().grid(row=5,column=0,columnspan=2,rowspan=2)

    # Figure Entropy
    fig_entropy = Figure(figsize=(1, 1), dpi=100)
    t = np.arange(0, 3, .01)
    fig_entropy.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    canvas_entropy = FigureCanvasTkAgg(fig_entropy, master=root)  # A tk.DrawingArea.
    canvas_entropy.draw()
    canvas_entropy.get_tk_widget().grid(row=5,column=2,columnspan=2,rowspan=2)

    # Figure States
    fig_states = Figure(figsize=(1, 1), dpi=100)
    t = np.arange(0, 3, .01)
    fig_states.add_subplot(111).bar(np.arange(0, int(e1.get())), np.diag(DMGraph.real))

    canvas_states = FigureCanvasTkAgg(fig_states, master=root)  # A tk.DrawingArea.
    canvas_states.draw()
    canvas_states.get_tk_widget().grid(row=7,column=0,columnspan=2,rowspan=2)

    # Figure Truth
    fig_truth = Figure(figsize=(1, 1), dpi=100)
    t = np.arange(0, 3, .01)
    fig_truth.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    canvas_truth = FigureCanvasTkAgg(fig_truth, master=root)  # A tk.DrawingArea.
    canvas_truth.draw()
    canvas_truth.get_tk_widget().grid(row=7,column=2,columnspan=2,rowspan=2)


printFigs()

tkinter.mainloop()




















# # creating main tkinter window/toplevel 
# master = Tk() 

# # this will create a label widget 
# l1 = Label(master, text = "Height") 
# l2 = Label(master, text = "Width") 

# # grid method to arrange labels in respective 
# # rows and columns as specified 
# l1.grid(row = 0, column = 0, sticky = W, pady = 2) 
# l2.grid(row = 1, column = 0, sticky = W, pady = 2) 

# # entry widgets, used to take entry from user 
# e1 = Entry(master) 
# e2 = Entry(master) 

# # this will arrange entry widgets 
# e1.grid(row = 0, column = 1, pady = 2) 
# e2.grid(row = 1, column = 1, pady = 2) 

# # checkbutton widget 
# c1 = Checkbutton(master, text = "Preserve") 
# c1.grid(row = 2, column = 0, sticky = W, columnspan = 2) 

# # adding image (remember image should be PNG and not JPG) 
# img = PhotoImage(file = r"./src/lindbladian/cat.png") 
# img1 = img.subsample(2, 2) 

# # setting image with the help of label 
# Label(master, image = img1).grid(row = 0, column = 2, 
# 	columnspan = 2, rowspan = 2, padx = 5, pady = 5) 

# # button widget 
# b1 = Button(master, text = "Zoom in") 
# b2 = Button(master, text = "Zoom out") 

# # arranging button widgets 
# b1.grid(row = 2, column = 2, sticky = E) 
# b2.grid(row = 2, column = 3, sticky = E) 

# # infinite loop which can be terminated 
# # by keyboard or mouse interrupt 
# mainloop() 