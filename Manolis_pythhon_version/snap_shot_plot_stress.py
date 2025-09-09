# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:57:45 2024

@author: parastatidise
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
import re
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from tkinter.filedialog import asksaveasfile
from tkinter import *
from tkinter import messagebox
import matplotlib
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import os
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from matplotlib.widgets import Slider
from vispy import app, gloo
#from mFile import mFile

window = Tk()
      
    # # setting the title 
window.title('Model Viewer')
      
    # # dimensions of the main window
window.geometry("1000x1000")


class Plot_snap:
    
    def __init__(self, master):
        self.myFrame = Frame(master)
        self.myFrame.pack()
        

        self.open_button = Button(text='Read model File',
                                      height = 2, 
                                      width = 20,
                                      command=self.open_text_file)
        self.open_button.place(x=10, y=600)

    
    
    def read_snp(self,m_filename,snp_filename): 
        self.m_filename=m_filename
        self.snp_filename=snp_filename
        global M_filename
        M_filename=self.m_filename
     
        with open(self.snp_filename, 'rb') as fid:
            snap_1D =  np.fromfile(fid, np.float32) 
        fid.close()
        
        Plot_snap.scan_mfile_v2(self, M_filename)
        N_snap_type = self.snap_parameters.shape[0-0]
        grid = self.FD_parameters[0,0].split(', ')
        #grid = int(grid)
        dx_dz_dy = self.FD_parameters[0,2]
        nx = int(grid[0])
        
        nz = int(grid[1])
        
        ny = int(grid[2])
        
        cycles = int(self.FD_parameters[0,3])
        rep_vec = np.zeros((N_snap_type))
        for k in range(N_snap_type):
            rep_vec[k] = self.snap_parameters[k,1]
        
        rep = np.amax(rep_vec)
        if N_snap_type != 0:
            n_snap = int(cycles / rep)
        else:
            n_snap = 0
        #x=self.snap_parameters[0,2]
        length_i = 0
        length_j = 0
        length_k = 0
        for m in range(N_snap_type):
            NZ_vabs_k = nz + 1
            NX_vabs_k = nx + 1
            NZ_vabs_k = NZ_vabs_k + 2
            length_vabs_k = NZ_vabs_k * NX_vabs_k
            length_k = length_k + length_vabs_k
            
            
            # elif 's22' == self.snap_parameters[m,0]:
            #     if 'i' == self.snap_parameters[m,2]:
            #         NZ_s22_i = nz + 1
            #         NY_s22_i = ny + 1
            #         NZ_s22_i = NZ_s22_i + 2
            #         length_s22_i = NZ_s22_i * NY_s22_i
            #         length_i = length_i + length_s22_i
            #     elif 'j' == self.snap_parameters[m,2]:
            #         NY_s22_j = ny + 1
            #         NX_s22_j = nx + 1
            #         NY_s22_j = NY_s22_j + 2
            #         length_s22_j = NY_s22_j * NX_s22_j
            #         length_j = length_j + length_s22_j
            #     elif 'k' == self.snap_parameters[m,2]:
            #         NZ_s22_k = nz + 1
            #         NX_s22_k = nx + 1
            #         NZ_s22_k = NZ_s22_k + 2
            #         length_s22_k = NZ_s22_k * NX_s22_k
            #         length_k = length_k + length_s22_k  
            
            
            
            

        tot_length = length_i + length_j + length_k
        #n_chunks = size(snap_1D,2)/tot_length;
        
        self.snaps =np.empty(shape=(N_snap_type, n_snap), dtype='object') #cell(N_snap_type,n_snap)
        if tot_length != 0:
            snap_1D_snapshots = snap_1D[0:tot_length * n_snap]
            snap_1_5D = snap_1D_snapshots.reshape(n_snap,(int(tot_length)))
            n_samples = np.zeros((N_snap_type + 1), dtype=int)
            
            
            # Define the grid limits (in kilometers)
            x_min, x_max = 0, NX_vabs_k  # X-axis from 0 to 100 km
            y_min, y_max = 0, NZ_vabs_k  # Y-axis from 0 to 100 km

            x1 = np.linspace(x_min, x_max, NX_vabs_k)
             
            y1 = np.linspace(y_min, y_max, NZ_vabs_k)


            y_1, x_1 = np.meshgrid(y1,x1)
            
            for h in range(n_snap):
                for m in range(1,N_snap_type+1,1): #np.arange(2,N_snap_type + 1+1).reshape(-1):
                    if 'vabs' == self.snap_parameters[m - 1,0]:
                        n_samples[m] = length_vabs_k
                        x=(sum(n_samples[0:m - 1])) #velocity
                        y=sum(n_samples[0:m+1])
                        snapshot = snap_1_5D[h,x:y].reshape(NX_vabs_k,NZ_vabs_k)
                        
                    # elif 's22' == self.snap_parameters[m - 1,0]:
                    #     n_samples[m] = length_vabs_k
                    #     #x=(sum(n_samples[0:m - 1])) #velocity
                    #     x=(sum(n_samples[0:m])) #stress
                    #     y=sum(n_samples[0:m+1])
                    #     snapshot = snap_1_5D[h,x:y].reshape(NX_vabs_k,NZ_vabs_k)
                    elif 'dil' == self.snap_parameters[m - 1,0]:
                        n_samples[m] = length_vabs_k
                        x=(sum(n_samples[0:m - 1])) #velocity
                        y=sum(n_samples[0:m+1])
                        snapshot = snap_1_5D[h,x:y].reshape(NX_vabs_k,NZ_vabs_k)
                        
                    elif 's22' == self.snap_parameters[m - 1,0]:
                        if 'k' == self.snap_parameters[m - 1,2]:
                            n_samples[m] = length_vabs_k
                            x=(sum(n_samples[0:m]))
                            y=sum(n_samples[0:m+1])
                            snapshot = snap_1_5D[h,x:y].reshape(NX_vabs_k,NZ_vabs_k)
                            self.snaps[m - 1,h] = np.transpose(snapshot)
                    
                    
                    vmin = -5e7  # Minimum value for color scale
                    vmax = 0e5  # Maximum value for color scale
                    fig, ax = plt.subplots()

                    plt.rcParams['figure.dpi']=1200
                    #p = ax.pcolor(x_1, y_1, snapshot, cmap=matplotlib.cm.jet)#, vmin=-0.5, vmax=1)
                    #p=plt.imshow(snapshot, origin='lower', aspect='auto', extent=[0, 400, 0, 560])#, vmin=vmin, vmax=vmax )
                    p=plt.imshow(np.transpose(snapshot), origin='lower', aspect='auto', extent=[0, 1400, 1000, 0], vmin=vmin, vmax=vmax )
                    #colorbar=fig.colorbar(p, ax=ax)
                    #colorbar.set_label('(Pa)', fontsize=12, weight='bold')
                    plt.title("Dilational Stress", fontsize=12, weight='bold')
                    plt.xlabel("Length (m)", fontsize=12, weight='bold')
                    plt.ylabel("Depth (m)", fontsize=12, weight='bold')
                    #plt.savefig(f"snapshot{m}.png")
                    plt.show()
                    
                    #self.snaps[m - 1,h] = np.transpose(snapshot)#[1:,]
                            
        
        

    
    def scan_mfile_v2(self,M_filename): 
        readfile = open(M_filename, "r")
        #with open(self.m_filename,'r') as readfile:
            #mylist = [line.rstrip('\n') for line in m_file_text]
            #m_file_text=str(m_file_text)
            # content=m_file_text.read()
        #m_file_text = open(m_filename,'r')
        
        self.snap_parameters =np.empty(shape=(0, 4),dtype=str)
        self.FD_parameters = np.empty(shape=(0, 4),dtype=str)
        dx_dz_dy=[]
        dt=[]
        cycle=[]
        grid=[]
        
    
         # =========================================================================
         # =================================== DT ==================================
    
        for line in readfile:
            Type = line.split("\n")
            x = Type[0].split(' ') 
            if  x != ['']:
                if len(x)>=1 :
                    if x[0].startswith('tstp=') : #or x[0]=='tstp' :
                        #axis=x[0].split('=')
                        dt=x[0].split('=')
                        dt=float(dt[1])
                        
        # =========================================================================
        # ================================ PL SNAP  ===============================

                    elif x[1] == 'snap':
                        #p=0
                        snap_type=x[2]
                        if len(x)>=3:
                            if x[3].startswith('rep'):
                                rep=x[3].split('=')
                                rep=rep[1]
                            else:
                                rep=1
                            if len(x)>=5:
                                axis=re.split('= |, ', x[4])
                                
                                axis_coord=int(axis[1].split(','))
                                snap_axis=axis[0]
                            else:
                                Gr=grid.split(', ')
                                Gr=int(Gr[2])
                                axis_coord=(int(Gr/2))
                                snap_axis='k'
                            self.snap_parameters=np.append(self.snap_parameters, np.array([[snap_type, rep, snap_axis, axis_coord]]), axis=0)    
                                
         # =========================================================================
         # ================================= CYCLE =================================        
                    elif x[0] == 'cy' or x[0] == 'cycles':    
                
                        i=int(x[1])
                #i=re.findall(r'\d+', i[0])
                #i=[eval(i) for i in i ]
                        cycle=i
         # =========================================================================
         # ================================== GRID  ================================     
                    elif x[0] == 'gr' or x[0] == 'g':
                        x=[l for l in x if l.strip()]
                        grid =f'{x[1]}, {x[2]}, {x[3]}'
                        dx_dz_dy=f'{x[4]}, {x[5]}, {x[6]}'
        
        self.FD_parameters = np.append(self.FD_parameters,  np.array([[grid, dt, dx_dz_dy, cycle]]), axis=0) 
       # return FD_parameters, snap_parameters
    


    def open_text_file(self):
        # file type
        filetypes = (
            ('text files', '*.snp'),
            #('text files', '*.m'),
            #('All files', '*.*')
        )
        # show the open file dialog
        f = fd.askopenfilename(filetypes=filetypes)
        #global filename
        x=str(f)
        x=x.split('.')
       # x.split('.m')
        #view_hist_files(
        global filename
        filename= x[0]
        Plot_snap.getModelData(self)
        
    def getModelData(self):
        if filename:
            m_filename=filename + '.m'
            snp_filename=filename +'.snp'
            #Plot_snap.readMfile(self,filenameM)
            Plot_snap.read_snp(self,m_filename,snp_filename)
            


e= Plot_snap(window)
    
global filename
filename= None     
              
# run the gui
window.mainloop()                       



    
    
# obj=Plot_snap()    
# obj.read_snp('lm.m','lm.snp')
    
