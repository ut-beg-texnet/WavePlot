"""
Parses a model (.m file) and plots the geometry
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
import re
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D

from tkinter import * 
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import os
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


# window = Tk()
      
#     # # setting the title 
# window.title('Model Viewer')
      
#     # # dimensions of the main window
# window.geometry("1000x700")


# Defining a custom-defined function cubes
class mFile: 
    
    def __init__(self,filename):
         
        
         self.filename=filename
    #     myFrame = Frame(master)
    #     myFrame.pack()

    #     self.open_button = Button(text='Read model File',
    #                                   height = 2, 
    #                                   width = 20,
    #                                   command=self.open_text_file)
    #     self.open_button.place(x=10, y=600)
    #     self.plot_button = tk.Button(command = self.getModelData,
    #                               height = 2, 
    #                               width = 20,
    #                               text = "Plot model Geometry")
    #     self.plot_button.place(x=170, y=600)
       



    def cubes(self):
        
        r1 = [0,self.side1]
        r2 = [0,self.side2]
        r3 = [0,self.side3]
        self.Y, self.Z = np.meshgrid(range(10), range(10))
        for s, e in combinations(np.array(list(product(r1, r3, r2))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                self.ax.plot3D(*zip(s, e), color="green")
        self.ax.set_title("Model Geometry")
        

    
    def stope_plot(self):
        index=2
        X=self.i
        Y=self.j
        Z=self.k
        
        if index > len(X):
            X.insert(1, X[0])
        elif index > len(Y):
            Y.insert(1, Y[0])
        elif index > len(Z):
            Z.insert(1, Z[0])    
            
        for s, e in combinations(np.array(list(product(X, Z, Y))), 2):
            if np.sum(np.abs(s-e)) == X[1]-X[0] or np.sum(np.abs(s-e)) == Y[1]-Y[0] or np.sum(np.abs(s-e)) == Z[1]-Z[0] : 
                self.ax.plot3D(*zip(s, e), color="red")

    def his_plot(self):
        i=self.his[0]
        j=self.his[1]
        k=self.his[2]
        self.ax.scatter(i,j,k, color='black')
        self.ax.text(i, j+1, k, f"S{self.num}")
 
    '''
    Parses the grid parameters from the model file
    '''
    def grid(self):
        self.x=[l for l in self.x if l.strip()]
        #side1 = np.array([ (int(x[1])), (int(x[2])), (int(x[3])) ])
        self.side1 = (int(self.x[1]))
        self.side2 = (int(self.x[2]))
        self.side3 = (int(self.x[3]))
        mFile.cubes(self)
       
    def stope(self):
        self.i = [i for i in self.x if i.startswith('i=')]
        self.j = [i for i in self.x if i.startswith('j=')]
        self.k = [i for i in self.x if i.startswith('k=')]
        #self.i=self.i.translate({ord('i='): None})
        if self.i != []: 
            self.i=re.findall(r'\d+', self.i[0])
            self.i=[eval(i) for i in self.i ]
            self.j=re.findall(r'\d+', self.j[0])
            self.j=[eval(i) for i in self.j ]
            self.k=re.findall(r'\d+', self.k[0])
            self.k=[eval(i) for i in self.k ]
            mFile.stope_plot(self)
            
    def history(self):
        
        yvel = [i for i in self.x if i.startswith('yvel')]
        xvel = [i for i in self.x if i.startswith('xvel')]
        zvel = [i for i in self.x if i.startswith('zvel')]
        #self.i=self.i.translate({ord('i='): None})
        if yvel !=[]:
            yvel=re.findall(r'\d+', yvel[0])
            yvel=[eval(i) for i in yvel ]
        if xvel !=[]: 
            xvel=re.findall(r'\d+', xvel[0])
            xvel=[eval(i) for i in xvel ]
        if zvel !=[]:       
            zvel=re.findall(r'\d+', zvel[0])
            zvel=[eval(i) for i in zvel ]
        if yvel==xvel and yvel==zvel:
            self.his=yvel
        if self.his !=[]:
            mFile.his_plot(self)
            
            
    '''
    Opens the file, creates a 3D axes, iterates over the lines, and splits into tokens
    '''
    def readMfile(self,filename):
            #self.filename = filename
        readfile = open(filename, "r")
        fig = plt.figure(figsize=(5,3), dpi=200)
        self.ax = fig.add_subplot(111, projection='3d')
        self.num=1
        for line in readfile:
            Type = line.split("\n")
            self.x = Type[0].split(' ')
            if self.x[0] == 'gr' or self.x[0] == 'g':
                mFile.grid(self)
                        
            elif self.x[0] == 'st' or self.x[0] == 'stope': 
                mFile.stope(self) 
            elif self.x[0] == 'his':
                mFile.history(self)
                self.num += 1
            
                # creating the Tkinter canvas
                # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master = window)  
        canvas.draw()
              
                # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
              
                # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,window)
        toolbar.pack()
        toolbar.update()
              
                # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()    
 
                
    def open_text_file(self):
        # file type
        filetypes = (
            ('text files', '*.hst'),
            ('text files', '*.m'),
            ('All files', '*.*')
        )
        # show the open file dialog
        f = fd.askopenfilename(filetypes=filetypes)
        #global filename
        x=str(f)
        #view_hist_files(
        global filename
        filename= x
        
    def getModelData(self):
        if filename: 
            mFile.readMfile(self,filename)

    
#e= mFile(window)
    
            
              
# run the gui

#window.mainloop()               
#       
 


            
