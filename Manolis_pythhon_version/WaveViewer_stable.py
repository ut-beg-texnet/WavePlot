"""
GUI for plotting waveforms
Reads history (.hst) files and model (.m) files
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

from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import os
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from scipy.fft import fft, fftfreq
from scipy import signal

from mFile import mFile

window = Tk()
      
# Setting the title and window properties
window.title('WavePlot')
window.geometry("600x400")
window.configure(bg='#f0f0f0')

# Configure style for modern look
style = ttk.Style()
style.theme_use('clam')

# Configure button styles
style.configure('Modern.TButton', 
                font=('Segoe UI', 10),
                padding=(10, 8),
                relief='flat',
                borderwidth=0)

style.map('Modern.TButton',
          background=[('active', '#e0e0e0'),
                     ('pressed', '#d0d0d0')])

style.configure('Primary.TButton',
                font=('Segoe UI', 10, 'bold'),
                padding=(10, 8),
                relief='flat',
                borderwidth=0)

style.map('Primary.TButton',
          background=[('active', '#4a90e2'),
                     ('pressed', '#357abd')])

# Defining a custom-defined function cubes
class WFile: 
    
    def __init__(self, master):
        # Main container with padding
        self.main_container = Frame(master, bg='#f0f0f0', padx=20, pady=20)
        self.main_container.pack(fill=BOTH, expand=True)
        
        # Title section
        title_frame = Frame(self.main_container, bg='#f0f0f0')
        title_frame.pack(fill=X, pady=(0, 20))
        
        title_label = Label(title_frame, 
                           text="WavePlot", 
                           font=('Segoe UI', 18, 'bold'),
                           bg='#f0f0f0',
                           fg='#2c3e50')
        title_label.pack()
        
        subtitle_label = Label(title_frame,
                              text="Waveform Plotting Tool",
                              font=('Segoe UI', 10),
                              bg='#f0f0f0',
                              fg='#7f8c8d')
        subtitle_label.pack()
        
        # Button container with grid layout
        button_frame = Frame(self.main_container, bg='#f0f0f0')
        button_frame.pack(expand=True)
        
        # Configure grid weights for responsive layout
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_rowconfigure(0, weight=1)
        button_frame.grid_rowconfigure(1, weight=1)
        
        # Create buttons the buttons
        self.open_button = ttk.Button(button_frame, 
                                     text='📁 Open History File',
                                     style='Primary.TButton',
                                     command=self.open_hist_file)
        self.open_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        self.plot_button = ttk.Button(button_frame,
                                     text='📊 Plot All Waveforms',
                                     style='Modern.TButton',
                                     command=self.plot)
        self.plot_button.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        
        self.plotS_button = ttk.Button(button_frame,
                                      text='🎯 Select Waveforms',
                                      style='Modern.TButton',
                                      command=self.open_popup)
        self.plotS_button.grid(row=0, column=2, padx=10, pady=10, sticky='ew')
        
        self.geometry_button = ttk.Button(button_frame,
                                         text='🏗️ Plot Model Geometry',
                                         style='Modern.TButton',
                                         command=self.Model_file)
        self.geometry_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        
        self.export_button = ttk.Button(button_frame,
                                       text='💾 Export to CSV',
                                       style='Modern.TButton',
                                       command=self.export)
        self.export_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        
        # Status bar
        self.status_frame = Frame(self.main_container, bg='#e8e8e8', height=30)
        self.status_frame.pack(fill=X, pady=(20, 0))
        self.status_frame.pack_propagate(False)
        
        self.status_label = Label(self.status_frame,
                                 text="Ready",
                                 font=('Segoe UI', 9),
                                 bg='#e8e8e8',
                                 fg='#555555')
        self.status_label.pack(side=LEFT, padx=10, pady=5)
        
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_label.config(text=message)
        self.status_label.update()
        
        
# read model file to identify the numeber of cycles, the timestep and the
# the velocity histories to use the results in order to make it available for 
# the user to select the waveforms need to plot        
    def readMfile(self,filenameM):
            #self.filename = filename
        readfile = open(filenameM, "r")
        allhis=[]
        
        for line in readfile:
            Type = line.split("\n")
            x = Type[0].split(' ')
            if 'tstp=' in x[0]:
                i = [i for i in x if i.startswith('tstp')]
                i=i[0].split('=')
                i=float(i[1])
                #i=re.findall(r'\d+', i[0])
                #i=[eval(i) for i in i ]
                self.dt=3.6e-4 #i
            elif x[0] == 'cy' or x[0] == 'cycles':    
                
                i=int(x[1])
                #i=re.findall(r'\d+', i[0])
                #i=[eval(i) for i in i ]
                self.cycles=i
            elif x[0] == 'his':
                yvel = [i for i in x if i.startswith('yvel')]
                if yvel != []:
                    allhis.append(yvel)    
                xvel = [i for i in x if i.startswith('xvel')]
                if xvel != []:
                    allhis.append(xvel)  
                zvel = [i for i in x if i.startswith('zvel')]
                if zvel != []:
                    allhis.append(zvel)
                hismx=[i for i in x if i.startswith('mx_')]
                if hismx != []:
                    allhis.append(hismx) 
        if allhis !=[]:
            allhst={}
            for index, element in enumerate(allhis):
                allhst[index]=element
            self.allhst=allhst
                
                
# read the history files from the '.hst' format to make it available for plot of export to csv            
    def read_W_file(self,filename):
        
        self.viewer= 1
       
        with open(filename, 'rb') as f: 
            traces =  np.fromfile(f, np.float32) 
            f.close()
        self.dt=3.6e-4
        tax = np.arange(start=0,stop=(self.dt*(self.cycles+1)),step=self.dt)
        traces = traces[1::3]

        nhist = int(len(traces) / (self.cycles + 1)) 

        
        self.trcs = np.zeros((self.cycles + 1,nhist + 1))
        trcs = np.zeros((self.cycles + 1,nhist + 1))
        n_plot = 1
        n_plot_temp = n_plot
        if np.remainder(nhist,n_plot) != 0: 
            n_iteration = np.rint(nhist / n_plot) + 1
        else: 
            n_iteration = np.rint(nhist / n_plot)
            
        x= np.arange(1,n_iteration+1).reshape(-1)  
        
        start=0
        stop=self.cycles+1
        for j in np.arange(1,n_iteration+1).reshape(-1): 
           m = 1
           k = n_plot * (j - 1)
           trace = traces[start : stop]
           self.trcs[0:(self.cycles+1),int(k+1)] = trace
           
           if j == np.rint(nhist / n_plot) + 1: 
               n_plot = np.remainder(nhist,n_plot)
           m = m + 1
           k = k + 1
           n_plot = n_plot_temp
           
           start+=self.cycles+1  #int((k) * (self.cycles + 1))
           stop+=self.cycles+1        #int((k+1)* (self.cycles
           
           
           

           # while k <= n_plot * j and k <= nhist: 
           #     trace = traces[start : stop]
           #         #np.insert(trcs, int(k+1), trace)
           #     self.trcs[0:(self.cycles+1),int(k+1)] = trace
           #     trcs[0:(self.cycles+1),int(k+1)] = trace
           #     if j == np.rint(nhist / n_plot) + 1: 
           #         n_plot = np.remainder(nhist,n_plot)
           #     m = m + 1
           #     k = k + 1
           #     n_plot = n_plot_temp
               
           #     start+=self.cycles+1  #int((k) * (self.cycles + 1))
           #     stop+=self.cycles+1        #int((k+1)* (self.cycles + 1)) 
               
        self.trcs[0:(self.cycles+1),0] = np.transpose(tax)
        self.nhist=nhist
     
 #plot original waveforms       
    def original(self):
        self.hide_all_frames()
        i=len(self.plotWvV)
        # adding the subplot
        fig, ax = plt.subplots(i)
        fig.tight_layout(pad=2.0)
        fig.figsize=(9,6)
        for j in range(0,i):
            x=self.plotWvV[j]
            ax[j].plot(self.trcs[0:(self.cycles+1),0], self.trcs[0:(self.cycles+1),(x+1)])
            #ax[j].plot.xlabel('Time (s)') 
            #ax[j].plot.ylabel('Velocity (m/s)') 
            ax[j].set_title(self.plotWvN[j])
            #ax.show()
        # plotting the graph
        canvas1=Canvas(self.main_frame)
        canvas1=FigureCanvasTkAgg(fig,master= self.top)
        canvas1.draw()
        toolbar = NavigationToolbar2Tk(canvas1,
                                 self.top)
        toolbar.pack()
        toolbar.update()
        canvas1.get_tk_widget().pack(side="top",fill='both',expand=True)
        canvas1.pack(side="top",fill='both',expand=True)
        canvas1.get_tk_widget().pack()   
        
        
 #plot filtered waveforms       
    def lowpassFilter(self):
        i=len(self.plotWvV)
        # adding the subplot
        fig, ax = plt.subplots(i)
        fig.tight_layout(pad=2.0)
        fig.figsize=(9,6)
        if self.L!=None or self.L!=0:
            sos = signal.butter(self.L, 'low', fs=1/float(self.dt), output='sos')
        for j in range(0,i):
            x=self.plotWvV[j]
            ax[j].plot(self.trcs[0:(self.cycles+1),0], signal.sosfilt(sos, self.trcs[0:(self.cycles+1),(x+1)]))
            #ax[j].plot.xlabel('Time (s)') 
            #ax[j].plot.ylabel('Velocity (m/s)') 
            ax[j].set_title(self.plotWvN[j])
            #ax.show()
        # plotting the graph
        canvas1=Canvas(self.main_frame)
        canvas1=FigureCanvasTkAgg(fig,master= self.top)
        canvas1.draw()
        toolbar = NavigationToolbar2Tk(canvas1,
                                 self.top)
        toolbar.pack()
        toolbar.update()
        canvas1.get_tk_widget().pack(side="top",fill='both',expand=True)
        canvas1.pack(side="top",fill='both',expand=True)
        canvas1.get_tk_widget().pack()   
        
        
#plot fourier waveforms        
    def fourier(self):
        self.hide_all_frames()
        i=len(self.plotWvV)
        # adding the subplot
        fig, ax = plt.subplots(i)
        fig.tight_layout(pad=2.0)
        fig.figsize=(9,6)
        #fig=Figure(figsize=(5,3),dpi=100)
        #gs = fig.add_gridspec(i, hspace=0)
        #ax=sg.subplots(sharex=True, sharey=True)
        N=int(self.cycles)
        T=float(self.dt)
        
        
        for j in range(0,i):
            x=self.plotWvV[j]
            yf=fft(self.trcs[0:(N+1),(x+1)])
            xf=fftfreq(N, T)[:N//2]
            
            ax[j].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
            #ax[j].plot.xlabel('Time (s)') 
            #ax[j].plot.ylabel('Velocity (m/s)') 
            ax[j].set_title(self.plotWvN[j])
            #ax.show()
            # plotting the graph
        canvas1=Canvas(self.main_frame)
        canvas1=FigureCanvasTkAgg(fig,master= self.top)
        canvas1.draw()
        toolbar = NavigationToolbar2Tk(canvas1,
                                     self.top)
        toolbar.pack()
        toolbar.update()
        canvas1.get_tk_widget().pack(side="top",fill='both',expand=True)
        canvas1.pack(side="top",fill='both',expand=True)
        canvas1.get_tk_widget().pack()   
            

        
        
    def plot(self):
        if filename != None:
            self.update_status("Generating plots...")
            top= Toplevel(window)
            top.title("Waveform Plots")
            main_frame = Frame(top)
            main_frame.pack(fill=BOTH, expand=True)

            fig = Figure(figsize = (9, 6),
                         dpi = 100)
          
            # list of squares
            #y = [i**2 for i in range(101)]
          
            # adding the subplot
            i=self.nhist
            gs = fig.add_gridspec(i, hspace=0)
            plot1=gs.subplots(sharex=True, sharey=True)
            
            for j in range(0,i):
                plot1[j].plot(self.trcs[0:(self.cycles+1),0], self.trcs[0:(self.cycles+1),(j+1)])
            canvas1=Canvas(main_frame)
            canvas1=FigureCanvasTkAgg(fig,master= top)
            canvas1.draw()
            toolbar = NavigationToolbar2Tk(canvas1,
                                     top)
            toolbar.pack()
            toolbar.update()
            canvas1.get_tk_widget().pack(side="top",fill='both',expand=True)
            canvas1.pack(side="top",fill='both',expand=True)
            canvas1.get_tk_widget().pack()
            self.update_status("Plots generated successfully")
        else:
            messagebox.showerror('Error', 'Error: No file found!')
            self.update_status("Error: No file loaded")
        
        
        
    def open_popup(self):
        if filename != None:
            top= Toplevel(window)
            main_frame = Frame(top)
            main_frame.pack(fill=BOTH, expand=1)
            # Create A Canvas
            my_canvas = Canvas(main_frame)
            my_canvas.pack(side=LEFT, fill=BOTH, expand=1)
            # Add A Scrollbar To The Canvas
            my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
            my_scrollbar.pack(side=RIGHT, fill=Y)
            # Configure The Canvas
            my_canvas.configure(yscrollcommand=my_scrollbar.set)
            my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))
            # Create ANOTHER Frame INSIDE the Canvas
            second_frame = Frame(my_canvas)
            # Add that New frame To a Window In The Canvas
            my_canvas.create_window((0,0), window=second_frame, anchor="nw")
            selected={}
            for x in range(0,self.nhist):
                is_selected = BooleanVar()
                l=Checkbutton(second_frame, text=self.allhst[x][0], variable=is_selected, onvalue=1, offvalue=0).grid(row=x, column=0, pady=10, padx=10)
                selected[x]=is_selected
            self.selected=selected
            Button(second_frame, command = self.Wvselection, text='OK').grid(row=0, column=1, pady=5, padx=5)
            self.top=top
        else:
            messagebox.showerror('Error', 'Error: No file found!')
   
    def Wvselection(self):
        plotWvN=[]
        plotWvV=[]
        allhst=self.allhst
        for key, value in self.allhst.items():
            if self.selected[key].get():
                plotWvN.append(value)
                plotWvV.append(key)
        if plotWvV !=[]:
            self.plotWvV=plotWvV
            self.plotWvN=plotWvN
        self.top.destroy()
        WFile.plot_sel(self)
           
            
        
    def lowpassF(self):
        topW=Toplevel(window)
        topW.geometry('400x400')
        main_frame = Frame(topW)
        main_frame.pack(fill=BOTH, expand=True)
        e=Entry(topW, width=50)
        e.pack()
        Button(main_frame, command = self.lowpassFilter, text='OK').grid(row=0, column=1, pady=5, padx=5)
        
        self.L=e.get()
        
        #Button(second_frame, command = self.lowpassFilter, text='OK').grid(row=0, column=1, pady=5, padx=5)
        
     
    def plot_sel(self):
        top= Toplevel(window)
        top.geometry('400x400')
        self.main_frame = Frame(top)
        self.main_frame.pack(fill=BOTH, expand=True)
        my_menu=Menu(top)
        top.config(menu=my_menu)
        file_menu=Menu(my_menu)
        my_menu.add_cascade(label='Data', menu=file_menu)
        self.top=top
        file_menu.add_command(label='Original', command=self.original)
        file_menu.add_command(label='FFT', command=self.fourier)
        file_menu.add_command(label='More Waveforms', command=self.open_popup)
        file_menu.add_command(label='Exit', command=top.quit)

        filter_menu=Menu(my_menu)
        my_menu.add_cascade(label='Filter', menu=filter_menu)
        #file_menu.add_command(label='Highpass Filter', command=self.highpassF)
        filter_menu.add_command(label='Lowpass Filter', command=self.lowpassF)
        #file_menu.add_command(label='Bandpass Filter', command=self.bandpassF)

   
    def export(self):
        if filename != None:
            data=self.trcs[:,1:self.nhist]
            file = fd.asksaveasfile(filetypes=[('text files','*.csv')], defaultextension='.csv')
            pd.DataFrame(data).to_csv(file, index=False, header=False, lineterminator='\n')
        else:
            messagebox.showerror('Error', 'Error: No file found!')


    def hide_all_frames(self):
        for widge in self.main_frame.winfo_children():
            widge.destroy()
        self.main_frame.pack_forget()
    
        
    def Model_file(self):
        #c=mFile(self)
        #c.mFile
        
        top= Toplevel(window)
        top.geometry("1000x700")
        top.title("Model geometry")
        self.main_frame = Frame(top)
        self.main_frame.pack(fill=BOTH, expand=True)
        my_menu=Menu(top)
        top.config(menu=my_menu)
        file_menu=Menu(my_menu)
        my_menu.add_cascade(label='Data', menu=file_menu)
        self.top=top
        file_menu.add_command(label='Read model File', command=self.open_hist_file)
        file_menu.add_command(label='Plot model Geometry', command=mFile.getModelData(filename))
        file_menu.add_command(label='Exit', command=top.quit)
        # #top= Toplevel(window)
        # main_frame = Frame(top)
        # main_frame.pack(fill=BOTH, expand=1)
        # # Create A Canvas
        # my_canvas = Canvas(main_frame)
        # my_canvas.pack(side=LEFT, fill=BOTH, expand=1)
        # # Add A Scrollbar To The Canvas
        # #my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
        # #my_scrollbar.pack(side=RIGHT, fill=Y)
        # # Configure The Canvas
        # #my_canvas.configure(yscrollcommand=my_scrollbar.set)
        # #my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))
        # # Create ANOTHER Frame INSIDE the Canvas
        # second_frame = Frame(my_canvas)
        # # Add that New frame To a Window In The Canvas
        # my_canvas.create_window((0,0), window=second_frame, anchor="nw")
        
        # #selected={}
        # #self.selected=selected
        # #Button(second_frame, command = self.Wvselection, text='OK').grid(row=0, column=1, pady=5, padx=5)
        # self.top=top
        
        # Button(second_frame,text='Read model File',
        #                               height = 2, 
        #                               width = 20)
        #                               #command=mFile.mFile(open_text_file))
        # #self.open_button.place(x=10, y=600)
        
        # Button(second_frame,text = "Plot model Geometry",
        #                           height = 2, 
        #                           width = 20)
                                  #command = mFile.mFile(getModelData),)
        #self.plot_button.place(x=170, y=600)
        
        # mFile.mFile()
 
                
 
    def open_hist_file(self):
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
        if filename:
            self.update_status("Loading file: " + os.path.basename(filename))
            WFile.getModelData(self)
            self.update_status("File loaded successfully")
        else:
            self.update_status("No file selected")
        
    def getModelData(self):
        if filename:
            filenameM=filename.split('.hst')[0] + '.m'
            WFile.readMfile(self,filenameM)
            WFile.read_W_file(self,filename)
            

    
e= WFile(window)
    
global filename
filename= None     
              
# run the gui
window.mainloop()                       
       
