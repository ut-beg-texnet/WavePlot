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
import sys
import logging
import argparse
from datetime import datetime

# Setup logging configuration
def setup_logging(debug_mode=False):
    """Configure logging with appropriate level and format"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'waveviewer_{timestamp}.log')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress PIL/Pillow debugging messages
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)

    # Also suppress PIL.PngImagePlugin specifically
    png_logger = logging.getLogger('PIL.PngImagePlugin')
    png_logger.setLevel(logging.WARNING)

    # Suppress matplotlib font debugging messages
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)
    
    # Also suppress matplotlib.font_manager logs
    font_logger = logging.getLogger('matplotlib.font_manager')
    font_logger.setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {log_file}")
    return logger

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='WavePlot - Waveform Plotting Tool')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging mode')
    return parser.parse_args()

# Initialize logging based on command line arguments
args = parse_arguments()
logger = setup_logging(args.debug)

# Allow importing utilities from the 'python_code' directory (might move all Python code there later)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_code'))
try:
    from read_map_file import (
        read_map_summary,
        try_resolve_hst_from_map,
        try_resolve_m_from_map,
        map_to_base_filename,
    )
    logger.info("Successfully imported map file utilities")
except Exception as e:
    # Fallback if utility not present; .map option will show an error
    read_map_summary = None
    logger.warning(f"Map file utilities not available: {e}")

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
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing WaveViewer GUI")
        
        try:
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
                                         text='📁 Open .hst File',
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

            self.open_map_button = ttk.Button(button_frame,
                                              text='🗺️ Open .map File',
                                              style='Modern.TButton',
                                              command=self.open_map_file)
            self.open_map_button.grid(row=1, column=2, padx=10, pady=10, sticky='ew')
            
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
            
            self.logger.info("GUI initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GUI: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize GUI: {e}")
            raise
        
    def update_status(self, message):
        """Update the status bar with a message"""
        try:
            self.status_label.config(text=message)
            self.status_label.update()
            self.logger.debug(f"Status updated: {message}")
        except Exception as e:
            self.logger.error(f"Failed to update status: {e}")
        
        
# read model file to identify the number of cycles, the timestep and the
# the velocity histories to use the results in order to make it available for 
# the user to select the waveforms need to plot        
    def readMfile(self, filenameM):
        """Read model file to extract simulation parameters and history variables"""
        self.logger.info(f"Reading model file: {filenameM}")
        
        try:
            if not os.path.exists(filenameM):
                raise FileNotFoundError(f"Model file not found: {filenameM}")
                
            with open(filenameM, "r") as readfile:
                allhis = []
                
                for line_num, line in enumerate(readfile, 1):
                    try:
                        Type = line.split("\n")
                        x = Type[0].split(' ')
                        
                        if 'tstp=' in x[0]:
                            i = [i for i in x if i.startswith('tstp')]
                            if i:
                                i = i[0].split('=')
                                if len(i) > 1:
                                    self.dt = float(i[1])
                                    self.logger.debug(f"Found timestep: {self.dt}")
                                else:
                                    self.dt = 3.6e-4  # Default value
                                    self.logger.warning("Using default timestep value")
                                    
                        elif x[0] == 'cy' or x[0] == 'cycles':    
                            if len(x) > 1:
                                self.cycles = int(x[1])
                                self.logger.debug(f"Found cycles: {self.cycles}")
                            else:
                                self.logger.warning(f"Invalid cycles line at {line_num}: {line.strip()}")
                                
                        elif x[0] == 'his':
                            # Extract different velocity components
                            for vel_type in ['yvel', 'xvel', 'zvel']:
                                vel_list = [i for i in x if i.startswith(vel_type)]
                                if vel_list:
                                    allhis.append(vel_list)
                                    self.logger.debug(f"Found {vel_type} histories: {len(vel_list)}")
                                    
                            # Extract other history types
                            hismx = [i for i in x if i.startswith('mx_')]
                            if hismx:
                                allhis.append(hismx)
                                self.logger.debug(f"Found mx histories: {len(hismx)}")
                                
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing line {line_num} in {filenameM}: {e}")
                        continue
                        
                if allhis:
                    allhst = {}
                    for index, element in enumerate(allhis):
                        allhst[index] = element
                    self.allhst = allhst
                    self.logger.info(f"Successfully parsed {len(allhis)} history groups from model file")
                else:
                    self.logger.warning("No history variables found in model file")
                    
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            messagebox.showerror("File Error", f"Model file not found: {filenameM}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading model file {filenameM}: {e}")
            messagebox.showerror("Read Error", f"Error reading model file: {e}")
            raise
                
                
# read the history files from the '.hst' format to make it available for plot of export to csv            
    def read_W_file(self, filename):
        """Read history files from .hst format for plotting and export"""
        self.logger.info(f"Reading history file: {filename}")
        
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"History file not found: {filename}")
                
            self.viewer = 1
            
            # Read binary file
            with open(filename, 'rb') as f: 
                traces = np.fromfile(f, np.float32)
                self.logger.debug(f"Read {len(traces)} float32 values from file")
                
            # Set default timestep if not already set
            if not hasattr(self, 'dt'):
                self.dt = 3.6e-4
                self.logger.warning("Using default timestep value")
                
            # Set default cycles if not already set  
            if not hasattr(self, 'cycles'):
                # Try to estimate cycles from file size
                traces_subset = traces[1::3]
                estimated_cycles = int(np.sqrt(len(traces_subset))) - 1
                self.cycles = max(estimated_cycles, 100)  # Minimum reasonable value
                self.logger.warning(f"Estimated cycles from file size: {self.cycles}")
                
            # Create time axis
            tax = np.arange(start=0, stop=(self.dt*(self.cycles+1)), step=self.dt)
            traces = traces[1::3]  # Extract every third element

            nhist = int(len(traces) / (self.cycles + 1))
            self.logger.debug(f"Calculated nhist: {nhist}, cycles: {self.cycles}")
            
            if nhist <= 0:
                raise ValueError(f"Invalid number of histories calculated: {nhist}")
                
            # Initialize arrays
            self.trcs = np.zeros((self.cycles + 1, nhist + 1))
            n_plot = 1
            n_plot_temp = n_plot
            
            if np.remainder(nhist, n_plot) != 0: 
                n_iteration = int(np.rint(nhist / n_plot) + 1)
            else: 
                n_iteration = int(np.rint(nhist / n_plot))
                
            self.logger.debug(f"Processing {n_iteration} iterations")
            
            start = 0
            stop = self.cycles + 1
            
            for j in range(1, n_iteration + 1): 
                try:
                    k = n_plot * (j - 1)
                    
                    if stop > len(traces):
                        self.logger.warning(f"Truncating at iteration {j} due to insufficient data")
                        break
                        
                    trace = traces[start:stop]
                    self.trcs[0:(self.cycles+1), int(k+1)] = trace
                    
                    if j == np.rint(nhist / n_plot) + 1: 
                        n_plot = np.remainder(nhist, n_plot)
                        
                    n_plot = n_plot_temp
                    start += self.cycles + 1
                    stop += self.cycles + 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing iteration {j}: {e}")
                    break
                    
            # Set time axis
            self.trcs[0:(self.cycles+1), 0] = np.transpose(tax)
            self.nhist = nhist
            
            self.logger.debug(f"Successfully read history file with {nhist} histories and {self.cycles} cycles")
            
        except FileNotFoundError as e:
            self.logger.error(f"History file not found: {e}")
            messagebox.showerror("File Error", f"History file not found: {filename}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading history file {filename}: {e}")
            messagebox.showerror("Read Error", f"Error reading history file: {e}")
            raise
     
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
        """Plot all waveforms in a new window"""
        self.logger.info("Plotting all waveforms")
        
        try:
            if filename is None:
                error_msg = "No file loaded. Please open a .hst file first."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                self.update_status("Error: No file loaded")
                return
                
            if not hasattr(self, 'trcs') or not hasattr(self, 'nhist'):
                error_msg = "No waveform data available. Please load a file first."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                self.update_status("Error: No data available")
                return
                
            self.update_status("Generating plots...")
            
            # Create new window
            top = Toplevel(window)
            top.title("Waveform Plots")
            main_frame = Frame(top)
            main_frame.pack(fill=BOTH, expand=True)

            # Create figure
            fig = Figure(figsize=(9, 6), dpi=100)
            
            i = self.nhist
            self.logger.debug(f"Creating plots for {i} histories")
            
            if i == 0:
                raise ValueError("No histories to plot")
                
            # Create subplots
            gs = fig.add_gridspec(i, hspace=0)
            
            if i == 1:
                plot1 = [gs.subplots()]
            else:
                plot1 = gs.subplots(sharex=True, sharey=True)
            
            # Plot each waveform
            for j in range(i):
                try:
                    if i == 1:
                        ax = plot1[0]
                    else:
                        ax = plot1[j]
                        
                    ax.plot(self.trcs[0:(self.cycles+1), 0], 
                           self.trcs[0:(self.cycles+1), (j+1)])
                    ax.set_title(f"Waveform {j+1}")
                    self.logger.debug(f"Plotted waveform {j+1}")
                    
                except Exception as e:
                    self.logger.error(f"Error plotting waveform {j+1}: {e}")
                    continue
                    
            # Create canvas and toolbar
            canvas1 = FigureCanvasTkAgg(fig, master=top)
            canvas1.draw()
            
            toolbar = NavigationToolbar2Tk(canvas1, top)
            toolbar.pack()
            toolbar.update()
            
            canvas1.get_tk_widget().pack(side="top", fill='both', expand=True)
            
            self.update_status("Plots generated successfully")
            self.logger.info("All waveforms plotted successfully")
            
        except Exception as e:
            error_msg = f"Error generating plots: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('Plot Error', error_msg)
            self.update_status("Error generating plots")
        
        
        
    def open_popup(self):
        """Open popup window for waveform selection"""
        self.logger.info("Opening waveform selection popup")
        
        try:
            if filename is None:
                error_msg = "No file loaded. Please open a .hst file first."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                return
                
            if not hasattr(self, 'allhst') or not self.allhst:
                error_msg = "No history variables found. Please check your .m file."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                return
                
            top = Toplevel(window)
            top.title("Select Waveforms")
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
            my_canvas.create_window((0,0), window=second_frame, anchor="nw")
            
            selected = {}
            
            # Create checkboxes for each history variable group
            row = 0
            for hist_index, hist_vars in self.allhst.items():
                self.logger.debug(f"Creating checkbox for history group {hist_index}: {hist_vars}")
                
                # Create a checkbox for each variable in the group
                for var_index, var_name in enumerate(hist_vars):
                    is_selected = BooleanVar()
                    checkbox = Checkbutton(second_frame, 
                                            text=var_name, 
                                            variable=is_selected, 
                                            onvalue=1, 
                                            offvalue=0)
                    checkbox.grid(row=row, column=0, pady=5, padx=10, sticky='w')
                    
                    # Store selection state with a unique key
                    key = f"{hist_index}_{var_index}"
                    selected[key] = is_selected
                    row += 1
            
            self.selected = selected
            self.logger.debug(f"Created {len(selected)} waveform selection checkboxes")
            
            # Add OK button
            Button(second_frame, command=self.Wvselection, text='OK').grid(row=row, column=1, pady=5, padx=5)
            self.top = top
            
        except Exception as e:
            error_msg = f"Error opening waveform selection: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('Error', error_msg)
   
    def Wvselection(self):
        """Process waveform selections and create plots"""
        self.logger.info("Processing waveform selections")
        
        try:
            plotWvN = []
            plotWvV = []
            
            # Process selections based on the new key format
            for key, is_selected in self.selected.items():
                if is_selected.get():
                    # Parse the key to get hist_index and var_index
                    hist_index, var_index = key.split('_')
                    hist_index = int(hist_index)
                    var_index = int(var_index)
                    
                    # Get the variable name and add to selection
                    var_name = self.allhst[hist_index][var_index]
                    plotWvN.append([var_name])  # Keep as list to match original format
                    plotWvV.append(hist_index)  # Use hist_index as the waveform index
                    
                    self.logger.debug(f"Selected waveform: {var_name} (index: {hist_index})")
            
            if plotWvV:
                self.plotWvV = plotWvV
                self.plotWvN = plotWvN
                self.logger.info(f"Selected {len(plotWvV)} waveforms for plotting")
                self.top.destroy()
                WFile.plot_sel(self)
            else:
                self.logger.warning("No waveforms selected")
                messagebox.showwarning("No Selection", "Please select at least one waveform to plot.")
                
        except Exception as e:
            error_msg = f"Error processing waveform selection: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('Selection Error', error_msg)
           
            
        
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
        """Export waveform data to CSV file"""
        self.logger.info("Exporting waveform data to CSV")
        
        try:
            if filename is None:
                error_msg = "No file loaded. Please open a .hst file first."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                self.update_status("Error: No file loaded")
                return
                
            if not hasattr(self, 'trcs') or not hasattr(self, 'nhist'):
                error_msg = "No waveform data available. Please load a file first."
                self.logger.error(error_msg)
                messagebox.showerror('Error', error_msg)
                self.update_status("Error: No data available")
                return
                
            self.update_status("Exporting data...")
            
            # Prepare data for export (exclude time column)
            data = self.trcs[:, 1:self.nhist+1]
            self.logger.debug(f"Exporting data shape: {data.shape}")
            
            # Open file dialog
            file = fd.asksaveasfile(filetypes=[('CSV files', '*.csv')], 
                                 defaultextension='.csv')
            
            if file is None:
                self.logger.info("Export cancelled by user")
                self.update_status("Export cancelled")
                return
                
            # Export to CSV
            df = pd.DataFrame(data)
            df.to_csv(file, index=False, header=False, lineterminator='\n')
            
            self.logger.info(f"Data exported successfully to: {file.name}")
            self.update_status("Data exported successfully")
            messagebox.showinfo("Export Complete", 
                              f"Data exported successfully to:\n{file.name}")
            
        except Exception as e:
            error_msg = f"Error exporting data: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('Export Error', error_msg)
            self.update_status("Error exporting data")


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
        """Open and load history (.hst) or model (.m) files"""
        self.logger.info("Opening file dialog for history/model files")
        
        try:
            # Define file types
            filetypes = (
                ('History files', '*.hst'),
                ('Model files', '*.m'),
                ('All files', '*.*')
            )
            
            # Show file dialog
            f = fd.askopenfilename(filetypes=filetypes)
            
            if not f:
                self.logger.info("No file selected by user")
                self.update_status("No file selected")
                return
                
            global filename
            filename = str(f)
            
            self.logger.info(f"Selected file: {filename}")
            self.update_status("Loading file: " + os.path.basename(filename))
            
            # Load the file data
            WFile.getModelData(self)
            
            self.update_status("File loaded successfully")
            self.logger.info("File loaded successfully")
            
        except Exception as e:
            error_msg = f"Error opening file: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('File Error', error_msg)
            self.update_status("Error loading file")
        
    def getModelData(self):
        """Load model and history data from files"""
        self.logger.info("Loading model and history data")
        
        try:
            global filename
            if not filename:
                raise ValueError("No filename specified")
                
            # Determine model file path
            if filename.endswith('.hst'):
                filenameM = filename.split('.hst')[0] + '.m'
            elif filename.endswith('.m'):
                filenameM = filename
                # For .m files, we need to find corresponding .hst file
                hist_file = filename.split('.m')[0] + '.hst'
                if os.path.exists(hist_file):
                    filename = hist_file
                else:
                    self.logger.warning("No corresponding .hst file found for .m file")
                    return
            else:
                self.logger.warning(f"Unsupported file type: {filename}")
                return
                
            self.logger.debug(f"Model file: {filenameM}")
            self.logger.debug(f"History file: {filename}")
            
            # Read model file if it exists
            if os.path.exists(filenameM):
                WFile.readMfile(self, filenameM)
            else:
                self.logger.warning(f"Model file not found: {filenameM}")
                # Set default values
                self.dt = 3.6e-4
                self.cycles = 1000
                
            # Read history file
            WFile.read_W_file(self, filename)
            
            self.logger.info("Model and history data loaded successfully")
            
        except Exception as e:
            error_msg = f"Error loading model data: {e}"
            self.logger.error(error_msg)
            messagebox.showerror('Load Error', error_msg)
            raise
            

    def open_map_file(self):
        """Open a .map file, resolve matching .m and .hst, load if possible."""
        filetypes = (
            ('map files', '*.map'),
            ('All files', '*.*')
        )
        f = fd.askopenfilename(filetypes=filetypes)
        path = str(f)
        if not path:
            self.update_status("No file selected")
            return
        if read_map_summary is None:
            messagebox.showerror('Error', 'MAP support module not available.')
            return
        try:
            self.update_status("Reading .map summary ...")
            summary = read_map_summary(path)
            base = map_to_base_filename(path)
            m_path = try_resolve_m_from_map(path)
            hst_path = try_resolve_hst_from_map(path)

            details = f"snap:{summary.snapshots} hist:{summary.histories} dump:{summary.dumps} geo:{summary.geometries}"
            self.update_status(f".map OK ({details})")

            # Prefer loading via resolved .hst like the existing flow
            if hst_path and os.path.exists(hst_path):
                global filename
                filename = hst_path
                # If .m exists, parse it for dt/cycles/wave labels
                if m_path and os.path.exists(m_path):
                    WFile.readMfile(self, m_path)
                WFile.read_W_file(self, hst_path)
                self.update_status("Loaded via MAP → HST successfully")
            else:
                messagebox.showinfo('MAP Summary',
                                    f"Found .map file but matching .hst not found.\n\n"+
                                    f"MAP summary: {details}\nBase: {base}")
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read .map: {e}')
            self.update_status('Error reading .map')

    
try:
    logger.info("Starting WavePlot application")
    e = WFile(window)
    
    global filename
    filename = None     
                  
    # run the gui
    logger.info("Starting GUI main loop")
    window.mainloop()
    logger.info("Application closed")
    
except Exception as e:
    logger.critical(f"Critical error in application startup: {e}")
    messagebox.showerror("Critical Error", f"Failed to start application: {e}")
    sys.exit(1)                       
       
