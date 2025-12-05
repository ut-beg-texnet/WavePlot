"""Tkinter GUI for the WavePlot Python port.

This module provides a graphical user interface for visualizing WAVE project data.
The GUI allows users to browse snapshots, histories, and dumps, plot them with
various colormaps, overlay geometry, and evaluate history formulas.
"""

from __future__ import annotations

import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, List, Optional, Tuple

from matplotlib import cm
from matplotlib import patches
import numpy as np
from matplotlib.widgets import TextBox, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import ttk

# Try to use modern Matplotlib colormap registry (3.5+)
try:
    from matplotlib import colormaps as _cm_registry  # type: ignore
except Exception:  # pragma: no cover - fallback for older Matplotlib
    _cm_registry = None

from ..geometry_loader import GeometryOverlay
from ..project import FormulaResult, WaveProject
from ..map_parser import DumpMapRecord, HistMapRecord, SnapMapRecord
from ..logger import get_logger


def _available_colormaps() -> List[str]:
    """Get list of available colormap names, excluding reversed variants.
    
    Returns:
        Sorted list of colormap names (e.g., 'viridis', 'plasma', 'coolwarm')
        Reversed colormaps (ending in '_r') are excluded to avoid clutter.
    """
    if _cm_registry is not None:
        names = list(_cm_registry)
    else:
        # Fallback for Matplotlib < 3.5
        try:
            names = sorted(cm.cmap_d.keys())  # type: ignore[attr-defined]
        except AttributeError:
            names = sorted(cm.datad.keys())  # type: ignore[attr-defined]
    return sorted(name for name in names if not str(name).endswith('_r'))


class WavePlotApp(tk.Tk):
    """Main Tkinter application window for WavePlot.
    
    This class provides the complete GUI interface including:
    - File browser for selecting .map files
    - Tree view for browsing snapshots, histories, and dumps
    - Matplotlib canvas for plotting data
    - Controls for colormap selection, dump component selection, and color scale
    - Formula evaluation and filtering capabilities
    - Geometry overlay visualization
    """

    def __init__(self, map_path: Optional[str] = None, debug: bool = False) -> None:
        """Initialize the WavePlot application window.
        
        Args:
            map_path: Optional path to a .map file to load on startup
            debug: If True, enable debug logging
        """
        super().__init__()
        self.title("WavePlot")
        self.geometry("1200x720")
        self.minsize(900, 600)
        self.configure(bg="#f4f6fb")

        # Logger instance (None if debug is disabled)
        self.logger = get_logger() if debug else None
        
        # Current project (None until a file is loaded)
        self.project: Optional[WaveProject] = None
        # Mapping: tree_item_id -> (record_type, index)
        self._tree_map: Dict[str, Tuple[str, int]] = {}
        # Current selection: (data_type, list_of_indices)
        self._current_selection: Optional[Tuple[str, List[int]]] = None
        # UI state variables
        self._current_cmap = tk.StringVar(value='viridis')
        self._dump_component = tk.StringVar(value='0')
        # Matplotlib artists and widgets for cleanup
        self._colorbars: List[object] = []
        self._image_artist: Optional[object] = None
        self._cb_widget_axes: List[object] = []  # Colorbar widget axes
        self._vmin_box: Optional[TextBox] = None
        self._vmax_box: Optional[TextBox] = None
        self._auto_btn: Optional[Button] = None
        self._overlay_artists: List[object] = []  # Geometry overlay patches
        # Color scale lock state
        self._lock_color_scale: bool = False
        self._locked_vmin: Optional[float] = None
        self._locked_vmax: Optional[float] = None
        self._lock_btn: Optional[Button] = None

        if self.logger:
            self.logger.info("WavePlot application starting")
        
        self._build_ui()
        
        # Bind cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        if map_path:
            self.load_project(map_path)

    # UI construction ------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu()
        container = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        self.tree_frame = ttk.Frame(container, padding=10)
        self.detail_frame = ttk.Frame(container, padding=(10, 10, 10, 5))
        container.add(self.tree_frame, weight=1)
        container.add(self.detail_frame, weight=3)

        self._build_tree_panel()
        self._build_detail_panel()

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open .map...", command=self._open_map_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu)

    def _build_tree_panel(self) -> None:
        header = ttk.Label(self.tree_frame, text="Datasets", font=('Segoe UI', 12, 'bold'))
        header.pack(anchor=tk.W, pady=(0, 8))

        self.tree = ttk.Treeview(
            self.tree_frame,
            show='tree',
            selectmode='extended',
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        self.formula_button = ttk.Button(
            self.tree_frame,
            text="Combine Histories",
            command=self._prompt_formula,
            state=tk.DISABLED,
        )
        self.formula_button.pack(fill=tk.X, pady=(8, 0))

        self.filter_button = ttk.Button(
            self.tree_frame,
            text="Filter History",
            command=self._prompt_filter,
            state=tk.DISABLED,
        )
        self.filter_button.pack(fill=tk.X, pady=(4, 0))

        self.export_hist_button = ttk.Button(
            self.tree_frame,
            text="Export Histogram Data CSV",
            command=self._export_histories_csv,
            state=tk.DISABLED,
        )
        self.export_hist_button.pack(fill=tk.X, pady=(4, 0))

    def _build_detail_panel(self) -> None:
        """Build the detail/plotting panel.
        
        Creates the right panel containing:
        - Colormap selector dropdown
        - Dump component selector (for multi-variable dumps)
        - Multi-variable slider (for browsing multiple snapshots)
        - Matplotlib figure canvas with toolbar
        - Metadata text display area
        """
        top_bar = ttk.Frame(self.detail_frame)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top_bar, text="Colour map:").pack(side=tk.LEFT)
        cmap_values = _available_colormaps()
        cmap_box = ttk.Combobox(
            top_bar,
            textvariable=self._current_cmap,
            values=cmap_values,
            width=16,
            state='readonly',
        )
        cmap_box.pack(side=tk.LEFT, padx=(4, 16))
        def _on_cmap_change(_event: tk.Event) -> None:
            if self.logger:
                self.logger.info(f"Colormap changed to: {self._current_cmap.get()}")
            self._refresh_plot()
        cmap_box.bind('<<ComboboxSelected>>', _on_cmap_change)

        ttk.Label(top_bar, text="Dump component:").pack(side=tk.LEFT)
        self.dump_combo = ttk.Combobox(
            top_bar,
            textvariable=self._dump_component,
            values=["0"],
            width=6,
            state='readonly',
        )
        self.dump_combo.pack(side=tk.LEFT, padx=(4, 16))
        def _on_dump_component_change(_event: tk.Event) -> None:
            if self.logger:
                self.logger.info(f"Dump component changed to: {self._dump_component.get()}")
            self._refresh_plot()
        self.dump_combo.bind('<<ComboboxSelected>>', _on_dump_component_change)

        self.multi_var = tk.IntVar(value=0)
        def _on_slider_change(value: str) -> None:
            if self.logger:
                self.logger.debug(f"Multi-snapshot slider moved to: {value}")
            self._refresh_plot()
        self.multi_slider = ttk.Scale(
            top_bar,
            from_=0,
            to=0,
            variable=self.multi_var,
            command=_on_slider_change,
        )
        self.multi_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.multi_slider.configure(state='disabled')

        # Note: min/max controls are embedded near the colorbar (not in top bar)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.detail_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.detail_frame)
        self.toolbar.update()

        self.meta_text = tk.Text(
            self.detail_frame,
            height=6,
            wrap='word',
            state=tk.DISABLED,
            background='#f5f7fb',
            relief=tk.FLAT,
        )
        self.meta_text.pack(fill=tk.X, pady=(8, 0))

    # Project loading ------------------------------------------------------

    def load_project(self, map_path: str) -> None:
        """Load a WAVE project from a .map file.
        
        Parses the map file, initializes the project, and populates the tree view
        with available snapshots, histories, and dumps.
        
        Args:
            map_path: Path to the .map file to load
        """
        if self.logger:
            self.logger.info(f"Loading project: {map_path}")
        try:
            self.project = WaveProject(map_path)
            if self.logger:
                self.logger.info(f"Successfully loaded project: {os.path.basename(map_path)}")
                # Log project summary
                snap_count = len(self.project.snapshot_records())
                hist_count = len(self.project.history_records())
                dump_count = len(self.project.dump_records())
                self.logger.debug(f"Project contains: {snap_count} snapshots, {hist_count} histories, {dump_count} dumps")
        except Exception as exc:
            if self.logger:
                self.logger.error(f"Failed to load project: {map_path}", exc_info=True)
            messagebox.showerror("WavePlot", f"Failed to load project:\n{exc}")
            return
        self._populate_tree()
        self._status_message(f"Loaded {os.path.basename(map_path)}")

    def _open_map_dialog(self) -> None:
        if self.logger:
            self.logger.info("Opening file dialog")
        path = filedialog.askopenfilename(
            title="Select .map file",
            filetypes=[('WavePlot map', '*.map'), ('All files', '*.*')],
        )
        if path:
            if self.logger:
                self.logger.info(f"User selected file: {path}")
            self.load_project(path)
        elif self.logger:
            self.logger.debug("File dialog cancelled by user")

    def _populate_tree(self) -> None:
        for item in self.tree.get_children(''):
            self.tree.delete(item)
        self._tree_map.clear()
        if not self.project:
            return
        snaps_node = self.tree.insert('', 'end', text='Snapshots')
        snap_count = 0
        for rec in self.project.snapshot_records():
            desc = (
                f"#{rec.index} {rec.variable} | {rec.x_qty}x{rec.y_qty} | "
                f"t={rec.time_end:.3f}s"
            )
            node = self.tree.insert(snaps_node, 'end', text=desc)
            self._tree_map[node] = ('snap', rec.index - 1)
            snap_count += 1
        hists_node = self.tree.insert('', 'end', text='Histories')
        hist_count = 0
        for rec in self.project.history_records():
            desc = (
                f"#{rec.index} {rec.variable} | samples={rec.sample_qty} | "
                f"dt={rec.dt:.3e}"
            )
            node = self.tree.insert(hists_node, 'end', text=desc)
            self._tree_map[node] = ('hist', rec.index - 1)
            hist_count += 1
        if hist_count > 0:
            self.export_hist_button.configure(state=tk.NORMAL)
        else:
            self.export_hist_button.configure(state=tk.DISABLED)
        dumps_node = self.tree.insert('', 'end', text='Dumps')
        dump_count = 0
        for rec in self.project.dump_records():
            desc = (
                f"#{rec.index} | grid={rec.i_qty}x{rec.j_qty}x{max(rec.k_qty,1)} | "
                f"vars={rec.v_qty}"
            )
            node = self.tree.insert(dumps_node, 'end', text=desc)
            self._tree_map[node] = ('dump', rec.index - 1)
            dump_count += 1
        self.tree.item(snaps_node, open=True)
        self.tree.item(hists_node, open=True)
        self.tree.item(dumps_node, open=False)
        if self.logger:
            self.logger.debug(f"Populated tree: {snap_count} snapshots, {hist_count} histories, {dump_count} dumps")

    # Tree interactions ----------------------------------------------------

    def _on_tree_select(self, _event: tk.Event) -> None:
        if not self.project:
            return
        selection = [item for item in self.tree.selection() if item in self._tree_map]
        if not selection:
            if self.logger:
                self.logger.debug("Tree selection cleared")
            return
        types = {self._tree_map[item][0] for item in selection}
        if len(types) > 1:
            if self.logger:
                self.logger.debug("User selected multiple types, showing warning")
            messagebox.showinfo("WavePlot", "Please select entries from one group at a time.")
            return
        selected_type = next(iter(types))
        indices = sorted(self._tree_map[item][1] for item in selection)
        self._current_selection = (selected_type, indices)
        if self.logger:
            self.logger.info(f"Tree selection: {selected_type} {indices}")
        self._update_controls(selected_type, len(indices))
        self._refresh_plot()

    def _update_controls(self, data_type: str, count: int) -> None:
        if self.logger:
            self.logger.debug(f"Updating controls for {data_type} (count={count})")
        if data_type == 'dump':
            record = self.project.dump_records()[self._current_selection[1][0]]
            values = [str(i) for i in range(record.v_qty)]
            self.dump_combo.configure(values=values, state='readonly')
            if self._dump_component.get() not in values:
                self._dump_component.set(values[0])
            if self.logger:
                self.logger.debug(f"Dump component selector enabled with {len(values)} components")
        else:
            self.dump_combo.configure(values=["0"], state='disabled')
        if data_type == 'snap' and count > 1:
            self.multi_slider.configure(state='normal', from_=0, to=count - 1)
            self.multi_var.set(0)
            if self.logger:
                self.logger.debug(f"Multi-snapshot slider enabled (range 0-{count-1})")
        else:
            self.multi_slider.configure(state='disabled', from_=0, to=0)
            self.multi_var.set(0)
        if data_type == 'hist' and count >= 2:
            self.formula_button.configure(state=tk.NORMAL)
        else:
            self.formula_button.configure(state=tk.DISABLED)
        if data_type == 'hist' and count == 1:
            self.filter_button.configure(state=tk.NORMAL)
        else:
            self.filter_button.configure(state=tk.DISABLED)

    # Plotting -------------------------------------------------------------

    def _refresh_plot(self) -> None:
        self._clear_colorbars()
        self._clear_overlay_artists()
        self.axes.clear()
        self._image_artist = None
        if not self.project or not self._current_selection:
            if self.logger:
                self.logger.debug("Plot refresh skipped: no project or selection")
            self.canvas.draw_idle()
            return
        data_type, indices = self._current_selection
        if self.logger:
            self.logger.info(f"Refreshing plot: {data_type} {indices}")
        if data_type == 'snap':
            self._plot_snapshot(indices)
        elif data_type == 'hist':
            self._plot_history(indices)
        elif data_type == 'dump':
            self._plot_dump(indices[0])
        self.canvas.draw_idle()

    def _plot_snapshot(self, indices: List[int]) -> None:
        """Plot snapshot data as a 2D image with optional geometry overlay.
        
        Args:
            indices: List of snapshot indices (uses slider position if multiple)
        """
        # Select snapshot based on slider if multiple selected
        idx = indices[int(self.multi_var.get())] if len(indices) > 1 else indices[0]
        record = self.project.snapshot_records()[idx]
        if self.logger:
            self.logger.info(f"Plotting snapshot #{record.index} ({record.variable})")
        data = self.project.snapshot_array(idx)
        if self.logger:
            self.logger.debug(f"Snapshot data shape: {data.shape}, range: [{data.min():.3e}, {data.max():.3e}]")
        cmap = cm.get_cmap(self._current_cmap.get())
        # Use locked color scale if enabled
        imshow_kwargs = {
            'origin': 'lower',
            'cmap': cmap,
            'aspect': 'auto',
        }
        if self._lock_color_scale and self._locked_vmin is not None and self._locked_vmax is not None:
            imshow_kwargs['vmin'] = self._locked_vmin
            imshow_kwargs['vmax'] = self._locked_vmax
            if self.logger:
                self.logger.debug(f"Using locked color scale: vmin={self._locked_vmin:.3e}, vmax={self._locked_vmax:.3e}")
        # Transpose data for correct orientation (imshow expects row-major)
        im = self.axes.imshow(data.T, **imshow_kwargs)
        self._image_artist = im
        # Add colorbar with min/max controls
        colorbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self._colorbars.append(colorbar)
        self._attach_colorbar_controls(colorbar)
        # Add geometry overlay if available
        if self.project:
            overlay = self.project.snapshot_geometry_overlay(idx)
            self._draw_geometry_overlay(overlay)
        self.axes.set_title(f"Snapshot #{record.index} {record.variable}")
        self.axes.set_xlabel(f"{record.ax1}-axis")
        self.axes.set_ylabel(f"{record.ax2}-axis")
        self._write_metadata(_snapshot_metadata(record))

    def _plot_history(self, indices: List[int]) -> None:
        """Plot one or more history time series.
        
        Multiple histories are plotted on the same axes with different colors/labels.
        
        Args:
            indices: List of history indices to plot
        """
        if self.logger:
            self.logger.info(f"Plotting {len(indices)} history/histories: {indices}")
        for idx in indices:
            axis, values, record = self.project.history_series(idx)
            if self.logger:
                self.logger.debug(f"History #{record.index} ({record.variable}): {len(values)} samples, range: [{values.min():.3e}, {values.max():.3e}]")
            self.axes.plot(axis, values, label=f"#{record.index} {record.variable}")
        self.axes.legend()
        self.axes.set_title("History")
        self.axes.set_xlabel("time (s)")
        self.axes.set_ylabel("value")
        details = "\n".join(_history_metadata(self.project.history_records()[idx]) for idx in indices)
        self._write_metadata(details)

    def _plot_dump(self, index: int) -> None:
        """Plot dump volume data as a 2D slice.
        
        Extracts a 2D slice from the 4D dump volume based on the selected component
        and displays it with optional geometry overlay.
        
        Args:
            index: Dump index to plot
        """
        record = self.project.dump_records()[index]
        if self.logger:
            self.logger.info(f"Plotting dump #{record.index}")
        volume, _ = self.project.dump_volume(index)
        # Get selected component index (clamped to valid range)
        comp = int(self._dump_component.get())
        comp = min(max(comp, 0), volume.shape[-1] - 1)
        if self.logger:
            self.logger.debug(f"Dump volume shape: {volume.shape}, selected component: {comp}")
        # Extract 2D slice: first k-layer, selected component
        data2d = volume[0, :, :, comp]
        if self.logger:
            self.logger.debug(f"2D slice shape: {data2d.shape}, range: [{data2d.min():.3e}, {data2d.max():.3e}]")
        cmap = cm.get_cmap(self._current_cmap.get())
        # Use locked color scale if enabled
        imshow_kwargs = {
            'origin': 'lower',
            'cmap': cmap,
            'aspect': 'auto',
        }
        if self._lock_color_scale and self._locked_vmin is not None and self._locked_vmax is not None:
            imshow_kwargs['vmin'] = self._locked_vmin
            imshow_kwargs['vmax'] = self._locked_vmax
            if self.logger:
                self.logger.debug(f"Using locked color scale: vmin={self._locked_vmin:.3e}, vmax={self._locked_vmax:.3e}")
        im = self.axes.imshow(data2d.T, **imshow_kwargs)
        colorbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self._colorbars.append(colorbar)
        # Add geometry overlay if available
        if self.project:
            overlay = self.project.dump_geometry_overlay(index)
            self._draw_geometry_overlay(overlay)
        self.axes.set_title(f"Dump #{record.index} component {comp}")
        self.axes.set_xlabel(f"{record.ax1}-axis")
        self.axes.set_ylabel(f"{record.ax2}-axis")
        self._write_metadata(_dump_metadata(record))

    # Formula handling -----------------------------------------------------

    def _prompt_formula(self) -> None:
        if not self.project or not self._current_selection:
            return
        data_type, indices = self._current_selection
        if data_type != 'hist':
            return
        if self.logger:
            self.logger.info("Opening formula dialog")
        formula = simpledialog.askstring(
            "Combine histories",
            "Enter formula using [1], [2], ... for the selected histories",
            parent=self,
        )
        if not formula:
            if self.logger:
                self.logger.debug("Formula dialog cancelled")
            return
        if self.logger:
            self.logger.info(f"Evaluating formula: {formula} on histories {indices}")
        try:
            result = self.project.evaluate_history_formula(indices, formula)
            if self.logger:
                self.logger.debug(f"Formula result: {len(result.values)} samples, axis: {result.axis_label}")
        except Exception as exc:
            if self.logger:
                self.logger.error(f"Formula evaluation failed: {formula}", exc_info=True)
            messagebox.showerror("WavePlot", f"Formula failed:\n{exc}")
            return
        self._plot_formula(result)

    def _prompt_filter(self) -> None:
        if not self.project or not self._current_selection:
            return
        data_type, indices = self._current_selection
        if data_type != 'hist' or len(indices) != 1:
            return

        if self.logger:
            self.logger.info("Opening Butterworth filter dialog")
        low = simpledialog.askstring(
            "Butterworth filter",
            "Low cut frequency (leave empty for low-pass).\n"
            "Accepts Hz (>1) or fraction of Nyquist (0-1).",
            parent=self,
        )
        if low is None:
            if self.logger:
                self.logger.debug("Filter dialog cancelled")
            return
        high = simpledialog.askstring(
            "Butterworth filter",
            "High cut frequency (leave empty for high-pass).\n"
            "Accepts Hz (>1) or fraction of Nyquist (0-1).",
            parent=self,
        )
        if high is None:
            if self.logger:
                self.logger.debug("Filter dialog cancelled")
            return

        order = simpledialog.askstring(
            "Butterworth filter",
            "Filter order (default 4).",
            parent=self,
        )
        if order is None:
            if self.logger:
                self.logger.debug("Filter dialog cancelled")
            return
        zero_phase = messagebox.askyesno(
            "Butterworth filter",
            "Use zero-phase filtering?",
            parent=self,
        )

        try:
            def _parse(value: str) -> Optional[float]:
                text = value.strip()
                return float(text) if text else None

            flo = _parse(low) if low is not None else None
            fhi = _parse(high) if high is not None else None
            if flo is None and fhi is None:
                if self.logger:
                    self.logger.warning("Filter parameters: both frequencies empty")
                messagebox.showerror("WavePlot", "Please provide at least one cutoff frequency.")
                return
            order_val = int(order.strip()) if order and order.strip() else 4
        except ValueError:
            if self.logger:
                self.logger.error("Invalid filter parameter values", exc_info=True)
            messagebox.showerror("WavePlot", "Invalid numeric value for filter parameters.")
            return

        zero_flag = 1 if zero_phase else 0
        args = ["[1]", str(flo) if flo is not None else "None", str(fhi) if fhi is not None else "None",
                str(max(order_val, 1)), str(zero_flag)]
        formula = "t(" + ", ".join(args) + ")"

        if self.logger:
            self.logger.info(f"Applying Butterworth filter: low={flo}, high={fhi}, order={order_val}, zero_phase={zero_phase}")

        try:
            result = self.project.evaluate_history_formula(indices, formula)
            result.description = "Butterworth filter"
            if self.logger:
                self.logger.debug(f"Filter result: {len(result.values)} samples")
        except Exception as exc:
            if self.logger:
                self.logger.error("Filtering failed", exc_info=True)
            messagebox.showerror("WavePlot", f"Filtering failed:\n{exc}")
            return
        self._plot_formula(result)

    def _plot_formula(self, result: FormulaResult) -> None:
        if self.logger:
            self.logger.info(f"Plotting formula result: {result.description}")
        self._clear_colorbars()
        self._clear_overlay_artists()
        self.axes.clear()
        self.axes.plot(result.axis, result.values, label=result.description)
        self.axes.legend()
        self.axes.set_xlabel(result.axis_label)
        self.axes.set_ylabel("value")
        self.axes.set_title("Derived history")
        self._write_metadata(f"Formula: {result.description}\nSamples: {len(result.values)}")
        self.canvas.draw_idle()

    # Helpers --------------------------------------------------------------

    def _clear_overlay_artists(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._overlay_artists.clear()

    def _clear_colorbars(self) -> None:
        # Remove any existing colorbars to avoid stacking new axes on the figure.
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                try:
                    cb.ax.remove()
                except Exception:
                    pass
        self._colorbars.clear()

        # Remove any colorbar-attached widget axes
        for ax in self._cb_widget_axes:
            try:
                ax.remove()
            except Exception:
                pass
        self._cb_widget_axes.clear()
        self._vmin_box = None
        self._vmax_box = None
        self._auto_btn = None

    def _draw_geometry_overlay(self, overlay: GeometryOverlay) -> None:
        if (
            not overlay.material_rects
            and not overlay.stope_rects
            and not overlay.source_points
        ):
            return

        for x0, y0, width, height, _ in overlay.material_rects:
            patch = patches.Rectangle(
                (x0, y0),
                width,
                height,
                linewidth=0.8,
                edgecolor="#ffffff",
                facecolor="none",
                alpha=0.6,
                clip_on=True,
            )
            self.axes.add_patch(patch)
            self._overlay_artists.append(patch)

        for x0, y0, width, height, _ in overlay.stope_rects:
            patch = patches.Rectangle(
                (x0, y0),
                width,
                height,
                linewidth=1.1,
                edgecolor="#ff9800",
                facecolor="none",
                linestyle=(0, (4, 2)),
                alpha=0.9,
                clip_on=True,
            )
            self.axes.add_patch(patch)
            self._overlay_artists.append(patch)

        if overlay.source_points:
            xs = [pt[0] for pt in overlay.source_points]
            ys = [pt[1] for pt in overlay.source_points]
            scatter = self.axes.scatter(
                xs,
                ys,
                marker="o",
                s=28,
                facecolors="#ffffff",
                edgecolors="#1f1f1f",
                linewidths=0.7,
                alpha=0.9,
            )
            self._overlay_artists.append(scatter)

    def _apply_clim(self, vmin: Optional[float] = None, vmax: Optional[float] = None, apply_if_empty: bool = True) -> None:
        """Apply vmin/vmax to the current image/colorbar.

        If both are None and apply_if_empty is True, autoscale to data range.
        """
        if not self._image_artist:
            return
        if vmin is None and vmax is None:
            if apply_if_empty:
                self._auto_clim()
            return
        if vmin is not None and vmax is not None and vmax <= vmin:
            messagebox.showerror("WavePlot", "Max must be greater than Min for color scale.")
            return
        # Apply limits to the current image
        self._image_artist.set_clim(vmin=vmin, vmax=vmax)
        for cb in self._colorbars:
            try:
                cb.update_normal(self._image_artist)
            except Exception:
                pass
        self.canvas.draw_idle()

    def _auto_clim(self) -> None:
        """Reset color limits to the data min/max and clear fields."""
        if not self._image_artist:
            return
        arr = self._image_artist.get_array()
        try:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
        except ValueError:
            # Empty array or all-NaN; do nothing
            if self.logger:
                self.logger.debug("Auto color scale: empty array or all-NaN, skipping")
            return
        self._image_artist.set_clim(vmin=vmin, vmax=vmax)
        for cb in self._colorbars:
            try:
                cb.update_normal(self._image_artist)
            except Exception:
                pass
        self.canvas.draw_idle()
        
        # If lock is enabled, update locked values to new auto-scaled values
        if self._lock_color_scale:
            self._locked_vmin = vmin
            self._locked_vmax = vmax
            if self.logger:
                self.logger.debug(f"Updated locked color scale: vmin={vmin:.3e}, vmax={vmax:.3e}")

    def _toggle_color_scale_lock(self) -> None:
        """Toggle color scale lock state."""
        if not self._image_artist:
            return
        
        if self._lock_color_scale:
            # Disable lock
            self._lock_color_scale = False
            self._locked_vmin = None
            self._locked_vmax = None
            if self._lock_btn:
                self._lock_btn.label.set_text('Lock')
            if self.logger:
                self.logger.info("Color scale lock disabled")
        else:
            # Enable lock - store current vmin/vmax
            vmin, vmax = self._image_artist.get_clim()
            self._lock_color_scale = True
            self._locked_vmin = float(vmin)
            self._locked_vmax = float(vmax)
            if self._lock_btn:
                self._lock_btn.label.set_text('Unlock')
            if self.logger:
                self.logger.info(f"Color scale lock enabled: vmin={self._locked_vmin:.3e}, vmax={self._locked_vmax:.3e}")
        
        self.canvas.draw_idle()

    def _attach_colorbar_controls(self, colorbar: object) -> None:
        """Place TextBox widgets at the top and bottom of the colorbar and an Auto button nearby."""
        if not self._image_artist:
            return
        try:
            cb_ax = colorbar.ax
            pos = cb_ax.get_position()
        except Exception:
            if self.logger:
                self.logger.debug("Failed to attach colorbar controls: colorbar.ax not available", exc_info=True)
            return
        # Dimensions in figure fraction
        pad = 0.005
        box_h = 0.035
        box_w = pos.width
        # Top and bottom y positions, clipped to figure
        top_y = min(pos.y1 + pad, 0.98 - box_h)
        bot_y = max(pos.y0 - box_h - pad, 0.02)

        vmax_ax = self.figure.add_axes([pos.x0, top_y, box_w, box_h])
        vmin_ax = self.figure.add_axes([pos.x0, bot_y, box_w, box_h])
        self._cb_widget_axes.extend([vmax_ax, vmin_ax])

        vmin_val, vmax_val = self._image_artist.get_clim()
        vmax_init = f"{vmax_val:.3e}" if np.isfinite(vmax_val) else ""
        vmin_init = f"{vmin_val:.3e}" if np.isfinite(vmin_val) else ""

        self._vmax_box = TextBox(vmax_ax, label='', initial=vmax_init)
        self._vmin_box = TextBox(vmin_ax, label='', initial=vmin_init)

        def _submit_vmax(text: str) -> None:
            text = (text or '').strip()
            if not text:
                return
            try:
                vmax_val = float(text)
                self._apply_clim(vmax=vmax_val, apply_if_empty=False)
                # Update locked value if lock is enabled
                if self._lock_color_scale:
                    self._locked_vmax = vmax_val
                    if self.logger:
                        self.logger.debug(f"Updated locked vmax: {vmax_val:.3e}")
            except ValueError:
                if self.logger:
                    self.logger.warning(f"Invalid Max value entered: {text}")
                messagebox.showerror("WavePlot", "Invalid Max value.")

        def _submit_vmin(text: str) -> None:
            text = (text or '').strip()
            if not text:
                return
            try:
                vmin_val = float(text)
                self._apply_clim(vmin=vmin_val, apply_if_empty=False)
                # Update locked value if lock is enabled
                if self._lock_color_scale:
                    self._locked_vmin = vmin_val
                    if self.logger:
                        self.logger.debug(f"Updated locked vmin: {vmin_val:.3e}")
            except ValueError:
                if self.logger:
                    self.logger.warning(f"Invalid Min value entered: {text}")
                messagebox.showerror("WavePlot", "Invalid Min value.")

        self._vmax_box.on_submit(_submit_vmax)
        self._vmin_box.on_submit(_submit_vmin)

        # Auto and Lock buttons placed to the right of the colorbar, stacked vertically
        btn_w = 0.05
        btn_h = 0.04
        btn_x = min(pos.x1 + pad, 0.98 - btn_w)
        # Calculate center of colorbar for vertical centering of button stack
        center_y = (pos.y0 + pos.y1) / 2
        # Stack height: 2 buttons + 1 gap between them
        stack_height = 2 * btn_h + pad
        # Auto button (top): center + half stack height - button height
        auto_btn_y = center_y + stack_height / 2 - btn_h
        auto_ax = self.figure.add_axes([btn_x, auto_btn_y, btn_w, btn_h])
        self._cb_widget_axes.append(auto_ax)
        self._auto_btn = Button(auto_ax, 'Auto')
        self._auto_btn.on_clicked(lambda _evt: self._auto_clim())
        
        # Lock button (bottom): center - half stack height
        lock_btn_y = center_y - stack_height / 2
        lock_ax = self.figure.add_axes([btn_x, lock_btn_y, btn_w, btn_h])
        self._cb_widget_axes.append(lock_ax)
        lock_btn_text = 'Unlock' if self._lock_color_scale else 'Lock'
        self._lock_btn = Button(lock_ax, lock_btn_text)
        self._lock_btn.on_clicked(lambda _evt: self._toggle_color_scale_lock())

    def _write_metadata(self, text: str) -> None:
        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete('1.0', tk.END)
        self.meta_text.insert(tk.END, text)
        self.meta_text.configure(state=tk.DISABLED)

    def _status_message(self, message: str) -> None:
        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete('1.0', tk.END)
        self.meta_text.insert(tk.END, message)
        self.meta_text.configure(state=tk.DISABLED)

    def _export_histories_csv(self) -> None:
        """Export all loaded histories to a CSV file with time as the first column."""
        if not self.project:
            messagebox.showinfo("WavePlot", "Load a project before exporting histories.")
            return

        hist_records = self.project.history_records()
        if not hist_records:
            messagebox.showinfo("WavePlot", "No histories available to export.")
            return

        if self.logger:
            self.logger.info(f"Exporting {len(hist_records)} histories to CSV")

        # Gather time/value arrays for alignment on the exact time stamps present
        time_axes: List[np.ndarray] = []
        values_list: List[np.ndarray] = []
        headers: List[str] = []

        try:
            for idx, rec in enumerate(hist_records):
                axis, values, record = self.project.history_series(idx)
                time_axes.append(np.asarray(axis, dtype=np.float64))
                values_list.append(np.asarray(values, dtype=np.float64))
                headers.append(f"hist_{record.index}_{record.variable}")

            union_times = np.unique(np.concatenate(time_axes))
            union_times.sort()
            data_matrix = np.full((len(union_times), len(values_list)), np.nan, dtype=np.float64)

            for col, (axis, vals) in enumerate(zip(time_axes, values_list)):
                positions = np.searchsorted(union_times, axis)
                # Ensure we only assign where the union time matches the rounded axis
                valid_mask = (positions >= 0) & (positions < len(union_times)) & (union_times[positions] == axis)
                data_matrix[positions[valid_mask], col] = vals[valid_mask]
        except Exception as exc:
            if self.logger:
                self.logger.error("Failed to prepare history data for export", exc_info=True)
            messagebox.showerror("WavePlot", f"Failed to prepare history data:\n{exc}")
            return

        path = filedialog.asksaveasfilename(
            title="Export histogram CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            if self.logger:
                self.logger.info("CSV export cancelled by user")
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["time"] + headers)
                for row_idx, t in enumerate(union_times):
                    row = [f"{t:.9g}"]
                    for col_idx in range(data_matrix.shape[1]):
                        val = data_matrix[row_idx, col_idx]
                        row.append("" if np.isnan(val) else f"{val:.9g}")
                    writer.writerow(row)
        except Exception as exc:
            if self.logger:
                self.logger.error("Failed to write CSV export", exc_info=True)
            messagebox.showerror("WavePlot", f"Failed to write CSV:\n{exc}")
            return

        if self.logger:
            self.logger.info(f"Histories exported to CSV: {path}")
        messagebox.showinfo("WavePlot", f"Histogram data exported to:\n{path}")

    def _on_closing(self) -> None:
        """Handle application shutdown."""
        if self.logger:
            self.logger.info("WavePlot application shutting down")
        self.destroy()


def _snapshot_metadata(record: SnapMapRecord) -> str:
    return (
        f"Snapshot #{record.index}\n"
        f"Variable: {record.variable}\n"
        f"Range i: {record.i_range}  j: {record.j_range}  k: {record.k_range}\n"
        f"Resolution: {record.x_qty} x {record.y_qty}\n"
        f"Time span: {record.time_start:.4f}s -> {record.time_end:.4f}s\n"
        f"Value range: {record.min_val:.3e} .. {record.max_val:.3e}"
    )


def _history_metadata(record: HistMapRecord) -> str:
    return (
        f"History #{record.index} {record.variable}\n"
        f"Samples: {record.sample_qty}  dt={record.dt:.3e}\n"
        f"Value scale: ±{record.vmax:.3e}"
    )


def _dump_metadata(record: DumpMapRecord) -> str:
    return (
        f"Dump #{record.index}\n"
        f"Grid: {record.i_qty} x {record.j_qty} x {max(record.k_qty,1)}\n"
        f"Variables: {record.v_qty}\n"
        f"Time: {record.time:.4f}s"
    )


def launch(map_path: Optional[str] = None, debug: bool = False) -> None:
    """Launch the WavePlot GUI application.
    
    This is the main entry point for the GUI. Creates and runs the application
    window, optionally loading a project file on startup.
    
    Args:
        map_path: Optional path to a .map file to load on startup
        debug: If True, enable debug logging
    """
    app = WavePlotApp(map_path, debug=debug)
    app.mainloop()


if __name__ == '__main__':
    launch()
