"""Tkinter GUI for the WavePlot Python port."""

from __future__ import annotations

import os
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

try:  # Matplotlib >= 3.5
    from matplotlib import colormaps as _cm_registry  # type: ignore
except Exception:  # pragma: no cover - fallback for older Matplotlib
    _cm_registry = None

from ..geometry_loader import GeometryOverlay
from ..project import FormulaResult, WaveProject
from ..map_parser import DumpMapRecord, HistMapRecord, SnapMapRecord


def _available_colormaps() -> List[str]:
    """Return a readable list of diverging/sequential maps without reversed variants."""
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
    """Main application window."""

    def __init__(self, map_path: Optional[str] = None) -> None:
        super().__init__()
        self.title("WavePlot (Python Edition)")
        self.geometry("1200x720")
        self.minsize(900, 600)
        self.configure(bg="#f4f6fb")

        self.project: Optional[WaveProject] = None
        self._tree_map: Dict[str, Tuple[str, int]] = {}
        self._current_selection: Optional[Tuple[str, List[int]]] = None
        self._current_cmap = tk.StringVar(value='viridis')
        self._dump_component = tk.StringVar(value='0')
        self._colorbars: List[object] = []
        self._image_artist: Optional[object] = None
        # Colorbar widget tracking (Matplotlib widgets placed near the colorbar)
        self._cb_widget_axes: List[object] = []
        self._vmin_box: Optional[TextBox] = None
        self._vmax_box: Optional[TextBox] = None
        self._auto_btn: Optional[Button] = None
        self._overlay_artists: List[object] = []

        self._build_ui()
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

    def _build_detail_panel(self) -> None:
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
        cmap_box.bind('<<ComboboxSelected>>', lambda _: self._refresh_plot())

        ttk.Label(top_bar, text="Dump component:").pack(side=tk.LEFT)
        self.dump_combo = ttk.Combobox(
            top_bar,
            textvariable=self._dump_component,
            values=["0"],
            width=6,
            state='readonly',
        )
        self.dump_combo.pack(side=tk.LEFT, padx=(4, 16))
        self.dump_combo.bind('<<ComboboxSelected>>', lambda _: self._refresh_plot())

        self.multi_var = tk.IntVar(value=0)
        self.multi_slider = ttk.Scale(
            top_bar,
            from_=0,
            to=0,
            variable=self.multi_var,
            command=lambda _: self._refresh_plot(),
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
        try:
            self.project = WaveProject(map_path)
        except Exception as exc:
            messagebox.showerror("WavePlot", f"Failed to load project:\n{exc}")
            return
        self._populate_tree()
        self._status_message(f"Loaded {os.path.basename(map_path)}")

    def _open_map_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .map file",
            filetypes=[('WavePlot map', '*.map'), ('All files', '*.*')],
        )
        if path:
            self.load_project(path)

    def _populate_tree(self) -> None:
        for item in self.tree.get_children(''):
            self.tree.delete(item)
        self._tree_map.clear()
        if not self.project:
            return
        snaps_node = self.tree.insert('', 'end', text='Snapshots')
        for rec in self.project.snapshot_records():
            desc = (
                f"#{rec.index} {rec.variable} | {rec.x_qty}x{rec.y_qty} | "
                f"t={rec.time_end:.3f}s"
            )
            node = self.tree.insert(snaps_node, 'end', text=desc)
            self._tree_map[node] = ('snap', rec.index - 1)
        hists_node = self.tree.insert('', 'end', text='Histories')
        for rec in self.project.history_records():
            desc = (
                f"#{rec.index} {rec.variable} | samples={rec.sample_qty} | "
                f"dt={rec.dt:.3e}"
            )
            node = self.tree.insert(hists_node, 'end', text=desc)
            self._tree_map[node] = ('hist', rec.index - 1)
        dumps_node = self.tree.insert('', 'end', text='Dumps')
        for rec in self.project.dump_records():
            desc = (
                f"#{rec.index} | grid={rec.i_qty}x{rec.j_qty}x{max(rec.k_qty,1)} | "
                f"vars={rec.v_qty}"
            )
            node = self.tree.insert(dumps_node, 'end', text=desc)
            self._tree_map[node] = ('dump', rec.index - 1)
        self.tree.item(snaps_node, open=True)
        self.tree.item(hists_node, open=True)
        self.tree.item(dumps_node, open=False)

    # Tree interactions ----------------------------------------------------

    def _on_tree_select(self, _event: tk.Event) -> None:
        if not self.project:
            return
        selection = [item for item in self.tree.selection() if item in self._tree_map]
        if not selection:
            return
        types = {self._tree_map[item][0] for item in selection}
        if len(types) > 1:
            messagebox.showinfo("WavePlot", "Please select entries from one group at a time.")
            return
        selected_type = next(iter(types))
        indices = sorted(self._tree_map[item][1] for item in selection)
        self._current_selection = (selected_type, indices)
        self._update_controls(selected_type, len(indices))
        self._refresh_plot()

    def _update_controls(self, data_type: str, count: int) -> None:
        if data_type == 'dump':
            record = self.project.dump_records()[self._current_selection[1][0]]
            values = [str(i) for i in range(record.v_qty)]
            self.dump_combo.configure(values=values, state='readonly')
            if self._dump_component.get() not in values:
                self._dump_component.set(values[0])
        else:
            self.dump_combo.configure(values=["0"], state='disabled')
        if data_type == 'snap' and count > 1:
            self.multi_slider.configure(state='normal', from_=0, to=count - 1)
            self.multi_var.set(0)
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
            self.canvas.draw_idle()
            return
        data_type, indices = self._current_selection
        if data_type == 'snap':
            self._plot_snapshot(indices)
        elif data_type == 'hist':
            self._plot_history(indices)
        elif data_type == 'dump':
            self._plot_dump(indices[0])
        self.canvas.draw_idle()

    def _plot_snapshot(self, indices: List[int]) -> None:
        idx = indices[int(self.multi_var.get())] if len(indices) > 1 else indices[0]
        record = self.project.snapshot_records()[idx]
        data = self.project.snapshot_array(idx)
        cmap = cm.get_cmap(self._current_cmap.get())
        im = self.axes.imshow(
            data.T,
            origin='lower',
            cmap=cmap,
            aspect='auto',
        )
        self._image_artist = im
        colorbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self._colorbars.append(colorbar)
        # Attach min/max boxes at the top and bottom of the colorbar and an Auto button
        self._attach_colorbar_controls(colorbar)
        if self.project:
            overlay = self.project.snapshot_geometry_overlay(idx)
            self._draw_geometry_overlay(overlay)
        self.axes.set_title(f"Snapshot #{record.index} {record.variable}")
        self.axes.set_xlabel(f"{record.ax1}-axis")
        self.axes.set_ylabel(f"{record.ax2}-axis")
        self._write_metadata(_snapshot_metadata(record))

    def _plot_history(self, indices: List[int]) -> None:
        for idx in indices:
            axis, values, record = self.project.history_series(idx)
            self.axes.plot(axis, values, label=f"#{record.index} {record.variable}")
        self.axes.legend()
        self.axes.set_title("History")
        self.axes.set_xlabel("time (s)")
        self.axes.set_ylabel("value")
        details = "\n".join(_history_metadata(self.project.history_records()[idx]) for idx in indices)
        self._write_metadata(details)

    def _plot_dump(self, index: int) -> None:
        record = self.project.dump_records()[index]
        volume, _ = self.project.dump_volume(index)
        comp = int(self._dump_component.get())
        comp = min(max(comp, 0), volume.shape[-1] - 1)
        data2d = volume[0, :, :, comp]
        cmap = cm.get_cmap(self._current_cmap.get())
        im = self.axes.imshow(data2d.T, origin='lower', cmap=cmap, aspect='auto')
        colorbar = self.figure.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self._colorbars.append(colorbar)
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
        formula = simpledialog.askstring(
            "Combine histories",
            "Enter formula using [1], [2], ... for the selected histories",
            parent=self,
        )
        if not formula:
            return
        try:
            result = self.project.evaluate_history_formula(indices, formula)
        except Exception as exc:
            messagebox.showerror("WavePlot", f"Formula failed:\n{exc}")
            return
        self._plot_formula(result)

    def _prompt_filter(self) -> None:
        if not self.project or not self._current_selection:
            return
        data_type, indices = self._current_selection
        if data_type != 'hist' or len(indices) != 1:
            return

        low = simpledialog.askstring(
            "Butterworth filter",
            "Low cut frequency (leave empty for low-pass).\n"
            "Accepts Hz (>1) or fraction of Nyquist (0-1).",
            parent=self,
        )
        if low is None:
            return
        high = simpledialog.askstring(
            "Butterworth filter",
            "High cut frequency (leave empty for high-pass).\n"
            "Accepts Hz (>1) or fraction of Nyquist (0-1).",
            parent=self,
        )
        if high is None:
            return

        order = simpledialog.askstring(
            "Butterworth filter",
            "Filter order (default 4).",
            parent=self,
        )
        if order is None:
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
                messagebox.showerror("WavePlot", "Please provide at least one cutoff frequency.")
                return
            order_val = int(order.strip()) if order and order.strip() else 4
        except ValueError:
            messagebox.showerror("WavePlot", "Invalid numeric value for filter parameters.")
            return

        zero_flag = 1 if zero_phase else 0
        args = ["[1]", str(flo) if flo is not None else "None", str(fhi) if fhi is not None else "None",
                str(max(order_val, 1)), str(zero_flag)]
        formula = "t(" + ", ".join(args) + ")"

        try:
            result = self.project.evaluate_history_formula(indices, formula)
            result.description = "Butterworth filter"
        except Exception as exc:
            messagebox.showerror("WavePlot", f"Filtering failed:\n{exc}")
            return
        self._plot_formula(result)

    def _plot_formula(self, result: FormulaResult) -> None:
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
            return
        self._image_artist.set_clim(vmin=vmin, vmax=vmax)
        for cb in self._colorbars:
            try:
                cb.update_normal(self._image_artist)
            except Exception:
                pass
        self.canvas.draw_idle()

    def _attach_colorbar_controls(self, colorbar: object) -> None:
        """Place TextBox widgets at the top and bottom of the colorbar and an Auto button nearby."""
        if not self._image_artist:
            return
        try:
            cb_ax = colorbar.ax
            pos = cb_ax.get_position()
        except Exception:
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
                self._apply_clim(vmax=float(text), apply_if_empty=False)
            except ValueError:
                messagebox.showerror("WavePlot", "Invalid Max value.")

        def _submit_vmin(text: str) -> None:
            text = (text or '').strip()
            if not text:
                return
            try:
                self._apply_clim(vmin=float(text), apply_if_empty=False)
            except ValueError:
                messagebox.showerror("WavePlot", "Invalid Min value.")

        self._vmax_box.on_submit(_submit_vmax)
        self._vmin_box.on_submit(_submit_vmin)

        # Auto button placed to the right of the colorbar, vertically centered
        btn_w = 0.05
        btn_h = 0.04
        btn_x = min(pos.x1 + pad, 0.98 - btn_w)
        btn_y = (pos.y0 + pos.y1) / 2 - btn_h / 2
        auto_ax = self.figure.add_axes([btn_x, btn_y, btn_w, btn_h])
        self._cb_widget_axes.append(auto_ax)
        self._auto_btn = Button(auto_ax, 'Auto')
        self._auto_btn.on_clicked(lambda _evt: self._auto_clim())

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


def launch(map_path: Optional[str] = None) -> None:
    app = WavePlotApp(map_path)
    app.mainloop()


if __name__ == '__main__':
    launch()
