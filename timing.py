try: 
    import Tkinter as tk
except ImportError:
    import tkinter as tk
import widgets

import psana
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import pylab as plt
import timing_help


class Timingz(tk.Frame):
    def __init__(self, master, run, exp, detector_name,  *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.master = master
        self.psana_images = timing_help.PsanaImages( exp, run, detector_name) 
        self.bind()
        self._setup_psana_event_hopper()
        
        self._set_plot_params() 
        self.fig = plt.figure(1)
        plt.subplot(121)
        self.ax = plt.gca()
        plt.subplot(122)
        self.ax2 = plt.gca()

        self.fig2 = plt.figure(2)
        self.ax3 = plt.gca()

        self.psana_event_position = 0

        self._create_indicator_label()
        self._update_indicator()
        self._setup_spec_img_adjuster()

    def _set_plot_params(self):
        self.low_gain_pixel_bins = np.linspace( -20, 100, 121)
        #self.low_gain_pixel_cent = (self.low_gain_pixel_bins[:-1] +
        #    self.low_gain_pixel_bins[1:])*.5
        self.plot_bar_width = 0.9

    def _create_indicator_label(self):
        self.indicator_label_frame = tk.Frame( self.master)
        self.indicator_label_frame.pack(side=tk.TOP)
        self.indicator_label = tk.Label(self.indicator_label_frame, text="", 
            font="Helvetica 14")
        self.indicator_label.pack(fill=tk.BOTH)

    def _update_indicator(self):
        self.psana_img, self.spec_line, self.spec_img = self.psana_images[self.psana_event_position]
        self.indicator_label.config(text=self.psana_images.event_info_string)
        self._process_psana_image()
        self._plot_data()
        self.hopper_variable.set(self.psana_event_position)

    def _process_psana_image(self):
        pids = [7,8,0]
        self.some_panels = self.psana_img[pids]
        mapping = ~self.psana_images.gain_map[pids]
        self.lowgain_pixels = self.some_panels[mapping]
        print self.lowgain_pixels

    def _plot_data(self):
        self.ax.clear()
        counts,_ = np.histogram( self.lowgain_pixels, bins=self.low_gain_pixel_bins)
        self.ax.bar( left=self.low_gain_pixel_bins[:-1], 
            height=counts, width=self.plot_bar_width)
        self.ax.set_yscale("log")
        self.ax.set_ylim(0,1e5)
        self.ax.set_xlim(-25,101)
        
        self.ax2.clear() 
        self.ax2.plot( self.spec_line, '.')
        self.ax2.set_xlim(0,1024)
        self.ax2.set_yscale('log')
        self.ax2.set_ylim(0,1e4)
       
        self.ax3.clear()
        self.ax3.imshow(self.spec_img, vmin=0, vmax=15, cmap='gist_rainbow')
        plt.draw()
        plt.pause(0.1)

    def _setup_spec_img_adjuster(self):
        self.spec_adj_fr = tk.Frame( self.master)
        self.spec_adj_fr.pack(side=tk.TOP, pady=10)
        self.vmin_variable = tk.DoubleVar()
        self.vmax_variable = tk.DoubleVar()

        self.vmin_entry = widgets.LabeledEntry(
            self.spec_adj_fr,
            "vmin", 
            variable=self.vmin_variable, 
            label_width=10, 
            entry_width=10)
        
        self.vmax_entry = widgets.LabeledEntry(
            self.spec_adj_fr,
            "vmax", 
            variable=self.vmax_variable, 
            label_width=10, 
            entry_width=10)


        self.vmin_entry.pack(side=tk.LEFT)
        self.vmax_entry.pack(side=tk.LEFT)
        
        self.vmax_entry.entry.bind('<Return>', self._adjust_spec_clim)
        self.vmin_entry.entry.bind('<Return>', self._adjust_spec_clim)

        self.vmin_variable.set(0)
        self.vmax_variable.set(10)

    def _adjust_spec_clim(self, tkevent):
        vmin = self.vmin_variable.get()
        vmax = self.vmax_variable.get()
        if self.ax3.images:
            self.ax3.images[0].set_clim(vmin,vmax)
            plt.draw()
            plt.pause(0.01)

    def _setup_psana_event_hopper(self):
        self.hopper_frame = tk.Frame( self.master)
        self.hopper_frame.pack(side=tk.TOP, pady=10)
        self.hopper_variable = tk.IntVar(self.hopper_frame)

        self.hopper_entry = widgets.LabeledEntry(
            self.hopper_frame,
            "Hop to event", 
            variable=self.hopper_variable, 
            label_width=10, 
            entry_width=10)

        self.hopper_entry.pack(side=tk.TOP)
        self.hopper_variable.set(0)
        self.hopper_entry.entry.bind('<Return>', self._hop)

    def _hop(self, tkevent):
        self.psana_event_position = self.hopper_variable.get()
        if self.psana_event_position < 0:
            self.psana_event_position = 0
        elif self.psana_event_position >= self.psana_images.N_events:
            self.psana_event_position = self.psana_images.N_events-1
        self._update_indicator()

    def bind(self):
        # key bindings
        self.master.bind_all("<Right>", self._next)
        self.master.bind_all("<Left>", self._prev)
        self.master.bind_all("<Shift-Right>", self._offset_forwards)
        self.master.bind_all("<Shift-Left>", self._offset_backwards)

    def _next(self, tkevent):
        print "next"
        self.psana_event_position = min( self.psana_images.N_events-1,
            self.psana_event_position+1)
        self._update_indicator()

    def _prev(self, tkevent):
        print "prev"
        self.psana_event_position = max( 0,
            self.psana_event_position-1)
        self._update_indicator()

    def _offset_forwards(self, tkevent):
        print "forward"

    def _offset_backwards(self, tkevent):
        print "backward"


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser(
        description='plot stuff insync')
    
    parser.add_argument(
        '-r',
        dest='run',
        type=int, help='psana run number',
        required=True)
    
    parser.add_argument(
        '--detector',
        dest='detector',
        type=str, help='psana detector string',
        default="CxiDs2.0:Cspad.0")
    
    parser.add_argument(
        '--experiment',
        dest='experiment',
        type=str, help='psana experiment string',
        default="cxid9114")
    
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Timingz")
    
    frame = Timingz(
        root, 
        args.run, 
        args.experiment, 
        args.detector)
    
    frame.pack( side=tk.TOP, expand=tk.YES)
    root.mainloop()

