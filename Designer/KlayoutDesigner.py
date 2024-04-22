import pya
import numpy as np
import klayout.db as db
import matplotlib.pyplot as plt

from enum import Enum
from matplotlib.patches import Rectangle, Circle

class ORIENTATION(Enum):
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

# The colors I used for the display at the bottom of the notebook, nothing special
COLORS = ["#fcba03", "#a63019", "#1c67b8"]
  
class Designer():
    """Wrapper class to call both the Notebook function as de Klayout functions."""

    def __init__(self, layout:db.Layout, dbu:float, layer_amount:int, ax:plt.axes, temp_filename:str, marker_filename:str, output_filename:str) -> None:
        """Initialize the Designer

        Args:
            layout (db.Layout): A pya layout.
            dbu (float): Data base unit, so what size in microns is one unit.
            layer_amount (int): How many layers are there in the design
            ax (plt.axes): plt.axes to show in the designer
            temp_filename (str): Temporary filename
            marker_filename (str): Filename of the marker
            output_filename (str): Filename of the final result
        """
        self.temp_filename = temp_filename
        self.marker_filename = marker_filename
        self.output_filename = output_filename

        self.kdrawer = KlayoutDrawer(layout, dbu, layer_amount, temp_filename, marker_filename, output_filename)
        self.ndrawer = NotebookDrawer(ax)

    def vernier(self, x:int, y:int, ms_spacing:float, vs_spacing:float, width:int, ms_layerIdx:int, vs_layerIdx:int, max_length:int, orientation:ORIENTATION) -> None:
        """Generate a vernier scale

        Args:
            x (int): What is the x position of the midpoint of this vernier
            y (int): What is the y position of the midpoint of this vernier
            ms_spacing (float): What is the main scale spacing
            vs_spacing (float): What is the vernier scale spacing
            width (int): What is the width of the lines in this vernier
            ms_layerIdx (int): On what layer is the main scale made
            vs_layerIdx (int): On what layer is the vernier scale made_
            max_length (int): How long can this vernier be at maximum
            orientation (ORIENTATION): In what orientation is this vernier
        """
        self.kdrawer.construct_vernier(x, y, ms_spacing, vs_spacing, width, ms_layerIdx, vs_layerIdx ,max_length, orientation)
        self.ndrawer.render_vernier(x, y, ms_spacing, vs_spacing, width, ms_layerIdx, vs_layerIdx, max_length,orientation)
    
    def bar(self, x:int, y:int, width:int, height:int, layer:int) -> None:
        """Generate a bar/line

        Args:
            x (int): What is the x position of the midpoint of this bar
            y (int): What is the y position of the midpoint of this bar
            width (int): What is the width of this bar
            height (int): What is the height of this bar
            layer (int): On what layer is the bar made
        """
        self.kdrawer.construct_bar(x, y, width, height, layer)
        self.ndrawer.render_bar(x, y, width, height, layer)
        
    def cross(self, x:int, y:int, width:int, height:int, layer:int) -> None:
        """Generate a cross

        Args:
            x (int): What is the x position of the midpoint of this cross
            y (int): What is the y position of the midpoint of this cross
            width (int): What is the width of this cross
            height (int): What is the height of this cross
            layer (int): On what layer is the cross made
        """
        self.bar(x, y , width, height, layer)
        self.bar(x, y, height, width, layer)

    def marker(self, layer:int, *positions:tuple) -> None:
        """Draw the marker from a specified file, else a default marker will be drawn.

        Args:
            layer (int): On what layer is the marker made
        """
        if self.marker_filename == "Default_Marker.gds":
            print(f"No marker selected! Default marker used!")
        else:
            print(f"Marker selected! GDS will contain: {self.marker_filename[:-4]} as a marker.")
        for pos in positions:
            self.ndrawer.render_marker(pos[0], pos[1], layer)
            self.kdrawer.construct_marker(*positions)

    def draw(self) -> None:
        """Draw the layout"""
        self.kdrawer.draw()


class NotebookDrawer(Designer):
    """Class that hold all functions to draw the layout in the notebook

    Args:
        Designer (Designer): Designer wrapper
    """

    def __init__(self, ax:plt.axes) -> None:
        """Initialize the NotebookDrawer

        Args:
            ax (plt.axes): plt.axes to show in the designer
        """
        self.ax = ax
        self.ax.grid()

        # Default sizes for the plot, feel free to change
        self.ax.set_xlim([-1700, 1700]) 
        self.ax.set_ylim([-700, 700])

    def render_vernier(self, x:int, y:int, ms_spacing:float, vs_spacing:float, width:int, ms_layerIdx:int, vs_layerIdx:int, max_length:int, orientation:ORIENTATION) -> None:
        """Render a vernier scale based on given coordinates, sizes and orientation in a mpl graph.

        Args:
            x (int): x position of the mid point of the vernier scale
            y (int): y position of the mid point of the vernier scale
            ms_spacing (float): the spacing between the ticks on the main scale
            vs_spacing (float): the spacing between the ticks on the second scale
            max_length (int): the maximum length the scale is allowed to be
            orientation (ORIENTATION): the orientation of the scale in cardinal directions
            ax (plt.axis): the axis on which needs to be drawn
        """
        MID_TICK_HEIGHT = 50 # micrometer
        
        mid_tick_orientation_coords = {
            ORIENTATION.TOP : [((x, y), width, MID_TICK_HEIGHT), ((x, y), width, -MID_TICK_HEIGHT)],
            ORIENTATION.BOTTOM : [((x, y), width, -MID_TICK_HEIGHT), ((x, y), width, MID_TICK_HEIGHT)],
            ORIENTATION.LEFT : [((x, y), -MID_TICK_HEIGHT, width), ((x, y), MID_TICK_HEIGHT, width)],
            ORIENTATION.RIGHT : [((x, y), MID_TICK_HEIGHT, width), ((x, y), -MID_TICK_HEIGHT, width)]
        }
        
        # Assigning coordinates for main and vernier scale based on orientation
        ms_coords, vs_coords = mid_tick_orientation_coords[orientation]
        
        # Make the centerline for the mainscale
        self.ax.add_patch(Rectangle(*ms_coords, color=COLORS[ms_layerIdx]))

        # Make the centerline for the vernierscale
        self.ax.add_patch(Rectangle(*vs_coords, color=COLORS[vs_layerIdx]))
        
        # Determine the amount of ticks/stalke there will be
        number_of_stalks = max_length / (width + ms_spacing)
        for n in range(1, int(np.ceil(number_of_stalks / 2))):
            height = 40 if n % 10 == 0 else 25  # height in micrometers
            
            # Calculate coordinates for main scale positive and negative side and smae for the verniers in each orientation
            # Good possibilty this can be made in a better way
            reg_ticks_orientation_coords = {
                ORIENTATION.TOP: {
                    "ms_p_tick": ((x + n*(width + ms_spacing), y), width, height),
                    "ms_n_tick": ((x + width / 2 - n*(width + ms_spacing), y), width, height),
                    "vs_p_tick": ((x + n*(width + vs_spacing), y), width, -height),
                    "vs_n_tick": ((x + width / 2 - n*(width + vs_spacing), y), width, -height)
                },
                ORIENTATION.BOTTOM: {
                    "ms_p_tick": ((x + n*(width + ms_spacing), y), width, -height),
                    "ms_n_tick": ((x + width / 2 - n*(width + ms_spacing), y), width, -height),
                    "vs_p_tick": ((x + n*(width + vs_spacing), y), width, height),
                    "vs_n_tick": ((x + width / 2 - n*(width + vs_spacing), y), width, height)
                },
                ORIENTATION.LEFT: {
                    "ms_p_tick": ((x, y +n*(width+ms_spacing) - width/2), -height, width),
                    "ms_n_tick": ((x, y+width/2-n*(width+ms_spacing)), -height, width),
                    "vs_p_tick": ((x, y +n*(width+vs_spacing) - width/2), height, width),
                    "vs_n_tick": ((x, y+width/2-n*(width+vs_spacing)), height, width)
                },
                ORIENTATION.RIGHT: {
                    "ms_p_tick": ((x, y +n*(width+ms_spacing) - width/2), height, width),
                    "ms_n_tick": ((x, y+width/2-n*(width+ms_spacing)), height, width),
                    "vs_p_tick": ((x, y +n*(width+vs_spacing) - width/2), -height, width),
                    "vs_n_tick": ((x, y+width/2-n*(width+vs_spacing)), -height, width)
                }
            }
            ms_p_tick, ms_n_tick, vs_p_tick, vs_n_tick = reg_ticks_orientation_coords[orientation].values()

            # Draw rectangles at given coordinates
            self.ax.add_patch(Rectangle(*ms_p_tick, color=COLORS[ms_layerIdx]))
            self.ax.add_patch(Rectangle(*ms_n_tick, color=COLORS[ms_layerIdx]))
            self.ax.add_patch(Rectangle(*vs_p_tick, color=COLORS[vs_layerIdx]))
            self.ax.add_patch(Rectangle(*vs_n_tick, color=COLORS[vs_layerIdx]))

    def render_bar(self, x:int, y:int, width:int, height:int, layerIdx:int) -> None:
        """Draw the bar in the notebook

        Args:
            x (int): What is the x position of the midpoint of this bar
            y (int): What is the y position of the midpoint of this bar
            width (int): What is the width of this bar
            height (int): What is the height of this bar
            layer (int): On what layer is the bar made
        """
        self.ax.add_patch(Rectangle((x - width/2, y - height / 2), width, height, color=COLORS[layerIdx]))

    def render_marker(self, x:int, y:int, layerIdx:int) -> None:
        """Draw a marker icon on the marker x and y position.

        Args:
            x (int): What is the x position of the midpoint of this marker
            y (int): What is the y position of the midpoint of this marker
            layer (int): On what layer is the marker made
        """
        self.ax.add_patch(Circle((x, y), 25, color=COLORS[layerIdx], fill=False))
        self.render_bar(x, y, 2, 100, layerIdx)
        self.render_bar(x, y, 100, 2, layerIdx)

class KlayoutDrawer(Designer):
    """Class that hold all functions to draw the layout in the Klayout file

    Args:
        Designer (Designer): Designer wrapper
    """
    
    def __init__(self, layout:pya.Layout, dbu:float, layer_amount:int, temp_filename:str, marker_filename:str, output_filename:str) -> None: 
        """Initialize the KlayoutDrawer

        Args:
            layout (db.Layout): A pya layout.
            dbu (float): Data base unit, so what size in microns is one unit.
            layer_amount (int): How many layers are there in the design
            temp_filename (str): Temporary filename
            marker_filename (str): Filename of the marker
            output_filename (str): Filename of the final result
        """
        self.temp_filename = temp_filename
        self.marker_filename = marker_filename
        self.output_filename = output_filename

        self.layout = layout
        self.layout.dbu = dbu
        self.top = self.layout.create_cell("TopCell")
        self.layers = self.construct_layers(layer_amount)

    def construct_layers(self, layer_amount:int) -> list:
        """Construct the layers we need

        Args:
            layer_amount (int): Amount of layers

        Returns:
            list: List of layers
        """
        layer_array = []
        for i in range(layer_amount):
            layer_array.append(self.layout.layer(i,0))
        return layer_array

    def construct_bar(self, x:int, y:int, width:int, height:int, layerIdx:int) -> None:
        """Draw the bar in the file

        Args:
            x (int): What is the x position of the midpoint of this bar
            y (int): What is the y position of the midpoint of this bar
            width (int): What is the width of this bar
            height (int): What is the height of this bar
            layer (int): On what layer is the bar made
        """
        box = pya.DBox(x - width/2, y + height/2, x+width/2, y - height/2)
        self.top.shapes(self.layers[layerIdx]).insert(box)

    def construct_vernier(self, x:int, y:int, ms_spacing:float, vs_spacing:float, width, ms_layerIdx:int, vs_layerIdx:int, max_length:int, orientation:ORIENTATION):
        """Construct a vernier scale based on given coordinates, sizes and orientation.

        Args:
            x (int): x position of the mid point of the vernier scale
            y (int): y position of the mid point of the vernier scale
            ms_spacing (float): the spacing between the ticks on the main scale
            vs_spacing (float): the spacing between the ticks on the second scale
            max_length (int): the maximum length the scale is allowed to be
            orientation (ORIENTATION): the orientation of the scale in cardinal directions
        """
        #width = 10 # micrometer
        MID_TICK_HEIGHT = 50 # micrometer

        mid_tick_orientation_coords = {
            ORIENTATION.TOP : [(x- width/2, y, x+width/2, y+ MID_TICK_HEIGHT), (x- width/2, y-MID_TICK_HEIGHT, x+ width/2, y)],
            ORIENTATION.BOTTOM: [(x- width/2, y-MID_TICK_HEIGHT, x + width/2, y), (x- width/2, y, x + width/2, y+MID_TICK_HEIGHT)],
            ORIENTATION.LEFT: [(x, y- width/2, x-MID_TICK_HEIGHT, y +width/2), (x+MID_TICK_HEIGHT, y- width/2, x, y+width/2)],
            ORIENTATION.RIGHT: [(x+MID_TICK_HEIGHT, y- width/2, x, y + width/2), (x, y- width/2, x-MID_TICK_HEIGHT, y + width/2)]
        }

        # Assigning coordinates for main and vernier scale based on orientation
        ms_coords, vs_coords = mid_tick_orientation_coords[orientation]

        # Make the centerline for the mainscale
        ms_main_tick = pya.DBox(*ms_coords)

        # Make the centerline for the vernierscale
        vs_main_tick = pya.DBox(*vs_coords)

        # Add shape to its respective layer
        self.top.shapes(self.layers[ms_layerIdx]).insert(ms_main_tick)
        self.top.shapes(self.layers[vs_layerIdx]).insert(vs_main_tick)

        # Determine the amount of ticks/stalke there will be
        number_of_ticks = max_length / (width + ms_spacing)
        for n in range(1, int(np.ceil(number_of_ticks / 2))):
            height = 40 if n % 10 == 0 else 25  # height in micrometers

            # Calculate coordinates for main scale positive and negative side and smae for the verniers in each orientation
            # Good possibilty this can be made in a better way
            reg_ticks_orientation_coords = {
                ORIENTATION.TOP: {
                    "ms_p_tick": (x + n * (width + ms_spacing) - width / 2, y, x + ((n + 1) * width) + n * ms_spacing - width / 2, y + height),
                    "ms_n_tick": (x + width / 2 - n * (width + ms_spacing), y, x + width / 2 - ((n + 1) * width) - n * ms_spacing, y + height),
                    "vs_p_tick": (x + n * (width + vs_spacing) - width / 2, y - height, x + ((n + 1) * width) + n * vs_spacing - width / 2, y),
                    "vs_n_tick": (x + width / 2 - n * (width + vs_spacing), y - height, x + width / 2 - ((n + 1) * width) - n * vs_spacing, y)
                },
                ORIENTATION.BOTTOM: {
                    "ms_p_tick": (x + n * (width + ms_spacing) - width / 2, y - height, x + ((n + 1) * width) + n * ms_spacing - width / 2, y),
                    "ms_n_tick": (x + width / 2 - n * (width + ms_spacing), y - height, x + width / 2 - ((n + 1) * width) - n * ms_spacing, y),
                    "vs_p_tick": (x + n * (width + vs_spacing) - width / 2, y, x + ((n + 1) * width) + n * vs_spacing - width / 2, y + height),
                    "vs_n_tick": (x + width / 2 - n * (width + vs_spacing), y, x + width / 2 - ((n + 1) * width) - n * vs_spacing, y + height)
                },
                ORIENTATION.RIGHT: {
                    "ms_p_tick": (x + height, y + n * (width + ms_spacing) - width / 2, x, y + ((n + 1) * width) + n * ms_spacing - width / 2),
                    "ms_n_tick": (x + height, y + width / 2 - n * (width + ms_spacing), x, y + width / 2 - ((n + 1) * width) - n * ms_spacing),
                    "vs_p_tick": (x, y + n * (width + vs_spacing) - width / 2, x - height, y + ((n + 1) * width) + n * vs_spacing - width / 2),
                    "vs_n_tick": (x, y + width / 2 - n * (width + vs_spacing), x - height, y + width / 2 - ((n + 1) * width) - n * vs_spacing)
                },
                ORIENTATION.LEFT: {
                    "ms_p_tick": (x, y + n * (width + ms_spacing) - width / 2, x - height, y + ((n + 1) * width) + n * ms_spacing - width / 2),
                    "ms_n_tick": (x, y + width / 2 - n * (width + ms_spacing), x - height, y + width / 2 - ((n + 1) * width) - n * ms_spacing),
                    "vs_p_tick": (x + height, y + n * (width + vs_spacing) - width / 2, x, y + ((n + 1) * width) + n * vs_spacing - width / 2),
                    "vs_n_tick": (x + height, y + width / 2 - n * (width + vs_spacing), x, y + width / 2 - ((n + 1) * width) - n * vs_spacing)
                }
            }

            # Assigning coordinates for main and vernier scale based (negative an positive direction) on orientation
            ms_p_tick, ms_n_tick, vs_p_tick, vs_n_tick = reg_ticks_orientation_coords[orientation].values()
            
            # Make all boxes in negative an positive direction
            ms_p_tick = pya.DBox(*ms_p_tick)
            ms_n_tick = pya.DBox(*ms_n_tick)
            vs_p_tick = pya.DBox(*vs_p_tick)
            vs_n_tick = pya.DBox(*vs_n_tick)

            # Add shape to its respective layer
            self.top.shapes(self.layers[ms_layerIdx]).insert(ms_p_tick)
            self.top.shapes(self.layers[ms_layerIdx]).insert(ms_n_tick)
            self.top.shapes(self.layers[vs_layerIdx]).insert(vs_p_tick)
            self.top.shapes(self.layers[vs_layerIdx]).insert(vs_n_tick)

    def construct_marker(self, *positions:tuple) -> None:
        """Construct the marker in the Klaout design at given positions
        """

        layout = pya.Layout()
        top_cell = layout.create_cell("TOP")
        
        x = 0
        y = 0
        files = [self.temp_filename] + [self.marker_filename] * len(positions)
        for idx, file in enumerate(files):
            layout_import = pya.Layout()
            layout_import.read(file)
            
            imported_top_cell = layout_import.top_cell()
            target_cell = layout.create_cell(imported_top_cell.name)
            target_cell.copy_tree(imported_top_cell)
            
            layout_import._destroy()

            inst = pya.DCellInstArray(target_cell.cell_index(), pya.DTrans(db.DVector(x, y)))
            top_cell.insert(inst)
            
            x = positions[idx-1][0]
            y = positions[idx-1][1]
        layout.write(self.output_filename)

    def draw(self) -> None:
        """Draw the layout"""
        self.layout.write(self.temp_filename)
        self.layout.write(self.output_filename)
    