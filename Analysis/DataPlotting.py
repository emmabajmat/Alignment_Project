import matplotlib.pyplot as plt
import numpy as np



def draw_axes_crossing_line( ax1:plt.axes, ax2:plt.axes, xy_data:tuple, xy_derivative:tuple) -> None:
    """Draw the fancy dotted line between the two axes.

    Args:
        ax1 (plt.axes): The mean intensity axes
        ax2 (plt.axes): The derivative axes
        xy_data (tuple): The position within the mean intensity axes
        xy_derivative (tuple): The position within the derivative axes
    """
    ax1.annotate(
        '', 
        xy= (xy_data[0], xy_data[1]), 
        xytext= (xy_data[0], xy_data[1]),
        xycoords= ax1.transData, 
        textcoords= ax2.transData, 
        arrowprops= dict(
            facecolor='black', 
            arrowstyle='-', 
            linestyle='dashed',
            clip_on=False
        )
    )
    
    ax2.annotate(
        '', 
        xy= (xy_data[0], xy_data[1]), 
        xytext= (xy_derivative[0], xy_derivative[1]),
        xycoords= ax1.transData, 
        textcoords= ax2.transData, 
        arrowprops= dict(
            facecolor='black', 
            arrowstyle='-', 
            linestyle='dashed'
        )
    )

def draw_diagonal_splits( ax1:plt.axes, ax2:plt.axes, size:float = 0.005) -> None:
    """Draw the diagonal splits between the two axes

    Args:
        ax1 (plt.axes): The mean intensity axes
        ax2 (plt.axes): The derivative axes
        size (float, optional): How big to make the diagonal lines in axes coordinates. Defaults to 0.005.
    """
    
    # Arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-size,1+size),(-size,+size), **kwargs)   # top-right diagonal
    ax1.plot((-size,size),(-size,size), **kwargs)       # top-left diagonal

    kwargs.update(transform=ax2.transAxes)              # switch to the bottom axes
    ax2.plot((1-size,1+size),(1-size,1+size), **kwargs) # bottom-right diagonal
    ax2.plot((-size,size),(1-size,1+size), **kwargs)    # bottom-left diagonal

def draw_mean_and_derivative_plot( mean_intensity_profile:np.ndarray, intensity_profile_derivative:np.ndarray, leading_edges:np.ndarray, trailing_edges:np.ndarray, filename:str = "") -> None:
    """_summary_

    Args:
        mean_intensity_profile (np.ndarray): _description_
        intensity_profile_derivative (np.ndarray): _description_
        leading_edges (np.ndarray): _description_
        trailing_edges (np.ndarray): _description_
        filename (str, optional): _description_. Defaults to "".
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 4), sharex=True)
    
    ax1.plot(mean_intensity_profile, color = 'black')
    ax2.plot(intensity_profile_derivative, color = 'red', linestyle='-.')
    
    for le, te in zip(leading_edges, trailing_edges):
        # Draw x-markers at the local maxima and minima
        ax2.scatter(le, intensity_profile_derivative[le], color='black', marker='x')
        ax2.scatter(te, intensity_profile_derivative[te], color='black', marker='x')
        
        # Make the dotted line between the two axes to show where the derivative
        # matches the location on the data. To keep this a bit cleaner a function was
        # made..
        draw_axes_crossing_line(ax1, ax2, (le, mean_intensity_profile[le]), (le, intensity_profile_derivative[le]))
        draw_axes_crossing_line(ax1, ax2, (te, mean_intensity_profile[te]), (te, intensity_profile_derivative[te]))
    
    # Styling the plot
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax1.set_ylabel('Mean Intensity')
    ax2.set_ylabel('Derivative')
    ax2.set_xlabel('Pixel Coordinates')
    
    ax1.tick_params(length=0)
    ax1.set_xlim(0, len(mean_intensity_profile))
    
    plt.subplots_adjust(hspace=0.05)
    draw_diagonal_splits(ax1, ax2)
    
    if filename != "":
        plt.savefig(f'{filename}_profile.png', bbox_inches='tight')