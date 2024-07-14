from plotnine import ggplot, theme, element_blank,element_text, element_line, element_rect,theme, ggtitle,geom_line,labs,scale_fill_brewer,scale_color_brewer

def configure_line_plot():
    """
    Configures the plot with the given parameters:
    - Title with a font of 20 and bold text
    - X axis text with an italic font of size 16
    - Y axis with similar formatting as the x axis
    - Eliminates the grid vertical lines and keeps the horizontal lines
    - Makes the horizontal lines transparent up to a 70% and colored with a grey color
    - Makes the background color of the plot smoke
    """
    # Create the plot with the specified customizations
    plot = ggplot() +labs(x="X Axis Title", y="Y Axis Title") + theme( 
         # Customize the plot title
         plot_title=element_text(size=20, face='bold'),
         
         # Customize the X axis text
         axis_text_x=element_text(size=12, face='italic'),
         axis_title_x=element_text(size=16, face='italic'),
         # Customize the Y axis text
         axis_text_y=element_text(size=12, face='italic'),
         axis_title_y=element_text(size=16, face='italic'),
         # Eliminate vertical grid lines and keep horizontal grid lines
         panel_grid_major_x=element_blank(),
         panel_grid_major_y=element_line(color='grey', alpha=0.3, linetype='dashed'),
         
         # Customize the grid background color
         panel_background=element_rect(fill='snow', color=None,alpha=0.5)

     ) + ggtitle("Customized Plot Title")  # Set the plot title
    return plot

def get_line_colors(plot,continuous=False):
    """
    Chooses the colors used in the plot given the number of categories for ggplot.
    If continuous=True, it generates a continuous color scale.
    If continuous=False, it returns a discrete list of colors for categorical variables.
    """
    if continuous:
        plot = plot + scale_color_brewer(type='qual',palette='Dark2')
        return plot
    else:
        return plot



