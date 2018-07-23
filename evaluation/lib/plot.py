import scipy as sc
import pylab as pl
import itertools as it
from matplotlib.markers import MarkerStyle


def fmeasure(p, r):
    """Calculates the fmeasure for precision p and recall r."""
    return 2*p*r / (p+r)


def _fmeasureCurve(f, p):
    """For a given f1 value and precision get the recall value.
    The f1 measure is defined as: f(p,r) = 2*p*r / (p + r).
    If you want to plot "equipotential-lines" into a precision/recall diagramm
    (recall (y) over precision (x)), for a given fixed f value we get this
    function by solving for r:
    """
    return f * p / (2 * p - f)


def _plotFMeasures(fstepsize=.1, stepsize=0.001, figSize='normal'):
    """Plots 10 fmeasure Curves into the current canvas."""
    p = sc.arange(0., 1.02, stepsize)[1:]
    for f in sc.arange(0., 1., fstepsize)[1:]:
        points = [(x, _fmeasureCurve(f, x)) for x in p
                  if 0 < _fmeasureCurve(f, x) <= 1.5]
        xs, ys = zip(*points)
        curve, = pl.plot(xs, ys, "--", color="gray", linewidth=0.5)  # , label=r"$f=%.1f$"%f) # exclude labels, for legend
        # bad hack:
        # gets the 10th last datapoint, from that goes a bit to the left, and a bit down
        
		#calculate dx of f1-score labels
        if figSize == 'normal':
            dx = 0.065
        elif figSize == 'small': 	
            dx = 0.13 
        elif figSize == 'ieee': 	
            dx = 0.09 
        else: #others and None
            dx = 0.065
				
        pl.annotate(r"$f=%.1f$" % f, xy=(xs[-10], ys[-10]), xytext=(xs[-10] - dx, ys[-10] - 0.035), size="small", color="gray")

# def _contourPlotFMeasure():
#    delta = 0.01
#    x = sc.arange(0.,1.,delta)
#    y = sc.arange(0.,1.,delta)
#    X,Y = sc.meshgrid(x,y)
#    cs = pl.contour(X,Y,fmeasure,sc.arange(0.1,1.0,0.1)) # FIXME: make an array out of fmeasure first
#    pl.clabel(cs, inline=1, fontsize=10)

colors_ = "bgrcmyk"  # 7 is a prime, so we'll loop over all combinations of colors and markers, when zipping their cycles
markers_ = "so^>v<dph8"  # +x taken out, as no color.

# # if you don't believe the prime loop:
# icons = set()
# for i,j in it.izip(it.cycle(colors),it.cycle(markers)):
#    if (i,j) in icons: break
#    icons.add((i,j))
# print len(icons), len(colors)*len(markers)


def plotPrecisionRecallDiagram(title="title", points=None, labels=None, loc=(1.01, 0), colors = None, markers = None, ncol=1, exportLocation = None, show=True, hideLegend=False, figSize=None, columnSpacing=1.1):
    """Plot (precision,recall) values with 10 f-Measure equipotential lines.
    Plots into the current canvas.
    Points is a list of (precision,recall) pairs.
    Optionally you can also provide labels (list of strings), which will be
    used to create a legend, which is located at loc.
    """
    if figSize == 'normal':
        figsize = (6,6)
        markersize = 50
    elif figSize == 'small': 	
        figsize = (3,3)
        markersize = 25
    elif figSize == 'ieee': 	
        figsize = (4.5,4.5)
        markersize = 37.5
    else: #others and None
        figsize = (6,4)
        markersize = 50
        
    fig = pl.figure(figsize=figsize)

	
    if labels:
        ax = pl.axes([0.1, 0.1, 1.0, 1.0])  # llc_x, llc_y, width, height
    else:
        ax = pl.gca()
    
    pl.title(title)
    pl.xlabel("Precision")
    pl.ylabel("Recall")
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(xmin=0.01)
    #ax.set_ylim(ymin=0.01)
    _plotFMeasures(figSize=figSize)

    # _contourPlotFMeasure()

    if points.any():
        #getColor = next(it.cycle(colors))
        #getMarker = next(it.cycle(markers))

        scps = []  # scatter points
        for i, (x, y) in enumerate(points):
            # get color
            if colors != None:
                color = colors[i]
            else:   
                color = colors_[int(i % 5)]
  
            # get marker
            if markers != None:
                marker = markers[i]
            else:
                marker = markers_[int(i / 5)]
            
            # get label
            label = None
            if labels:
                label = labels[i]
            
            # plot
            scp = ax.scatter(x, y, label=label, s=markersize, linewidths=0.75, facecolor=color, alpha=0.5,
                             marker=marker,
                             #marker=MarkerStyle(marker=marker, fillstyle='none'),
         #markeredgecolor='red',
         #markeredgewidth=0.0)
                            )
            scps.append(scp)
            # pl.plot(x,y, label=label, marker=getMarker(), markeredgewidth=0.75, markerfacecolor=getColor())
            # if labels: pl.text(x, y, label, fontsize="x-small")
        if labels and not hideLegend:
            #pl.legend(scps, labels, loc=loc, scatterpoints=1, numpoints=1, fancybox=True) # passing scps & labels explicitly to work around a bug with legend seeming to miss out the 2nd scatterplot
            pl.legend(scps, labels, loc=loc, scatterpoints=1, numpoints=1, fancybox=True, ncol=ncol, handletextpad=0.1, columnspacing=columnSpacing)  # passing scps & labels explicitly to work around a bug with legend seeming to miss out the 2nd scatterplot
    pl.axis([-0.01, 1.02, -0.01, 1.02])  # xmin, xmax, ymin, ymax
    #pl.axis([0.01, 1.02, 0.01, 1.02])  # xmin, xmax, ymin, ymax
    if show:
        pl.show()
	
    if exportLocation != None:
        fig.savefig(exportLocation, bbox_inches='tight')