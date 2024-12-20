#------------------------------------------

def write_pickle(filename,variable):
    import pickle
    with open(filename, 'wb') as p:
        pickle.dump(variable, p)
    del p

def read_pickle(filename):
    import pickle
    with open(filename, 'rb') as p:
        output = pickle.load(p)
    del p
    return output

#------------------------------------------
# plotting-related

def symbols_ck(keyword):
    if keyword == 'degreeC':
        symb = r'$\hspace{-0.1}{^\circ}\hspace{-0.25}$C'
    elif keyword == 'degree':
        symb = r'$\hspace{-0.1}{^\circ}\hspace{-0.25}$'
    elif keyword == 'r2':
        symb = r'r$\mathregular{^2} \!$'
    elif keyword == 'km3':
        symb = r'km$\mathregular{^3} \!$'
    elif keyword == 'km2':
        symb = r'km$\mathregular{^2} \!$'
    elif keyword == 'm2':
        symb = r'm$\mathregular{^2} \!$'
    elif keyword == 'm3':
        symb = r'm$\mathregular{^3} \!$'
    elif keyword == 'delta':
        symb = r'$\Delta\!\!$ '
    else:
        symb = 'FALSE KEYWORD'
        
    return symb

def minus_formatter(x): 
    return str(x).replace('-', u'\u2212')
    
def pm_formatter(x): 
    return str(x).replace('#', u'\u00B1')
    
def times_formatter(x): 
    return str(x).replace('#', u'\u00D7')

def digit_formatter(inputnumber, digits):
        
    return minus_formatter(("{0:0.%sf}"%(digits)).format(round(inputnumber, digits) + 0.0))
    
def annotatefun(ax, textlist, xstart, ystart, ydiff = 0.05, fonts = 12, col = 'k', ha='left'):
    counter = 0
    for textstr in textlist:
        ax.text(xstart, (ystart-counter*ydiff), textstr, fontsize = fonts, ha=ha, va='center', transform=ax.transAxes, color = col)
        counter += 1
                       
def xlabeling(ax, ticks, size, labels = [], rot = 0):
    
    ticks_checked = []
    
    for tick in ticks:
        if abs(tick-0) < 0.00001:    
            ticks_checked.append(0)
        else:
            ticks_checked.append(tick)
            
    if labels == []:
        ticksstr = [minus_formatter(str(tick)) for tick in ticks_checked]
    else:
        ticksstr = labels
    ax.set_xticks((ticks_checked))
    ax.set_xticklabels((ticksstr), fontsize = size, rotation = rot)

def xlabeling_km(ax, ticks, size):
    ticksstr = [minus_formatter(str(int(tick*0.0001))) for tick in ticks]
    ax.set_xticks((ticks))
    ax.set_xticklabels((ticksstr), fontsize = size)

def ylabeling(ax, ticks, size, labels = [], rot = 0):
    
    ticks_checked = []
    
    for tick in ticks:
        if abs(tick-0) < 0.00001:    
            ticks_checked.append(0)
        else:
            ticks_checked.append(tick)
            
    if labels == []:
        ticksstr = [minus_formatter(str(tick)) for tick in ticks_checked]
    else:
        ticksstr = labels
    ax.set_yticks((ticks_checked))
    ax.set_yticklabels((ticksstr), fontsize = size, rotation = rot)

def ylabeling_km(ax, ticks, size):
    ticksstr = [str(int(tick*0.0001)) for tick in ticks]
    ax.set_yticks((ticks))
    ax.set_yticklabels((ticksstr), fontsize = size)

def majorlabeling(ax, ticks, side, si = 12, rot = 0):
    ticksstr = [minus_formatter(str(tick)) for tick in ticks]
    if side == 'x':
        ax.set_xticks((ticks))
        ax.set_xticklabels((ticksstr), name = 'Arial', fontsize = si, rotation = rot)
    elif side == 'y':
        ax.set_yticks((ticks))
        ax.set_yticklabels((ticksstr), name = 'Arial', fontsize = si, rotation = rot)
        
def minorlabeling(ax, ticks, side):
    if side == 'x':
        ax.set_xticks((ticks), minor = 1)
        ax.tick_params('x', length=2, width=.5, which='minor')
    elif side == 'y':
        ax.set_yticks((ticks), minor = 1)
        ax.tick_params('y', length=2, width=.5, which='minor')

def colors_ck():

    colornamelist = ['orange','orange_light','green','green_light',
    'purple_light','purple','blue','blue_light','red_light','red','brown','brown_light', 'darkgray']    
    
    colorlist = [[255,127,0],[253,191,111],[51,160,44],[178,223,138],[202,178,214],[106,36,194], # 106,61,154 new purple  
      [25,116,210],[166,206,227],[252,116,94],[227,26,28],[106,53,24],[177,89,40],[50,50,50]];   #31,120,180 old blue

    #-------------------------
    numbers = range(0, len(colornamelist), 1)
    colordict = {}

    for n in numbers:
        colorlist[n][0] = colorlist[n][0]/255.0
        colorlist[n][1] = colorlist[n][1]/255.0
        colorlist[n][2] = colorlist[n][2]/255.0

    for n, colorname in zip(numbers, colornamelist):
        colordict[colorname] = colorlist[n]

    return colordict
         
#------------------------------------------
# date-related

def cdate(n, s = "01/03/1979"):
    """Returns list with a) time since epoch (Jan 1 1970) in sec and b) actual timeobject"""
    se = 86400
    starttime = time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y").timetuple())
    timenew = starttime + n*se
    timeobject = datetime.datetime.fromtimestamp(int(timenew))
    return [timenew,timeobject]
    ## year = timeobject.year
    ## month = timeobject.month
    ## day = timeobject.day

def cnumber(d1 = '01/03/1979', d2 = '01/01/1982'):
    """Returns list with daily values of time since epoch (Jan 1 1970) in sec"""
    se = 86400
    starttime = time.mktime(datetime.datetime.strptime('01/03/1979', "%m/%d/%Y").timetuple())
    d1time = time.mktime(datetime.datetime.strptime(d1, "%m/%d/%Y").timetuple())
    d2time = time.mktime(datetime.datetime.strptime(d2, "%m/%d/%Y").timetuple())

    n1 = int((d1time - starttime)/se + 1)
    n2 = int((d2time - starttime)/se + 1)

    return range(n1,n2)

def matlab2datetime(matlab_datenum):
    import datetime
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac
    
def datetime2matlab(py_datetime):
    import datetime
    mdn = py_datetime + datetime.timedelta(days = 366)
    frac_seconds = (py_datetime - datetime.datetime(py_datetime.year, py_datetime.month, py_datetime.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = py_datetime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def datetime2ordinal(do):
    '''function to convert datetime object to ordinal'''
    import datetime
    t = datetime.date.toordinal(do) 
    return t

def ordinal2date(o):
    import datetime
    d = datetime.date.fromordinal(o)
    return d

def date2ordinal(y,m,d):
    '''function to convert y, m, d into ordinal'''
    import datetime
    do = datetime.date(y,m,d)
    t = datetime.date.toordinal(do) 
    return t

def year2ordinal(y):
    '''function to convert years into ordinal'''
    import datetime   
    do = datetime.date(int(y),1,1)
    t = datetime.date.toordinal(do) 
    return t

def year2ordinal_array(years):
    '''function to convert years into ordinal'''
    import datetime
    import numpy as np
    li = list(years)
    d = [year2ordinal(x) for x in li]
    ar = np.array(d)
    return ar
    
def matlab2datetime_array(matlabdates):
    import numpy as np
    li = list(matlabdates)
    d = [matlab2datetime(x) for x in li]
    ar = np.array(d)
    return ar

def matlab2ordinal_array(matlabdates):
    import numpy as np
    li = list(matlabdates)
    d = [datetime2ordinal(matlab2datetime(x)) for x in li]
    ar = np.array(d)
    return ar

def format_date_axis(plt,ax,kw,rot=45,yr=10,sbyr=1):
    
    import matplotlib.dates as mpd
        
    if kw == 'tenyear_year':

        years10 = mpd.YearLocator(yr)
        years = mpd.YearLocator(sbyr)
        Fmt = mpd.DateFormatter('%Y')
    
        # format the ticks
        ax.xaxis.set_major_locator(years10)
        ax.xaxis.set_major_formatter(Fmt)
        ax.xaxis.set_minor_locator(years)
        
    if  kw == 'months':
        years = mpd.YearLocator(1)
        months = mpd.MonthLocator()  # every month
        Fmt = mpd.DateFormatter('%m/%y')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(Fmt)
        #ax.xaxis.set_minor_locator(months)
        
    if  kw == 'days':
        years = mpd.YearLocator(1)
        months = mpd.MonthLocator()
        days = mpd.DayLocator() # every month
        Fmt = mpd.DateFormatter('%m/%d')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(Fmt)
        #ax.xaxis.set_minor_locator(months)
        
    if  kw == 'day_month_year':
        years = mpd.YearLocator(1)
        months = mpd.MonthLocator()
        days = mpd.DayLocator() # every month
        Fmt = mpd.DateFormatter('%m/%d/%y')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(Fmt)
        #ax.xaxis.set_minor_locator(months)
    
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=rot)

    
#------------------------------------------
# map projection-related

def get_wkt_prj(epsg_code):
    import urllib
    wkt = urllib.urlopen('http://spatialreference.org/ref/epsg/{0}/prettywkt/'.format(epsg_code))
    remove_spaces = wkt.read().replace(" ","")
    output = remove_spaces.replace("\n", "")
    return output

#------------------------------------------
# animation-related

# Function to create an animation from folder of images (works on Windows, not tested on Linux)
def create_animation(folder_images, file_anim, duration=500):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt 
    import matplotlib.animation as animation
    from pathlib import Path
    
    # List all files in the folder and filter for image files
    image_files = [f for f in os.listdir(folder_images) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    image_files.sort()  # Sort the files to maintain the correct order

    # Load the images into a list
    images = []
    for f in image_files: #f=image_files[0]
        image_path = os.path.join(folder_images, f)
        img = Image.open(image_path)
        images.append(img)
    
    # Create a figure with the dimensions desired for the video
    scalefactor=1 #set to 3 to reduce size by a factor of 3
    figw=images[0].size[0]/100/scalefactor #/100 account for dpi, reduce size by scalefactor
    figh=images[0].size[1]/100/scalefactor
    fig = plt.figure(figsize=(figw,figh), dpi=100)
    # Add an axes to the figure
    ax = fig.add_axes([0, 0, 1, 1])  # Full size
    ax.axis('off')  # Hide the axis

    # Function to update the image in the animation
    def update_frame(i):
        ax.clear()  # Clear the previous frame
        ax.axis('off')  # Keep the axis hidden
        ax.imshow(images[i])  # Show the current image (defaults to aspect='equal')
    
    # Create the shell of the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(images), interval=duration, repeat=True, repeat_delay=1000)
    # Save the animation as video using ffmpeg (the real work)
    ani.save(Path(folder_images,file_anim), writer='ffmpeg', fps=30)

    print(f"  Animation saved as {file_anim} in {folder_images}")

