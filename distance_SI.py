#!/usr/bin/env python

def readvar(file, var, time=None):
    #read the field var in netCDF file
    from netCDF4 import Dataset
    import numpy
    fid = Dataset(file, 'r')
    vartmp = fid.variables[var]
    if  len(vartmp.shape) == 1:
          V = numpy.array(fid.variables[var][:]).squeeze()
    elif len(vartmp.shape) == 2:
        if time is None:
                V = numpy.array(fid.variables[var][:,:]).squeeze()
        else:
                V = numpy.array(fid.variables[var][time,:]).squeeze()
    elif len(vartmp.shape) == 3:
        if time is None:
                V = numpy.array(fid.variables[var][:,:,:]).squeeze()
        else:
                V = numpy.array(fid.variables[var][time,:,:]).squeeze()
    elif len(vartmp.shape) == 4:
        if time is None:
                V = numpy.array(fid.variables[var][:,:,:,:]).squeeze()
        else:
                V = numpy.array(fid.variables[var][time,:,:,:]).squeeze()
    else:
          print('unknown dimension for advected variable')
    fid.close()
    return V

def read_sic_nsidc(filename=[]):

    try: 
        from netCDF4 import Dataset
    except ImportError:
        print('package netCDF4 missing')
    import numpy
    import myDate
    listout = {}
    listunit = {}
     
    root_grp = Dataset(filename)
    dataset = root_grp.variables['latitude']
    lat = numpy.array(root_grp.variables['latitude'][:])
    dataset = root_grp.variables['longitude']
    lon = numpy.array(root_grp.variables['longitude'][:])
    dataset = root_grp.variables['time']
    time = numpy.array(root_grp.variables['time'][:])+myDate.date2jj0(1601,1,1)-myDate.date2jj0(1950,1,1)
    
    dataset = root_grp.variables['seaice_conc_cdr']
    sic = numpy.array(root_grp.variables['seaice_conc_cdr'][:,:,:]).squeeze()
    listunit['sic'] = ''
    sic[numpy.where(sic >= 251)] = float('nan')
    
    listout['sic'] = numpy.ma.array(sic,mask=(~numpy.isfinite(sic))) #create a masked array for the nans to appear in white
    root_grp.close()
    return lat, lon, time, listout, listunit  
 
 
def distance_latlon(lat1,lon1,lat2,lon2):
    #compute the distance in km between 2 lat/lon points
    import math
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    R = 6371  #radius of the earth in km
    x = (lon2 - lon1) * math.cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    d = R * math.sqrt( x*x + y*y )
    return d

def SIE(year,month,day,resolution): 
    #open the sea ice concentration data on the selected day, regrid the data at a higher resolution (ex 1km), and return the lat/lon of the 15% sea ice contour
    import nsidc
    import numpy
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset
    #resolution in km
    filename = '/data1/sassie/satellite/seaice/seaice_conc_daily_nh_f17_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'_v03r01.nc'
    [LA, LO,  ttime, llistvar, llistunit] = read_sic_nsidc(filename)
    x=readvar(filename,'xgrid')*10**(-3)
    y=readvar(filename,'ygrid')*10**(-3)
    [X,Y]=numpy.meshgrid(x,y)
    sic=numpy.ma.filled(llistvar['sic'],fill_value=numpy.nan)   

    ratio=25/resolution
    true_scale_lat = 70
    re = 6378.273
    e = 0.081816153
    delta=45
    xgrid=numpy.linspace(x[0],x[-1],int((len(x)-1)*ratio+1))
    ygrid=numpy.linspace(y[0],y[-1],int((len(y)-1)*ratio+1))
    [Xgrid,Ygrid]=numpy.meshgrid(xgrid,ygrid)
    sic_grid = griddata(( numpy.reshape(X,(numpy.prod(X.shape))),numpy.reshape(Y,(numpy.prod(Y.shape)))), numpy.reshape(sic,(numpy.prod(sic.shape))), (Xgrid, Ygrid), method='linear')

    [LOgrid,LAgrid]=nsidc.polar_xy_to_lonlat(Xgrid,Ygrid,true_scale_lat,re,e,1)
    LOgrid = LOgrid - delta
    LOgrid = LOgrid + numpy.less(LOgrid, 0)*360
    LOgrid[numpy.where(LOgrid>=180)]=LOgrid[numpy.where(LOgrid>=180)]-360
    
    cc2=plt.contour(xgrid,ygrid,sic_grid*100,levels=15,colors='m',linewidths=3)
    dat0=cc2.allsegs
    lon_sie=[]
    lat_sie=[]
    for i in range(0,len(dat0[0])):
        [lon,lat]=nsidc.polar_xy_to_lonlat(dat0[0][i][:,0],dat0[0][i][:,1],true_scale_lat,re,e,1)
        lon = lon - delta
        lon = lon + numpy.less(lon, 0)*360
        lon[numpy.where(lon>=180)]=lon[numpy.where(lon>=180)]-360
        lon_sie=numpy.hstack((lon_sie,lon))
        lat_sie=numpy.hstack((lat_sie,lat))
    plt.close('all')
    
    return lon_sie,lat_sie

def distance_sie_point(lon,lat,lon_sie,lat_sie,lon_min,lon_max,lat_min,lat_max): 
    #within an area defined by lat/lon min and max, for a lat/lon point return the distance from the sea ice edge
    #lon/lat: point
    #lon_sie/lat_sie: array of sea ice edge contours
    import numpy
        
    ind = numpy.where(numpy.logical_and.reduce((lon_sie>=lon_min,lon_sie<=lon_max,lat_sie>=lat_min,lat_sie<=lat_max)))
    lon_sie=lon_sie[ind[0]]
    lat_sie=lat_sie[ind[0]]

    d_tmp=99999
    for k in range(0,len(lon_sie)):
        d = distance_latlon(lat,lon,lat_sie[k],lon_sie[k])
        if d<d_tmp:
            d_tmp=d
    distancefromice=d_tmp
    
    return distancefromice 
    

# [lon_sie,lat_sie]=SIE(year,month,day,resolution)
# for i in 
# # distancefromice[i,j]=myGrid.distance_sie_point(LO[i,j],LA[i,j],lon_sie,lat_sie,lonmin,lonmax,latmin,latmax)
