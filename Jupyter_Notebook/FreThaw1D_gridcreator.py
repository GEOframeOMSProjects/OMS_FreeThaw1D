# -*- coding: utf-8 -*-
"""
Created on 10/29/2019

This is used to create the grid for FreThaw1D model.

@author: Niccolo` Tubini
@license: creative commons 4.0
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from netCDF4 import Dataset

def grid1D(data_grid, dz_min, b, grid_type):
	'''
    This function creates the geometry of 1D grid for a finite volume numerical.     
    
    :param data_grid: pandas dataframe containg the grid_input_file.csv
    :type data_grid: pandas dataframe

     :param dz_min: thickness of the first layer
    :type dz_min: float

    :param b: growth rate, range [0, 1[.
    :type b: float

    :param grid_type: type of the grid. The grid can be created using picewise mesh spacing, classical, or an exponential function, exponential
	
    return:
    
    KMAX: number of control volumes
    type NMAX: int

    eta: vertical coordinate of control volumes centroids positive upward with origin set at soil surface.
    type eta: array

    eta_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil surface.
    type eta_dual: array

    space_delta: is the distance between two adjacent control volumes. 
                This quantity is used to compute gradients.    
    type space_delta: array
    
    z: vertical coordinate of control volumes centroids positive upward with origin set at soil column bottom. This is the spatial coordinate used to to write the equation
    type z: array

    z_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil column bottom. 
    type zDual: array		
        
    '''
	if(grid_type=='classical'):
		return build_grid(data_grid)
	elif(grid_type=='exponential'):
		return build_grid_exponential(data_grid, dz_min, b)
	else:
		print('Check grid_type')
		

def build_grid(data_grid):
    '''
    This function creates the geometry of 1D grid for a finite volume numerical. The discretizion is 
	with constant mesh spacing in each layers.
    scheme.
    
    
    :param data_grid: pandas dataframe containg the grid_input_file.csv
	:type data_grid: pandas dataframe
	

    return:
    
	KMAX: number of control volumes
	type KMAX: int
	
	VECTOR_LENGTH: length of the array. This value is equal to NMAX is size_factor 1, suggested whenever there is no need to regrid.
	type VECTOR_LENGTH: int
	
    eta: vertical coordinate of control volumes centroids positive upward with origin set at soil surface.
	type eta: array
	
    eta_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil surface.
	type eta_dual: array

	space_delta: is the distance between two adjacent control volumes. 
                This quantity is used to compute gradients.    
	type space_delta: array
    
    z: vertical coordinate of control volumes centroids positive upward with origin set at soil column bottom. This is the spatial coordinate used to to write the equation
	type z: array
    
    z_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil column bottom. 
	type zDual: array		
	
    control_volume: dimension of control volume 
	type control_volume: array		
        
    '''
    # get the number og control volumes
    KMAX = int(data_grid['K'].sum())
#     VECTOR_LENGTH = int(np.ceil(data_grid['K'].sum())*size_factor)

    
    # list containing centroids coordinates
    tmp_eta = []
    # list containing control volumes interface coordinates
    tmp_eta_dual = []
    
    # array containing centroids coordinates measured along eta
    eta = np.zeros(KMAX,dtype=float)
    # array containing control volumes interface coordinates measured along eta
    eta_dual = np.zeros(KMAX+1,dtype=float)
    # array containing centroids coordinates measured along z
    z = np.zeros(KMAX,dtype=float)
    # array containing control volumes interface coordinates measured along z
    z_dual = np.zeros(KMAX+1,dtype=float)
    # array containing distances between centroids (used to compute gradient)
    space_delta = np.zeros(KMAX+1,dtype=float)
    # array containing control volume size
    control_volume = np.zeros(KMAX,dtype=float)



    
    for i in range(np.size(data_grid.index)-1,0,-1):
		
        if data_grid['Type'][i]=='L' and data_grid['Type'][i-1]=='L':
			
            deta = ( data_grid['eta'][i]-data_grid['eta'][i-1])/data_grid['K'][i-1]
            tmp_eta=np.append(tmp_eta, np.linspace(data_grid['eta'][i]-deta/2,data_grid['eta'][i-1]+deta/2,num=data_grid['K'][i-1],endpoint=True) )
            tmp_eta_dual=np.append(tmp_eta_dual, np.linspace(data_grid['eta'][i],data_grid['eta'][i-1],num=data_grid['K'][i-1]+1,endpoint=True) )
			
        elif data_grid['Type'][i]=='L' and data_grid['Type'][i-1]=='M':
			
            deta = ( data_grid['eta'][i]-data_grid['eta'][i-1])/data_grid['K'][i-1]
            tmp_eta=np.append(tmp_eta, np.linspace(data_grid['eta'][i]-deta/2,data_grid['eta'][i-1],num=data_grid['K'][i-1],endpoint=True) )
            tmp_eta_dual=np.append(tmp_eta_dual, np.linspace(data_grid['eta'][i],data_grid['eta'][i-1]+deta/2,num=data_grid['K'][i-1],endpoint=True) )
			
        elif data_grid['Type'][i]=='M' and data_grid['Type'][i-1]=='L':
			
            deta = ( data_grid['eta'][i]-data_grid['eta'][i-1])/data_grid['K'][i-1]
            tmp_eta=np.append(tmp_eta, np.linspace(data_grid['eta'][i],data_grid['eta'][i-1]+deta/2,num=data_grid['K'][i-1],endpoint=True) )
            tmp_eta_dual=np.append(tmp_eta_dual, np.linspace(data_grid['eta'][i]-deta/2,data_grid['eta'][i-1],num=data_grid['K'][i-1],endpoint=True) )
			
        else:
            print("ERROR!!")  
        
    # to eliminate doubles
    tmp_eta=[ii for n,ii in enumerate(tmp_eta) if ii not in tmp_eta[:n]]
    tmp_eta_dual=[ii for n,ii in enumerate(tmp_eta_dual) if ii not in tmp_eta_dual[:n]]

    # move from list to array
    for i in range(0,len(tmp_eta)):
		
        eta[i] = tmp_eta[i]
        z[i] = tmp_eta[i] - data_grid['eta'][np.size(data_grid['eta'])-1]
        
    for i in range(0,len(tmp_eta_dual)):
		
        eta_dual[i] = tmp_eta_dual[i]
        z_dual[i] = tmp_eta_dual[i] - data_grid['eta'][np.size(data_grid['eta'])-1]
		
        if i==0:
			
            space_delta[i] = np.abs(eta_dual[i]-eta[i])
			
        elif i==np.size(eta_dual)-1:
			
            space_delta[i] = np.abs(eta_dual[i]-eta[i-1])
			
        else:
			
            space_delta[i] = np.abs(eta[i-1]-eta[i]) 
           
    for i in range(0,len(eta_dual)-1):
        control_volume[i] = np.abs(eta_dual[i]-eta_dual[i+1])		
		
    return [KMAX, eta, eta_dual, space_delta, z, z_dual, control_volume]


def build_grid_exponential(data_grid, dz_min, b):
    '''
    This function creates the geometry of 1D grid for a finite volume numerical. The discretizion is 
    with exponential function (Gubler S. et al. 2013, doi:10.5194/gmd-6-1319-2013).
    
    
    :param data_grid: pandas dataframe containg the grid_input_file.csv
    :type data_grid: pandas dataframe

     :param dz_min: thickness of the first layer
    :type dz_min: float

    :param b: growth rate, range [0, 1[.
    :type b: float

   
    
    return:
    
    KMAX: number of control volumes
    type NMAX: int

    eta: vertical coordinate of control volumes centroids positive upward with origin set at soil surface.
    type eta: array

    eta_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil surface.
    type eta_dual: array

    space_delta: is the distance between two adjacent control volumes. 
                This quantity is used to compute gradients.    
    type space_delta: array
    
    z: vertical coordinate of control volumes centroids positive upward with origin set at soil column bottom. This is the spatial coordinate used to to write the equation
    type z: array

    z_dual: vertical coordinate of control volumes interfaces positive upward with origin set at soil column bottom. 
    type zDual: array		
	
    control_volume: dimension of control volume 
	type control_volume: array	
        
    '''
  
    # list containing layer thickness
    tmp_dz = []

    z_max = -data_grid['eta'][1]
    dz_sum = 0
    k = 0
    while (z_max-dz_sum)>1E-12:

        tmp_dz.append(dz_min*(1+b)**k)
        dz_sum = dz_sum + dz_min*(1+b)**k

        if(dz_sum>z_max):

            tmp_dz.pop()
            dz_sum = dz_sum-dz_min*(1+b)**k
            tmp_dz.append(z_max-dz_sum)
            dz_sum = dz_sum + z_max-dz_sum

        k = k+1
        
    # get the number og control volumes
    KMAX = len(tmp_dz)
    dz = np.zeros(KMAX,dtype=float)
    
    for i in range(0,KMAX):

        dz[i] = tmp_dz[i]
    
    # array containing centroids coordinates measured along eta
    eta = np.zeros(KMAX,dtype=float)
    # array containing control volumes interface coordinates measured along eta
    eta_dual = np.zeros(KMAX+1,dtype=float)
    # array containing centroids coordinates measured along z
    z = np.zeros(KMAX,dtype=float)
    # array containing control volumes interface coordinates measured along z
    z_dual = np.zeros(KMAX+1,dtype=float)
    # array containing distances between centroids (used to compute gradient)
    space_delta = np.zeros(KMAX+1,dtype=float)
    # array containing control volume size
    control_volume = np.zeros(KMAX,dtype=float)
	
	
    tmp = 0
    for i in range(0,KMAX):
        z[i] = dz[KMAX-1-i]/2+tmp
        z_dual[i] = tmp
        tmp = tmp+dz[KMAX-1-i]
        eta[i] = -z_max + z[i]
        eta_dual[i] = -z_max + z_dual[i]


    z_dual[KMAX] = z_max
    eta_dual[KMAX] = 0.0       


    for i in range(0,KMAX+1):

        if i==0:

            space_delta[i] = np.abs(eta_dual[i]-eta[i])

        elif i==np.size(eta_dual)-1:

            space_delta[i] = np.abs(eta_dual[i]-eta[i-1])

        else:

            space_delta[i] = np.abs(eta[i-1]-eta[i]) 

    for i in range(0,len(eta_dual)-1):
        control_volume[i] = np.abs(eta_dual[i]-eta_dual[i+1])	
		
    return [KMAX, eta, eta_dual, space_delta, z, z_dual, control_volume]


def set_initial_condition(data, eta, interp_model):
    '''
    This function define the problem initial condition for temperature. The initial condition
    is interpolated starting from some pairs (eta,T0) contained in a .csv file.
    The interpolation is performed using the class scipy.interpolate.interp1d
    
    
    :param data: pandas dataframe containg pairs of (eta, T0)
    :type data_grid: pandas dataframe
    
    :param eta: vertical coordinate of control volume centroids. It is positive upward with 
        origin set at soil surface.
    :type eta: list
    
    :param interp_model: specifies the kind of interpolation as a string. 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
    :type ic_type: str
    
    return:
    
    ic: initial condition
    type ic: array
        
    '''
    eta_points = data['eta']
    ic_points = data['T0']
    f = interp1d(eta_points, ic_points, kind=interp_model, assume_sorted=False)
    
    ic = f(eta)
   
    return ic




def set_parameters(data_grid, data_parameter, KMAX, eta):
    '''
    This function associate to each control volume a label that identifies 
    the rheology model, the set of parameters describing the soil type, and the max/min cell size 
    for regridding.
    
    :param data_grid: pandas dataframe containg the grid_input_file.csv
    :type data: pandas dataframe
    
    :param data_parameter: pandas dataframe containg the parameter_input_file.csv
    :type data: pandas dataframe
    
    :param KMAX: number of control volumes.
    :type KMAX: int
    
    :param eta: vertical coordinate of control volume centroids. It is positive upward with 
        origin set at soil surface.
    :type eta: list
    
    return:
    
    rheology_ID:
    type: array
    
    parameters_ID:
    type: array
    
    regrid_ID:
    type: array
    	
    soil_particles_density:
    type:array
    
    thermal_conductivity_soil_particles:
    type:array
    
    specific_thermal_capacity_soil_particles:
    type:array
    
    theta_s:
    type:array
    
    theta_r:
    type:array
    
    melting_temperature:
    type:array
    
    par_1:
    type:array
    
    par_2:
    type:array
    
    par_3:
    type:array
    
    par_4:
    type:array
    '''
    rheology_ID = np.zeros(KMAX, dtype=float)
    parameter_ID = np.zeros(KMAX, dtype=float)
    
    soil_particles_density = np.zeros(data_parameter.shape[0], dtype=float)
    thermal_conductivity_soil_particles = np.zeros(data_parameter.shape[0], dtype=float)
    specific_heat_capacity_soil_particles = np.zeros(data_parameter.shape[0], dtype=float)
    theta_s = np.zeros(data_parameter.shape[0], dtype=float)
    theta_r = np.zeros(data_parameter.shape[0], dtype=float)
    melting_temperature = np.zeros(data_parameter.shape[0], dtype=float)
    par_1 = np.zeros(data_parameter.shape[0], dtype=float)
    par_2 = np.zeros(data_parameter.shape[0], dtype=float)
    par_3 = np.zeros(data_parameter.shape[0], dtype=float)
    par_4 = np.zeros(data_parameter.shape[0], dtype=float)
    
    coord_layer = []
    tmp_rheology_ID = []
    tmp_parameter_ID = []
    tmp_regrid_ID = []

    for i in data_grid.index:

        if data_grid['Type'][i] == 'L':
            
            coord_layer.append(data_grid['eta'][i])
            tmp_rheology_ID.append(data_grid['rheologyID'][i])
            tmp_parameter_ID.append(data_grid['parameterID'][i])
           
     
    for i in range(np.size(coord_layer)-1,0,-1):
        
        for j in range(0,KMAX):
            
            if(eta[j]>coord_layer[i] and eta[j]<coord_layer[i-1] ):
                
                rheology_ID[j] = tmp_rheology_ID[i-1]
                parameter_ID[j] = tmp_parameter_ID[i-1]
        
        
    for i in range(0, data_parameter.iloc[:,0].size):
        soil_particles_density[i] = data_parameter['spDensity'][i]
        thermal_conductivity_soil_particles[i] = data_parameter['spConductivity'][i]
        specific_heat_capacity_soil_particles[i] = data_parameter['spSpecificHeatCapacity'][i]
        theta_s[i] = data_parameter['thetaS'][i]
        theta_r[i] = data_parameter['thetaR'][i]
        melting_temperature[i] = data_parameter['meltingT'][i]
        par_1[i] = data_parameter['par1'][i]
        par_2[i] = data_parameter['par2'][i]
        par_3[i] = data_parameter['par3'][i]
        par_4[i] = data_parameter['par4'][i]
        
        # nan must be changed to number
        soil_particles_density[np.isnan(soil_particles_density)] = -999.0
        thermal_conductivity_soil_particles[np.isnan(thermal_conductivity_soil_particles)] = -999.0
        specific_heat_capacity_soil_particles[np.isnan(specific_heat_capacity_soil_particles)] = -999.0
        theta_s[np.isnan(theta_s)] = -999.0
        theta_r[np.isnan(theta_r)] = -999.0
        melting_temperature[np.isnan(melting_temperature)] = -999.0
        par_1[np.isnan(par_1)] = -999.0
        par_2[np.isnan(par_2)] = -999.0
        par_3[np.isnan(par_3)] = -999.0
        par_4[np.isnan(par_4)] = -999.0

    return [rheology_ID, parameter_ID, soil_particles_density, thermal_conductivity_soil_particles, specific_heat_capacity_soil_particles,
         theta_s, theta_r, melting_temperature, par_1, par_2, par_3, par_4]


def write_grid_netCDF(eta, eta_dual, z, z_dual, space_delta, soil_volume, ic, rheology_ID, parameter_ID, KMAX, soil_particles_density,              
					  thermal_conductivity_soil_particles, 
					  specific_heat_capacity_soil_particles, theta_s, theta_r, melting_temperature, par_1, par_2, par_3, par_4,
					  output_file_name, output_title, output_institution, output_summary, output_date,
					  grid_input_file_name, parameter_input_file_name):
	'''
	Save all grid data in a NetCDF file
	
	:param eta: vertical coordinate of control volume centroids. It is positive upward with.
        origin set at soil surface.
	:type eta: array
	
	:param eta_dual: vertical coordinate of control volume interface. It is positive upward with.
        origin set at soil surface.
	:type eta_dual: array
	
	:param z: vertical coordinate of control volume centroids. It is positive upward with.
        origin set at soil column bottom.
	:type z: array
	
	:param z_dual: vertical coordinate of control volume interfaces. It is positive upward with.
        origin set at soil column bottom.
	:type z_dual: array
	
	:param space_delta: is the distance between two adjacent control volumes.
        This quantity is used to compute gradients
	:type space_delta: array
	
	:param soil_volume: soil volume of each control volume.
	:type soil_volume: array.
	
	:param ic: temperature initial condition.
	:type ic: array.
	
	:param rheology_ID: containing a label for each control volume defining the type of the rheology to be used.
	:type rheology_ID: array.
	
	:param parameter_ID: containing a label for each control volume defining the parameter set to be used.
	:type parameter_ID: array.
		
	:param KMAX: number of control volumes.
	:type KMAX: int.
		
	:param soil_particles_density: array containing the soil particles density.
	:type soil_particles_density: array.
	
	:param thermal_conductivity_soil_particles: array containing the thermal conductivity of soil particles.
	:type thermal_conductivity_soil_particles: array.
	
	:param specific_heat_capacity_soil_particles: array containing the specific heat capacity of soil particles.
	:type specific_heat_capacity_soil_particles: array.
	
	:param theta_s: array containing the values of the water content at saturation.
	:type theta_s: array.
	
	:param theta_r: array containing the values of the residual water content.
	:type theta_r: array.

	:param melting_temperature: array containing the melting temperature.
	:type melting_temperature: array.
	
	:param par1: array containing the values of the SFCC parameter.
	:type par1: array.
	
	:param par2: array containing the values of the SFCC parameter.
	:type par2: array.
	
	:param par3: array containing the values of the SFCC parameter.
	:type par3: array.
	
	:param par4: array containing the values of the SFCC parameter.
	:type par4: array.

	:param output_file_name: 
	:type output_file_name: str
	
	:param output_title: 
	:type output_title: str
	
	:param output_institution: 
	:type param output_institution: str
	
	:param output_summary: 
	:type output_summary: str
	
	:param output_date: 
	:type output_date: str
	
	:param input_file_name: 
	:type input_file_name: str
	
	'''
	
    # the output array to write will be nx x ny
	dim = np.size(eta);
	dim1 = np.size(eta_dual);
	dim_parameter = np.size(par_1)
	dim_scalar = 1
	
	
    # open a new netCDF file for writing.
	ncfile = Dataset(output_file_name,'w') 
	
    # Create global attributes
	ncfile.title = output_title + '\\n' + 'grid input file' + grid_input_file_name + 'parameter input file' + parameter_input_file_name
	ncfile.institution =  output_institution
	ncfile.summary = output_summary
    #ncfile.acknowledgment = ""
	ncfile.date_created = output_date
	
    # create the z dimensions.
	ncfile.createDimension('z',dim)
	ncfile.createDimension('z_dual',dim1)
	ncfile.createDimension('parameter',dim_parameter)
	ncfile.createDimension('scalar',dim_scalar)
	
    # create the variable
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.
	data_KMAX = ncfile.createVariable('KMAX','i4',('scalar'))
	data_KMAX.unit = '-'
	
	data_eta = ncfile.createVariable('eta','f8',('z'))
	data_eta.unit = 'm'
	data_eta.long_name = '\u03b7 coordinate of volume centroids: zero is at soil surface and and positive upward'
	
	data_eta_dual = ncfile.createVariable('etaDual','f8',('z_dual'))
	data_eta_dual.unit = 'm'
	data_eta_dual.long_name = '\u03b7 coordinate of volume interfaces: zero is at soil surface and and positive upward. '
	
	data_z = ncfile.createVariable('z','f8',('z'))
	data_z.unit = 'm'
	data_z.long_name = 'z coordinate  of volume centroids: zero is at the bottom of the column and and positive upward'
	
	data_z_dual = ncfile.createVariable('zDual','f8',('z_dual'))
	data_z_dual.unit = 'm'
	data_z_dual.long_name = 'z coordinate of volume interfaces: zero is at soil surface and and positive upward.'
	
	data_ic = ncfile.createVariable('ic','f8',('z'))
	data_ic.units = 'K'
	data_ic.long_name = 'Temperature initial condition'
		
	data_space_delta = ncfile.createVariable('spaceDelta','f8',('z_dual'))
	data_space_delta.unit = 'm'
	data_space_delta.long_name = 'Distance between consecutive controids, is used to compute gradients'
	
	data_soil_volume = ncfile.createVariable('volumeSoil','f8',('z'))
	data_soil_volume.unit = 'm'
	data_soil_volume.long_name = 'Volume of soil in each control volume'
	
	data_rheology_ID = ncfile.createVariable('rheologyID','f8',('z'))
	data_rheology_ID.units = '-'
	data_rheology_ID.long_name = 'label describing the rheology model'
	
	data_parameter_ID = ncfile.createVariable('parameterID','f8',('z'))
	data_parameter_ID.units = '-'
	data_parameter_ID.long_name = 'label identifying the set of parameters'

	data_soil_particles_density = ncfile.createVariable('soilParticlesDensity','f8',('parameter'))
	data_soil_particles_density.units = 'kg/m3'
	data_soil_particles_density.long_name = 'density of soil particles'
	
	data_thermal_conductivity_soil_particles = ncfile.createVariable('thermalConductivitySoilParticles','f8',('parameter'))
	data_thermal_conductivity_soil_particles.units = 'W/m2'
	data_thermal_conductivity_soil_particles.long_name = 'thermal conductivity of soil particles'
	
	data_specific_heat_capacity_soil_particles = ncfile.createVariable('specificThermalCapacitySoilParticles','f8',('parameter'))
	data_specific_heat_capacity_soil_particles.units = 'J/kg m3'
	data_specific_heat_capacity_soil_particles.long_name = 'specific thermal capacity of soil particles'
	
	data_theta_s = ncfile.createVariable('thetaS','f8',('parameter'))
	data_theta_s.units = '-'
	data_theta_s.long_name = 'adimensional water content at saturation'
	
	data_theta_r = ncfile.createVariable('thetaR','f8',('parameter'))
	data_theta_r.units = '-'
	data_theta_r.long_name = 'adimensional residual water content'
	
	data_melting_temperature = ncfile.createVariable('meltingTemperature','f8',('parameter'))
	data_melting_temperature.units = 'K'
	data_melting_temperature.long_name = 'melting temperature of soil water'
	
	data_par_1 = ncfile.createVariable('par1','f8',('parameter'))
	data_par_1.units = '-'
	data_par_1.long_name = 'SFCC parameter'
	
	data_par_2 = ncfile.createVariable('par2','f8',('parameter'))
	data_par_2.units = '-'
	data_par_2.long_name = 'SFCC parameter'
	
	data_par_3 = ncfile.createVariable('par3','f8',('parameter'))
	data_par_3.units = '-'
	data_par_3.long_name = 'SFCC parameter'
	
	data_par_4 = ncfile.createVariable('par4','f8',('parameter'))
	data_par_4.units = '-'
	data_par_4.long_name = 'SFCC parameter'
    
	
	## write data to variable.

	data_KMAX[0] = KMAX

	for i in range(0,dim):
		data_eta[i] = eta[i]
		data_z[i] = z[i]
		data_soil_volume[i] = soil_volume[i]
		data_ic[i] = ic[i]
		data_rheology_ID[i] = rheology_ID[i]
		data_parameter_ID[i] = parameter_ID[i]
		
	for i in range(0,dim1):
		data_eta_dual[i] = eta_dual[i]
		data_z_dual[i] = z_dual[i]
		data_space_delta[i] = space_delta[i]
		
	for i in range(0,dim_parameter):
		data_soil_particles_density[i] = soil_particles_density[i]
		data_thermal_conductivity_soil_particles[i] = thermal_conductivity_soil_particles[i]
		data_specific_heat_capacity_soil_particles[i] = specific_heat_capacity_soil_particles[i]
		data_theta_s[i] = theta_s[i]
		data_theta_r[i] = theta_r[i]
		data_melting_temperature[i] = melting_temperature[i]
		data_par_1[i] = par_1[i]
		data_par_2[i] = par_2[i]
		data_par_3[i] = par_3[i]
		data_par_4[i] = par_4[i]
	
	## close the file.
	ncfile.close()
	print ('\n\n***SUCCESS writing!  '+ output_file_name)

	
	return