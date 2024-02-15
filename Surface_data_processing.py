import numpy as N
from sys import path
import matplotlib.pyplot as plt
from tracer.spatial_geometry import rotation_to_z, rotz
from scipy.stats import iqr
from python_utils_io import pickle_dump_data_in_file, pickle_load_file
import pickle

'''
These functions were originally created as part of the MSCA IF HEASeRS program funde by the European Comission (Grant agreement ID: 101027316) to interprete the data form an Alicona Infinite Focus white light interferometer and might need to be modified to be able to process other data.
'''

def load_original_profilo_data(xyzfile):
	xyz = N.loadtxt(open(xyzfile,'rt').readlines()[:-1], delimiter=' ', skiprows=1)
	return xyz

def load_profilo_data(xyzfile):
	xyz = N.loadtxt(open(xyzfile,'rt').readlines(), delimiter=' ', skiprows=1)
	return xyz

def save_profilo_data(xyz, fname):
	N.savetxt(fname, xyz, delimiter=' ', header='x y z')

def get_surface_dimensions(original_file):
	xyz = load_original_profilo_data(original_file)
	# Find the regular grid of measure and number of points per side
	unix, uniy = N.sort(N.unique(xyz[:,0])), N.sort(N.unique(xyz[:,1]))
	nx, ny = len(unix), len(uniy)
	x0, xmax, y0, ymax = unix[0], unix[-1], uniy[0], uniy[-1]
	return [nx, ny], [[x0,xmax], [y0,ymax]]

def xyz_to_stl(xyz, nx, ny, xdims=None, ydims=None, stl_fname=None):
	from ray_trace_utils.stl_utils import make_stl # this ca

	if xdims is None:
		xdims = [N.amin(xyz[:,0]), N.amax(xyz[:,0])]
	if ydims is None:
		ydims = [N.amin(xyz[:,1]), N.amax(xyz[:,1])]

	# filter according to dimensions
	in_x = N.logical_and(xyz[:,0]>=xdims[0], xyz[:,0]<=xdims[1])
	in_y = N.logical_and(xyz[:,1]>=ydims[0], xyz[:,1]<=ydims[1])
	valid = N.logical_and(in_x, in_y)
	xyz = xyz[valid]
	# assign points to triangles
	xyz = N.round(xyz, decimals=6)

	faces = []
	for j in range(ny-1):
		for i in range(nx-1):
			A = i+j*nx
			B = A+nx
			C = A+1
			D = B+1
			# Find shortest diagonal:
			BC = N.sqrt(N.sum((xyz[C]-xyz[B])**2))
			AD = N.sqrt(N.sum((xyz[D]-xyz[A])**2))
			# make common side this diagonal:
			t1, t2 = [A, B, C], [B, D, C]
			if AD<BC:
				t1, t2 = [A, B, D], [A, D, C]
			# only make face if none of the points has none as z
			if ~N.isnan(xyz[t1, 2]).any():
				faces.append(t1)
			if ~N.isnan(xyz[t2, 2]).any():
				faces.append(t2)

	print(xyz.shape[0], 'vertices and ', len(faces), 'faces')
	# build and save stl file
	if stl_fname is None:
		stl_fname = xyzfile[:-3]+'stl'

	# Make STL file
	make_stl(xyz, N.array(faces), filename=stl_fname)

def repair_surface(xyz, fname=None, tol=1e-4):
	'''
	xyz: xyz data form the profilometer
	fname: file save location
	tol: tolerance to floating point precision to find missing grid elements in unit of measurement.
	'''
	# Find the regular grid of measure and number of points per side
	unix, uniy = N.sort(N.unique(xyz[:,0])), N.sort(N.unique(xyz[:,1]))
	nx, ny = len(unix), len(uniy)
	x0, xmax, y0, ymax = unix[0], unix[-1], uniy[0], uniy[-1]

	# Check the number of points measures versus the theoretical full grid
	# if the number of points measured is lower than the full grid, there are points missing and we nned to fill them with None values.
	# if the number of points measured is larger than the theoretical gird then there is something wrong with the grid estimated. It shoudl not happen!
	if nx*ny<xyz.shape[0]:
		print("ERROR in measurement grid estimation!")
		stop
		
	bad = 0
	for j in range(ny):
		for i in range(nx):
			# node?
			idx = i+j*nx
			#xth, yth = N.round([x0+i*dx, ymax-j*dy], decimals=6)
			xth, yth = unix[i], uniy[::-1][j]
			if idx >= xyz.shape[0]:
				xyz = N.insert(xyz, idx, [xth, yth, None], axis=0)
			x, y = N.round(xyz[idx, :2], decimals=6)
			xdif, ydif = N.abs([x-xth, y-yth])
			
			if (xdif>tol or ydif>tol):
				xyz = N.insert(xyz, idx, [xth, yth, None], axis=0)
				bad +=1
	print(bad, 'bad nodes')
	if fname is not None:
		N.savetxt(fname, xyz, delimiter=' ')
	return xyz, nx, ny

def autocorrelation(z, nx, ny, savefile):
	# Calculate autocorrelation using 2D FFT
	zs = N.reshape(z, (ny, nx))

	fftmap = N.fft.fft2(zs)
	PSD = fftmap * N.conjugate(fftmap) # This is not strictly the PSD as it does not include the constant factor in front of the moduluis square of the fft
	ACF = N.fft.ifft2(PSD)
	ACF = N.fft.fftshift(ACF).real
	ACF /= N.amax(ACF)
	if savefile:
		data = {'ACF_2D':ACF}
		pickle_dump_data_in_file(data, savefile)

	return ACF

def surface_stats(xyzfile, original_file, savefile=None):
	# get dimensions form original, unrotated file
	ns, ds = get_surface_dimensions(original_file)
	nx, ny = ns

	# Surface x_y evenness:
	x, y, z = load_profilo_data(xyzfile).T		
	xres, yres = (N.amax(x)-N.amin(x))/float(nx), (N.amax(y)-N.amin(y))/float(ny)

	# Freedman-Diaconis histogram bin number
	IQR = iqr(z)
	n = len(x)
	h = 2.*IQR*n**(-1./3.)
	nbins_FD = int(N.ceil((N.amax(z)-N.amin(z))/h))

	# Statistical moments
	m = []
	avgz = N.sum(z)/n
	for i in range(1, 5):
		moment = 1./n*N.sum((z-avgz)**i)
		m.append(moment)

	# statistical indicators
	mean = avgz
	std = n/(n-1.)*m[1]**0.5
	skew = m[2]/std**3
	excess_kurtosis = m[3]/(std**4)-3.
	print('min:', N.amin(z), ', max:', N.amax(z), ', mean:',mean, ', sigma:',std, ', skew:',skew, ', excess kurtosis:',excess_kurtosis)

	if savefile is not None:
		d = {'nx':nx, 'ny':ny, 'xres':xres, 'yres':yres, 'nbins_FD':nbins_FD, 'mean':mean, 'std':std, 'skew':skew, 'excess_kurtosis':excess_kurtosis}
		pickle_dump_data_in_file(d, savefile)

	return nx, ny, nbins_FD, mean, std, skew, excess_kurtosis

def calculate_ACL(stats_file, ACL_rad_file, ACL_threshold):

	stats = pickle_load_file(stats_file)
	ACF_data = pickle_load_file(ACL_rad_file)
	avgACF = (ACF_data['distbins'][1:]+ACF_data['distbins'][:-1])/2.
	tau_rad = avgACF[ACF_data['ACF_rad']<ACL_threshold][0]
	stats.update({'tau_rad':tau_rad})
	pickle_dump_data_in_file(stats, stats_file)
	return tau_rad

def get_heights_distribution(xyzfile, density=True, savefile=None):
	# histogram
	xyz = load_profilo_data(xyzfile)
	pdf, bins = N.histogram(z, bins=nbins, density=density)
	if savefile is not None:
		with open(savefile, 'w') as fo:
			pickle.dump([pdf, bins], fo)
	return bins, pdf

def plot_heights_distribution(xyzfile, text=False, hist_data_file=None, **kwargs):

	if hist_data_file is None:
		pdf, bins = get_heights_distribution(xyz, density=True)
	else:
		with open(hist_data_file, 'r') as fo:
			pdf, bins = pickle.load(fo)

	# plot
	bins = (bins[1:]+bins[:-1])/2.
	hist_data = plt.plot(bins, pdf, **kwargs)

	plt.xlabel('z (${\mathrm{\mu}}$m)')
	plt.ylabel('$\overline{\mathrm{PDF}}$')
	
	if text:
		meantxt = '${\overline{z}}=%s}$ ${\mu}$m'%N.round(mean, decimals=4)
		stdtxt = '${s=%s}$ ${\mu}$m'%N.round(std, decimals=4)
		skewtxt = '${g=%s}$'%N.round(skew, decimals=4)
		kurttxt = '${\kappa=%s}$'%N.round(excess_kurtosis, decimals=4)

		plt.text(plt.xlim()[0], plt.ylim()[1], meantxt+'\n'+stdtxt+'\n'+skewtxt+'\n'+kurttxt, va='top', ha='left')

def get_radial_ACF(xyzfile, statsfile, savefile=None):

	x, y, z = load_profilo_data(xyzfile).T

	statdict = pickle_load_file(statsfile)
	nx = statdict['nx']
	ny = statdict['ny']
	xres = statdict['xres']
	yres = statdict['yres']
	
	ACF = autocorrelation(z, nx, ny, savefile=savefile[:-3]+'2D')

	# Calculate ACF pixels radial distance
	dists = N.reshape(N.sqrt(x**2+y**2), (ny,nx))

	# Evaluate average ACL at distances 0 ... max
	maxdist = N.amin([N.amax(x), N.amax(y)])
	maxres = 2.*N.amax([xres, yres])
	distbins = N.arange(0., maxdist, maxres)
	rad_ACF = []
	for i in range(len(distbins)-1):
		distmin, distmax = distbins[i:i+2]
		valid = N.logical_and(dists>=distmin, dists<=distmax)
		rad_ACF.append(N.average(ACF[valid]))
	rad_ACF = N.array(rad_ACF)/rad_ACF[0]
	if savefile is not None:
		d = {'distbins': distbins, 'ACF_rad':rad_ACF}
		pickle_dump_data_in_file(d, savefile)

	return distbins, rad_ACF

def plot_2D_ACF(ACF_2D_file, stats_file):

	ACF_2D = pickle_load_file(ACF_2D_file)['ACF_2D']

	stats = pickle_load_file(stats_file)
	nx, ny = stats['nx'], stats['ny']
	xres, yres = stats['xres'], stats['yres']
	xbins = N.arange(nx+1)*xres
	ybins = N.arange(ny+1)*yres
	X, Y = N.meshgrid(xbins, ybins)
	AR = (nx*xres)/(ny*yres)

	top = 0.95
	bottom = 0.1

	ax = plt.axes([0.1, bottom, AR*(top-bottom), top-bottom])
	ax.set_aspect('equal', anchor='SW')
	img = ax.pcolormesh(X, Y, ACF_2D)
	cax = plt.axes([0.9,bottom,0.025,top-bottom])
	plt.colorbar(img, cax = cax)

def plot_radial_ACF(ACF_rad_file, ACL, **kwargs):
	data = pickle_load_file(ACF_rad_file)
	lag = data['distbins']
	ACF = data['ACF_rad']
	ACF = N.hstack([ACF[0], ACF])
	p = plt.plot(lag, ACF, **kwargs)
	c = p[0].get_color()
	plt.hlines(1./N.exp(1.), 0, ACL, zorder=100)
	plt.text(0, 1./N.exp(1.), r'${\frac{1}{e}}$   ', ha='right', va='top')
	plt.vlines(ACL, 0, 1./N.exp(1.), linestyle='--', lw=1, color=c)
	plt.xlabel('Radial lag (${\mathrm{\mu}}$m)')
	plt.ylabel('$\overline{\mathrm{ACF}}$')

def get_2D_ACL(ACF, correlation_threshold=1./N.exp(1.)):
	zone = ACF>=correlation_threshold
	hor_borders = N.argwhere(zone[:,:-1]^zone[:,1:]).T
	ver_borders = N.argwhere(zone[:-1,:]^zone[1:,:]).T
	borders = N.array([N.hstack((hor_borders[1], ver_borders[1])),N.hstack((hor_borders[0], ver_borders[0]))], dtype=float)

def plot_2D_ACL(borders):
	plt.scatter(borders[0], borders[1], color='r', s=1)

def plot_polar_ACL(xyz, nx, ny, borders):

	x, y, z = xyz.T
	xres, yres = (N.amax(x)-N.amin(x))/float(nx), (N.amax(y)-N.amin(y))/float(ny)

	center = N.array(ACF.shape)/2
	borders[0] -= center[1]
	borders[1] -= center[0]
	# scale borders

	borders[0] = xres*borders[0]
	borders[1] = yres*borders[1]
	rs = N.sqrt(N.sum(borders**2, axis=0))
	thetas = N.arctan2(borders[1], borders[0])#*180./N.pi
	sort = N.argsort(thetas)
	thetas = thetas[sort]
	rs = rs[sort]

	dthetas = thetas[1:]-thetas[:-1]
	avg_r = N.sum((rs[1:]+rs[:-1])/2.*dthetas)/(2.*N.pi)
	plt.plot(thetas, rs)
	plt.plot(thetas, N.ones(len(thetas))*avg_r)


def interpolate_Nones(xyz):
	# go though the surface z-Nones. 
	# For each z-None, find the closest axis-aligned points and interpolate over the lines resulting in 2 points. Then assign weights to these two interpolated points as a function of the inverse of the horizontal distance to known points and determine the z-position of the nodes based on the weighted average of the z positions of the interpolated points.
	
	Nones_loc = N.nonzero(N.isnan(xyz[:,2]))[0]

	unix, uniy = N.sort(N.unique(xyz[:,0])), N.sort(N.unique(xyz[:,1]))
	for zNone in Nones_loc:

		# Lines crossing the None node
		xline = xyz[:,0] == xyz[zNone,0]
		yline = xyz[:,1] == xyz[zNone,1]

		# Exclude z-Nones:
		xline[N.isnan(xyz[:,2])] = False
		yline[N.isnan(xyz[:,2])] = False

		xline, yline = N.nonzero(xline)[0], N.nonzero(yline)[0]

		# Nearest x and y points, below and over the unknown point and that are not None in z:
		near_on_x, near_on_y = [], []
		below, above = xyz[yline,0]<xyz[zNone,0], xyz[yline,0]>xyz[zNone,0]

		if below.any():
			near_on_x.append(yline[below][-1])
		if above.any():
			near_on_x.append(yline[above][0])

		below, above = xyz[xline,1]<xyz[zNone,1], xyz[xline,1]>xyz[zNone,1]

		if below.any():
			near_on_y.append(xline[below][-1])
		if above.any():
			near_on_y.append(xline[above][0])

		# Neighbours:
		zs = N.hstack((xyz[near_on_x,2], xyz[near_on_y,2]))

		# Weights
		d_on_xs = N.abs(xyz[near_on_x,0]-xyz[zNone,0])
		d_on_ys = N.abs(xyz[near_on_y,1]-xyz[zNone,1])
		ws = N.hstack((1./d_on_xs, 1./d_on_ys))
	
		# Interpolation via weighted sums:
		xyz[zNone,2] = N.sum(zs*ws)/N.sum(ws)

	return xyz

def correct_orientation(xyz, nx, ny):
	from scipy.optimize import minimize
	# Subtract everage height:
	avg_z = N.average(xyz[~N.isnan(xyz[:,2]),2])
	xyz[:,2] -= avg_z
	# find mid-plane normal:
	def distsq_to_plane(plane, args):
		xyz = args[:]
		a, b, c, d = plane
		# ax + by + cz + d = 0 -> z = -a/cx, -b/cy, -d/c
		zplane = -a/c*xyz[:,0] -b/c*xyz[:,1] -d/c
		return N.sum((zplane-xyz[:,2])**2)
		
	def fit_plane(xyz):
		plane = N.array([0,0,1,0])
		res = minimize(distsq_to_plane, plane, args=(xyz), method='Nelder-Mead')
		a, b, c, d = res.x
		normal = N.hstack([a,b,c])
		return normal/N.sqrt(N.sum(normal**2))

	normal_best_fit = fit_plane(xyz)
	rot = rotation_to_z(normal_best_fit)
	# rotate points so that plane normal goes to 0,0,1
	xyz = N.dot(xyz, rot[:3,:3])
	# align x on previous x
	# determine x:
	x = xyz[nx-1,:2]-xyz[0,:2]
	# find angle of rotation:
	ang = N.arctan2(x[1], x[0])
	# rotate
	xyz = N.dot(xyz, rotz(ang)[:3,:3])
	# Place on average height
	xyz[:,2] -= N.average(xyz[:,2])
	return xyz

def center_surface(xyz):
	x, y, z = xyz.T
	#halfx, halfy = (N.amax(x)-N.amin(x))/2., (N.amax(y)-N.amin(y))/2.
	xmid, ymid = (N.amax(x)+N.amin(x))/2., (N.amax(y)+N.amin(y))/2.
	
	xyz[:,0] -= xmid
	xyz[:,1] -= ymid

	return xyz

