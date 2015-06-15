import matplotlib.pyplot as plt
import numpy as np, numpy.linalg as nlg

def project2d(pts):
	"""
	Projects the points into 2d using SVD.
	"""
	pts = np.array(pts)
	mean = pts.mean(axis=0)
	cpts = pts - mean

	_, _, VT = nlg.svd(cpts, full_matrices=True)
	return VT[0:2,:].dot(cpts.T).T + mean[0:2]

def visualize2d(pts1, pts2=None, show=True, rtn=False):
	
	fig = plt.figure()

	pts1 = np.array(pts1)
	if pts1.shape[1] == 1:
		prj1 = np.c_[pts1, np.zeros(pts1.shape[0])]
	elif pts1.shape[1] == 2:
		prj1 = pts1
	else:
		prj1 = project2d(pts1)

	plt.scatter(prj1[:,0], prj1[:,1], color='blue')

	if pts2 is not None:
		pts2 = np.array(pts2)
		if pts2.shape[1] == 1:
			prj2 = np.c_[pts2, np.zeros(pts2.shape[0])]
		elif pts2.shape[1] == 2:
			prj2 = pts2
		else:
			prj2 = project2d(pts2)

		plt.scatter(prj2[:,0], prj2[:,1], color='red')

	if show:
		plt.show()

	if rtn: return fig


def drawCircle(center, radius, fig=None, show=True):
	circle = plt.Circle(center,radius,fill=False)

	if fig is None: fig = plt.gcf()
	fig.gca().add_artist(circle)

	if show:
		plt.show()