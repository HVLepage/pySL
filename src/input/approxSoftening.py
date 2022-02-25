import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def approx1D(params):
    def eval_integral(y1_vals,y2_vals,z1_vals,z2_vals,sigSofty,sigSoftz,V,N):
        A=1/(np.pi**0.25*sigSofty**0.5)
        B=1/(np.pi**0.25*sigSoftz**0.5)
        f= lambda y1,y2,z1,z2: np.exp(-y1**2/(sigSofty**2))*np.exp(-y2**2/(sigSofty**2))*np.exp(-z1**2/(sigSoftz**2))*np.exp(-z2**2/(sigSoftz**2))*((z1-z2)**2+(y1-y2)**2)**0.5 * A**4 * B**4
        integral=np.sum(f(y1_vals,y2_vals,z1_vals,z2_vals))
        integral=integral*V/N

        return integral
        
    N=20000000 #number of points in Monte Carlo, should be sufficient as is
    miny=-4*params["sigSofty"] #e^-16 at boundaries of integration domain
    maxy=4*params["sigSofty"]
    minz=-4*params["sigSoftz"]
    maxz=4*params["sigSoftz"]

    rangey=maxy-miny
    rangez=maxz-minz

    V=rangey**2 * rangez**2

    y1_vals=miny+rangey*np.random.rand(N)
    y2_vals=miny+rangey*np.random.rand(N)
    z1_vals=minz+rangez*np.random.rand(N)
    z2_vals=minz+rangez*np.random.rand(N)
    
    integral=eval_integral(y1_vals,y2_vals,z1_vals,z2_vals,params["sigSofty"],params["sigSoftz"],V,N)

    return integral

def approx2D(params):

    A=1/(np.pi**0.25*params["sigSoftz"]**0.5)
    f= lambda y,x: np.exp(-x**2/(params["sigSoftz"]**2))*np.exp(-y**2/(params["sigSoftz"]**2))*((np.abs((x-y))))*A**4
    integral=integrate.dblquad(f,-np.inf,np.inf,lambda x: -np.inf, lambda x: np.inf)[0]


    return integral
