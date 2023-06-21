# Transport routines for 2D transport: diffusion and
# advection. The routines use JIT compilation.
# Advection scheme mainly follows that of Lin & Rood 1996:
# https://doi.org/10.1175/1520-0493(1996)124<2046:MFFSLT>2.0.CO;2
# Diffusion is Eulerian. Off-diagonal diffusion follows
# du Toit et al. 2018: https://doi.org/10.1016/j.cpc.2018.03.004
#
# TODO:combine both 1D diffusion matrices into a single
# matrix in in a single loop.

import numpy as np
from numba import jit


@jit(nopython=True)
def eulerian_diffusion_zz(Dzz, dx, mva):
    """
    Creates matrix for one-dimensional Eulerian diffusion in z-direction 
    using central differencing.

    Args:
        Dzz (array): Diffusion (m2/s) in the z-direction (alitude) at all points.
        dx (float): Uniform grid spacing in the z-direction.
        mva (array): Molar density of air (mol/m3) for each vertical layer. 

    Returns:
        array: A matrix containing diffusivity in z-direction.
    """
    dx2 = dx*dx
    nx = Dzz.shape[0]
    ny = Dzz.shape[1]
    M = np.repeat(mva, ny).reshape(-1, ny).T.flatten()  # jit-compatible
    Dzzf = Dzz.T.flatten()
    # Want A @ bold = bnew
    A = np.zeros((nx*ny, ny*nx))

    for i in range(nx, ny*nx-nx-1):
        if (i % nx) and ((i % nx) != (nx-1)):
            A[i, i] = -((M[i-1]+M[i])*Dzzf[i] + Dzzf[i+1]
                        * (M[i]+M[i+1]))/(4*dx2*M[i])
            A[i, i+1] = Dzzf[i+1] * (M[i]+M[i+1])/(4*dx2*M[i+1])
            A[i, i-1] = Dzzf[i] * (M[i-1]+M[i])/(4*dx2*M[i-1])
    # Now need to deal with boundaries
    # Create artificial points outside to deal with it
    for i in range(1, nx-1):
        # South boundary
        A[i, i] = -((M[i-1]+M[i])*Dzzf[i] + Dzzf[i+1]
                    * (M[i]+M[i+1]))/(4*dx2*M[i])
        A[i, i+1] = Dzzf[i+1] * (M[i]+M[i+1])/(4*dx2*M[i+1])
        A[i, i-1] = Dzzf[i] * (M[i-1]+M[i])/(4*dx2*M[i-1])
    for i in range(ny*nx-nx+1, ny*nx-1):
        # North boundary
        A[i, i] = -((M[i-1]+M[i])*Dzzf[i] + Dzzf[i+1]
                    * (M[i]+M[i+1]))/(4*dx2*M[i])
        A[i, i+1] = Dzzf[i+1] * (M[i]+M[i+1])/(4*dx2*M[i+1])
        A[i, i-1] = Dzzf[i] * (M[i-1]+M[i])/(4*dx2*M[i-1])
    for i in range(nx, nx*(ny-1), nx):
        # Bottom boundary
        A[i, i] = -(Dzzf[i+1]*(M[i]+M[i+1]))/(4*dx2*M[i])
        A[i, i+1] = Dzzf[i+1] * (M[i]+M[i+1])/(4*dx2*M[i+1])
    for i in range(2*nx-1, ny*nx-nx, nx):
        # Top boundary
        A[i, i] = -(Dzzf[i]*(M[i-1]+M[i]))/(4*dx2*M[i])
        A[i, i-1] = Dzzf[i] * (M[i-1]+M[i])/(4*dx2*M[i-1])

    # Corners
    # South Pole
    A[0, 0] = -(Dzzf[1])*(M[0]+M[1])/(4*dx2*M[0])
    A[0, 1] = Dzzf[1] * (M[0]+M[1])/(4*dx2*M[1])
    # North Top
    A[-1, -1] = -(Dzzf[-1])*(M[-2]+M[-1])/(4*dx2*M[-1])
    A[-1, -2] = Dzzf[-1] * (M[-2]+M[-1])/(4*dx2*M[-2])
    # South Top
    A[nx-1, nx-1] = -((M[nx-2]+M[nx-1])*Dzzf[nx-1])/(4*dx2*M[nx-1])
    A[nx-1, nx-2] = Dzzf[nx-1] * (M[nx-2]+M[nx-1])/(4*dx2*M[nx-2])
    # North Pole
    A[-nx, -nx] = -(Dzzf[-nx+1])*(M[-nx]+M[-nx+1])/(4*dx2*M[-nx])
    A[-nx, -nx+1] = Dzzf[-nx+1] * (M[-nx]+M[-nx+1])/(4*dx2*M[-nx+1])

    return A


@jit(nopython=True)
def eulerian_diffusion_yy(Dyy, dy, cosc_in, cose_in):
    """
    Creates matrix for one-dimensional Eulerian diffusion in y-direction 
    using central differencing.

    Args:
        Dzz     (array): Diffusion (m2/s) in the y-direction (alitude) at all points.
        dy      (float): Uniform grid spacing in the y-direction.
        cosc_in (array): Cosine of latitude at grid centre 
        cose_in (array):Cosine of latitude at grid edge 

    Returns:
        array: A matrix containing diffusivity in y-direction.

    """
    # Set up as matrix
    dy2 = dy*dy
    nx = Dyy.shape[0]
    ny = Dyy.shape[1]
    cosc = np.repeat(cosc_in, nx)
    cose = np.repeat(cose_in[:-1], nx)
    Dyyf = Dyy.T.flatten()
    A = np.zeros((nx*ny, ny*nx))
    for i in range(nx, ny*nx-nx-1):
        if (i % nx) and ((i % nx) != (nx-1)):
            A[i, i] = - (cose[i] * Dyyf[i] + cose[i+nx]
                         * Dyyf[i+nx])/(dy2*cosc[i])
            A[i, i+nx] = cose[i+nx] * Dyyf[i+nx]/(dy2*cosc[i])
            A[i, i-nx] = cose[i] * Dyyf[i]/(dy2*cosc[i])
    # Now need to deal with boundaries
    # Create artificial points outside to deal with it
    for i in range(1, nx-1):
        # South boundary
        A[i, i] = - (cose[i] * Dyyf[i] + cose[i+nx] * Dyyf[i+nx])/(dy2*cosc[i])
        A[i, i+nx] = cose[i+nx] * Dyyf[i+nx]/(dy2*cosc[i])
    for i in range(ny*nx-nx+1, ny*nx-1):
        # North boundary
        A[i, i] = - (cose[i] * Dyyf[i])/(dy2*cosc[i])
        A[i, i-nx] = cose[i] * Dyyf[i]/(dy2*cosc[i])
    for i in range(nx, nx*(ny-1), nx):
        # Bottom boundary
        A[i, i] = - (cose[i] * Dyyf[i] + cose[i+nx] * Dyyf[i+nx])/(dy2*cosc[i])
        A[i, i+nx] = cose[i+nx] * Dyyf[i+nx]/(dy2*cosc[i])
        A[i, i-nx] = cose[i] * Dyyf[i]/(dy2*cosc[i])
    for i in range(2*nx-1, ny*nx-nx, nx):
        # Top boundary
        A[i, i] = - (cose[i] * Dyyf[i] + cose[i+nx] * Dyyf[i+nx])/(dy2*cosc[i])
        A[i, i+nx] = cose[i+nx] * Dyyf[i+nx]/(dy2*cosc[i])
        A[i, i-nx] = cose[i] * Dyyf[i]/(dy2*cosc[i])
    # Corners
    # South Pole
    A[0, 0] = - (cose[nx] * Dyyf[nx])/(dy2*cosc[0])
    A[0, nx] = cose[nx] * Dyyf[nx]/(dy2*cosc[0])
    # North Top
    A[-1, -1] = - (cose[-1] * Dyyf[-1])/(dy2*cosc[-1])
    A[-1, -nx-1] = cose[-1] * Dyyf[-1]/(dy2*cosc[-1])
    # South Top
    A[nx-1, nx-1] = - (cose[nx+nx-1] * Dyyf[nx+nx-1])/(dy2*cosc[nx-1])
    A[nx-1, nx+nx-1] = cose[nx+nx-1] * Dyyf[nx+nx-1]/(dy2*cosc[nx-1])
    # North Pole
    A[-nx, -nx] = - (cose[-nx] * Dyyf[-nx])/(dy2*cosc[-nx])
    A[-nx, -nx-nx] = cose[-nx] * Dyyf[-nx]/(dy2*cosc[-nx])

    return A


@jit(nopython=True)
def diff_matmul(cin, mva, A, dt):
    """Simple matrix multiplication for Eulerian diffusion (not used)"""
    nza = cin.shape[0]
    nya = cin.shape[1]
    return cin + ((A @ (cin*mva).T.flatten())*dt).reshape((nya, nza)).T/mva


@jit(nopython=True)
def rk4(cin, mva, Amat, dt):
    """Runga-Kutta 4 solver for diffusion"""
    nza = cin.shape[0]
    nya = cin.shape[1]
    cinf = (cin*mva).T.flatten()
    A = np.dot(Amat, cinf)
    B = np.dot(Amat, (cinf + dt * A / 2.0))
    C = np.dot(Amat, (cinf + dt * B / 2.0))
    D = np.dot(Amat, (cinf + dt * C))
    cinf += dt / 6.0 * (A + 2.0 * (B + C) + D)
    return cinf.reshape((nya, nza)).T/mva

# Mixed derivative diffusion


@jit(nopython=True)
def interpolate1D(Fold, xnew, xold):
    """
    Fast linear 1D interpolation for Dyz diffusion

    Args:
        Fold (array): Field to interpolate.
        xnew (array): Old cartesian coordinate locations.
        xold (array): New cartesian coordinate locations.

    Returns:
        array: Interpolated field.
    """
    Fnew = np.zeros((len(xnew), Fold.shape[1]))
    for i in range(1, len(xnew)-1):
        idx2 = np.argwhere(xold > xnew[i]).min()
        idx1 = np.argwhere(xold <= xnew[i]).max()
        Fnew[i, :] = (Fold[idx1, :]*(xold[idx2] - xnew[i]) + Fold[idx2, :]
                      * (xnew[i] - xold[idx1])) / (xold[idx2] - xold[idx1])
    return Fnew


@jit(nopython=True)
def diff_Q0(Q0in, dx, epsilon=1e-3):
    """
    Calculate central difference of d/dx
    dq/dx -> inf when q -> 0. Therefore place threshold for mole fraction.

    Args:
        Q0in (array): Mole fraction field
        dx (array): Uniform grid spacing.
        epsilon (float, optional): Cut-off value for off-diagonal diffusion. Defaults to 1e-3.

    Returns:
        array: Central differenced mole fraction field.
    """
    nx = Q0in.shape[0]
    ny = Q0in.shape[1]
    dqdx = np.zeros((nx, ny))
    dqdx[1:-1, :] = np.where(Q0in[1:-1, :] > epsilon, (Q0in[2:, :] -
                             Q0in[0:-2, :])/(2*np.expand_dims(dx[1:-1], 1)*Q0in[1:-1, :]), 0)
    dqdx[0, :] = np.where(Q0in[0, :] > epsilon, -(3*Q0in[0, :] -
                          4*Q0in[1, :] + Q0in[2, :]) / (2*dx[0]*Q0in[0, :]), 0)
    dqdx[-1, :] = np.where(Q0in[-1, :] > epsilon, (3*Q0in[-1, :] -
                           4*Q0in[-2, :] + Q0in[-3, :]) / (2*dx[-1]*Q0in[-1, :]), 0)
    return dqdx


@jit(nopython=True)
def u_diffusive(Q0, Dzy, z, y, ze, ye, dz, dy):
    """    
    A velocity field, effectively Dyz*1/q*d/dx, which can be 
    passed to an advection scheme for mixed derivative diffusion. 

    Args:
        Q0 (array): Mole fraction field
        Dzy (array): Diffusivitiy on Arakawa C-grid
        z (array): z at grid centres
        y (array): y at grid centres
        ze (array): z at grid edges
        ye (array): y at grid edges
        dz (array): Uniform spacing of z
        dy (array): Uniform spacing of y

    Returns:
        array: w, advective representation of diffusive field in z-direction
        array: v, advective representation of diffusive field in y-direction
    """
    nz = Dzy.shape[0]
    ny = Dzy.shape[1]
    dqdz = diff_Q0(Q0, dz)
    dqdy = diff_Q0(Q0.T, dy).T
    w_c = np.zeros((nz, ny))
    v_c = np.zeros((nz, ny))
    w_c = Dzy * dqdy
    v_c = Dzy * dqdz
    # Interpolate from c to a grid
    w = interpolate1D(w_c, ze, z)
    v = interpolate1D(v_c.T, ye, y).T
    return w, v

# Advection routines


@jit(nopython=True)
def cross_terms(Q0, Cz):
    """Compute 'cross terms' following Eq. 3.11 Lin & Rood 96 """
    nx = Q0.shape[0]
    ny = Q0.shape[1]
    Q0 = np.hstack((Q0, np.zeros_like(Q0[:, -1:])))  # jit-compatible
    Qg = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            Js = int(j - np.sign(Cz[i, j]))
            Qg[i, j] = 0.5 * ((2*Q0[i, j]) + abs(Cz[i, j])
                              * (Q0[i, Js] - Q0[i, j]))
    return Qg


@jit(nopython=True)
def deltaQ(Qg):
    """Compute delta_Q using Eq. 4.1 Lin & Rood 96"""
    nx = Qg.shape[0]
    dQg = np.zeros_like(Qg)
    Qg = np.vstack((Qg, np.zeros_like(Qg[-1:, :])))
    # Don't need I notation from LR96 if Cz < 1
    for i in range(1, nx-1):
        if i < 2 or i > (nx-3):
            dQg[i, :] = (Qg[i+1, :] - Qg[i-1, :])/2
        else:
            dQg[i, :] = (8.*(Qg[i+1, :] - Qg[i-1, :]) +
                         Qg[i-2, :] - Qg[i+2, :])/12.

    return dQg


@jit(nopython=True)
def Qmonotonic(Qg, dQg):
    """Compute Qmono using Eq.5 Lin et al. 94"""
    nx = Qg.shape[0]
    Qmono = np.zeros_like(Qg)
    Qg = np.vstack((Qg, np.zeros_like(Qg[-1:, :])))
    for i in range(1, nx-1):
        for j in range(Qg.shape[1]):
            Qmin = np.min(np.array([Qg[i, j], Qg[i+1, j], Qg[i-1, j]]))
            Qmax = np.max(np.array([Qg[i, j], Qg[i+1, j], Qg[i-1, j]]))
            Qmono[i, j] = np.sign(
                dQg[i, j]) * np.min(np.array([abs(dQg[i, j]), 2*(Qg[i, j] - Qmin), 2*(Qmax - Qg[i, j])]))
    return Qmono


@jit(nopython=True)
def calc_As(Qg, Qmono):
    """Compute AL, AR and A6"""
    nx = Qmono.shape[0]
    ny = Qmono.shape[1]
    # Use equations from P589, Carpenter et al. 1990
    AL = np.zeros_like(Qg)
    AR = np.zeros_like(Qg)
    A6 = np.zeros_like(Qg)
    AL[0, :] = np.where(np.isfinite(np.sqrt(Qg[0, :]/Qg[1, :])*Qg[0, :]),
                        np.sqrt(Qg[0, :]/Qg[1, :])*Qg[0, :], np.zeros(ny))**2
    for i in range(1, nx):
        AL[i, :] = 0.5 * (Qg[i-1, :] + Qg[i, :]) - 1/6 * \
            (Qmono[i, :] - Qmono[i-1, :])
    AR[:-1, :] = AL[1:, :]
    AR[-1, :] = np.where(np.isfinite(np.sqrt(Qg[-1, :]/Qg[-2, :])*Qg[-1, :]),
                         np.sqrt(Qg[-1, :]/Qg[-2, :])*Qg[-1, :], np.zeros(ny))**2
    A6 = 6*(Qg - 0.5*(AL + AR))
    # Constrain using first constraint in Appendix C of Lin & Rood 1996
    dA = AR - AL
    for i in range(nx):
        for j in range(ny):
            if Qmono[i, j] == 0:
                AL[i, j] = Qg[i, j]
                AR[i, j] = Qg[i, j]
                A6[i, j] = 0
            elif A6[i, j]*dA[i, j] < -(dA[i, j])**2:
                A6[i, j] = 3*(AL[i, j] - Qg[i, j])
                AR[i, j] = AL[i, j] - A6[i, j]
            elif A6[i, j]*dA[i, j] > (dA[i, j])**2:
                A6[i, j] = 3*(AR[i, j] - Qg[i, j])
                AL[i, j] = AR[i, j] - A6[i, j]
    return AR, AL, A6


@jit(nopython=True)
def calc_flux(Qg, Qmono, Cz, dtdx, cosc, cose, mvae, mva, dz=np.array([0, 0]), u=None, horizontal=True):
    """
    Calculates the flux between grid cells

    Args:
        Qg (array): Cross-terms calculated by function cross_terms
        Qmono (array): Monotonic concentrations calculated by function Qmonotonic
        Cz (array): Curant number
        dtdx (float): Time step divided by grid spacing
        cosc (array): Cosine of latitude at grid centres
        cose (array): Cosine of latitude at grid edges
        mvae (array):  Molar density of air (mol/m3) for each vertical layer at cell edges.
        mva (array):  Molar density of air (mol/m3) for each vertical layer at cell centre.
        dz (array, optional): Uniform grid spacing in z-direction. Only needed for z-direction flux. Defaults to np.array([0,0]).
        u (array, optional): Wind velocity. Defaults to None.
        horizontal (bool, optional): Whether flux transport is in horizontal (True) or vertical (False). Defaults to True.

    Returns:
        array: Flux between grid cells
    """
    nx = Qg.shape[0]
    ny = Qg.shape[1]
    Qg = np.vstack((Qg, np.zeros_like(Qg[-1:, :])))
    AR, AL, A6 = calc_As(Qg, Qmono)
    # Calculate fluxes using 1.12 of Coella & Woodward 1983
    flux = np.zeros_like(Qg)
    for i in range(nx):
        for j in range(ny):
            if Cz[i, j] > 0:
                flux[i, j] = AR[i-1, j] - 0.5 * Cz[i, j] * \
                    (AR[i-1, j] - AL[i-1, j] -
                     (1 - (2./3.) * Cz[i, j]) * A6[i-1, j])
            else:
                flux[i, j] = AL[i, j] - 0.5 * Cz[i, j] * \
                    (AR[i, j] - AL[i, j] + (1 + (2./3.)*Cz[i, j]) * A6[i, j])
    # Follow Coella & Woodward 1983 to calculate fluxes
    ffy = np.zeros_like(Qg)
    dqv = np.zeros_like(Qg)
    if horizontal == True:
        ffy[:-1, :] = flux[:-1, :] * u[:-1, :] * dtdx * \
            np.expand_dims(mva[:, 0].T, 0) * np.expand_dims(cose[:-1], 1)
        for j in range(ny):
            ffy[0, j] = flux[0, j] * \
                np.max(np.array([u[0, j] * dtdx, 0])) * mva[j, 0] * cose[0]
        for i in range(nx):
            dqv[i, :] = (ffy[i, :] - ffy[i+1, :])/cosc[i]
    else:
        ffy[:-1, :] = flux[:-1, :] * u[:-1, :] * mvae[:-1, :] * dtdx * dz[0]
        for j in range(ny):
            ffy[0, j] = flux[0, j] * \
                np.max(np.array([u[0, j], 0])) * mvae[0, 0] * dtdx*dz[0]
        for i in range(nx):
            dqv[i, :] = (ffy[i, :] - ffy[i+1, :])/dz[i]

    return dqv[:-1, :]


@jit(nopython=True)
def linrood_advection(Q0, v, w, dy, dz, dt, mva, mvae, cosc, cose):
    """
    Wrapper to compute advection in horizonal and vertical.

    Args:
        Q0 (array): Mole fraction field.
        v (array): Horizontal velocity field.
        w (array): Vertical velocity field.
        dy (array): Uniform spacing of y.
        dz (array): Uniform spacing of z.
        dt (float): Time step.
        mva (array): Molar density of air (mol/m3) for each vertical layer at cell centre.
        mvae (array): Molar density of air (mol/m3) for each vertical layer at cell edges.
        cosc (array): Cosine of latitude at grid centres.
        cose (array): Cosine of latitude at grid edges.

    Returns:
        array: Advected mole fraction field.
    """
    Cy = 0.5*dt/np.expand_dims(dy, 0) * (v[:, :-1] + v[:, 1:])
    Cz = 0.5*dt/np.expand_dims(dz, 1) * (w[:-1, :] + w[1:, :])
    Qgz = cross_terms(Q0, Cy)
    Qgy = cross_terms(Q0.T, Cz.T).T
    dQgz = deltaQ(Qgy)
    dQgy = deltaQ(Qgy.T).T
    Qmonoz = Qmonotonic(Qgz, dQgz)
    Qmonoy = Qmonotonic(Qgy.T, dQgy.T).T
    dqvz = calc_flux(Qgz, Qmonoz, Cz, dt /
                     dz[0], cosc, cose, mvae, mva, dz=dz, u=w, horizontal=False)
    dqvy = calc_flux(Qgy.T, Qmonoy.T, Cy.T, dt /
                     dy[0], cosc, cose, mvae, mva, u=v.T).T
    return Q0 + (dqvz + dqvy)/mva
