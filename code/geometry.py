from pysph.tools.geometry import (get_2d_block, get_2d_tank,
                                  get_3d_block)
import numpy as np


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def create_circle(diameter=1, spacing=0.05, center=None):
    radius = diameter/2.
    xtmp, ytmp = get_2d_block(spacing, diameter+spacing, diameter+spacing)
    x = []
    y = []
    for i in range(len(xtmp)):
        dist = xtmp[i]**2. + ytmp[i]**2.
        if dist < radius**2:
            x.append(xtmp[i])
            y.append(ytmp[i])

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def hydrostatic_tank_2d(fluid_length, fluid_height, tank_height, tank_layers,
                        fluid_spacing, tank_spacing):
    xt, yt = get_2d_tank(dx=tank_spacing,
                         length=fluid_length + 2. * tank_spacing,
                         height=tank_height,
                         num_layers=tank_layers)
    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length + fluid_spacing,
                          height=fluid_height,
                          center=[-1.5, 1])

    xf += (np.min(xt) - np.min(xf))
    yf -= (np.min(yf) - np.min(yt))

    # now adjust inside the tank
    xf += tank_spacing * (tank_layers)
    yf += tank_spacing * (tank_layers)

    return xf, yf, xt, yt


def get_fluid_tank_3d(fluid_length,
                      fluid_height,
                      fluid_depth,
                      tank_length,
                      tank_height,
                      tank_layers,
                      fluid_spacing,
                      tank_spacing,
                      hydrostatic=False):
    """
    length is in x-direction
    height is in y-direction
    depth is in z-direction
    """
    xf, yf, zf = get_3d_block(dx=fluid_spacing,
                              length=fluid_length,
                              height=fluid_height,
                              depth=fluid_depth)

    # create a tank layer on the left
    xt_left, yt_left, zt_left = get_3d_block(dx=fluid_spacing,
                                             length=tank_spacing *
                                             (tank_layers - 1),
                                             height=tank_height,
                                             depth=fluid_depth)

    xt_right, yt_right, zt_right = get_3d_block(dx=fluid_spacing,
                                                length=tank_spacing *
                                                (tank_layers - 1),
                                                height=tank_height,
                                                depth=fluid_depth)

    # adjust the left wall of tank
    xt_left += np.min(xf) - np.max(xt_left) - tank_spacing
    yt_left += np.min(yf) - np.min(yt_left) + 0. * tank_spacing

    # adjust the right wall of tank
    xt_right += np.max(xf) - np.min(xt_right) + tank_spacing
    if hydrostatic is False:
        xt_right += tank_length - fluid_length

    yt_right += np.min(yf) - np.min(yt_right) + 0. * tank_spacing

    # create the wall in the front
    xt_front, yt_front, zt_front = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_height,
        depth=tank_spacing * (tank_layers - 1))
    xt_front += np.min(xt_left) - np.min(xt_front)
    yt_front += np.min(yf) - np.min(yt_front) + 0. * tank_spacing
    zt_front += np.max(zt_left) - np.min(zt_front) + tank_spacing * 1

    # create the wall in the back
    xt_back, yt_back, zt_back = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_height,
        depth=tank_spacing * (tank_layers - 1))
    xt_back += np.min(xt_left) - np.min(xt_back)
    yt_back += np.min(yf) - np.min(yt_back) + 0. * tank_spacing
    zt_back += np.min(zt_left) - np.max(zt_back) - tank_spacing * 1

    # create the wall in the bottom
    xt_bottom, yt_bottom, zt_bottom = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_spacing * (tank_layers - 1),
        depth=np.max(zt_front) - np.min(zt_back))
    xt_bottom += np.min(xt_left) - np.min(xt_bottom)
    yt_bottom += np.min(yt_left) - np.max(yt_bottom) - tank_spacing * 1

    xt = np.concatenate([xt_left, xt_right, xt_front, xt_back, xt_bottom])
    yt = np.concatenate([yt_left, yt_right, yt_front, yt_back, yt_bottom])
    zt = np.concatenate([zt_left, zt_right, zt_front, zt_back, zt_bottom])
    return xf, yf, zf, xt, yt, zt


def create_tank_2d_from_block_2d(xf, yf, tank_length, tank_height,
                                 tank_spacing, tank_layers):
    """
    This is mainly used by granular flows

    Tank particles radius is spacing / 2.
    """
    ####################################
    # create the left wall of the tank #
    ####################################
    xleft, yleft = get_2d_block(dx=tank_spacing,
                                length=(tank_layers - 1) * tank_spacing,
                                height=tank_height,
                                center=[0., 0.])
    xleft += min(xf) - max(xleft) - tank_spacing
    yleft += min(yf) - min(yleft)

    xright = xleft + abs(min(xleft)) + tank_length + tank_spacing
    yright = yleft

    xbottom, ybottom = get_2d_block(dx=tank_spacing,
                                    length=max(xright) - min(xleft),
                                    height=(tank_layers - 1) * tank_spacing,
                                    center=[0., 0.])
    xbottom += min(xleft) - min(xbottom)
    ybottom += min(yleft) - max(ybottom) - tank_spacing

    x = np.concatenate([xleft, xright, xbottom])
    y = np.concatenate([yleft, yright, ybottom])

    return x, y
