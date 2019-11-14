#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import copy
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from math import cos, sin, radians
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle
# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


class open3d_operations:
    '''
    Class to do various operations with mesh, point cloud etc,
    imported using Open3d
    '''
    def __init__(self, debug_mode=False):
        '''
        Args:
            - debug_mode (bool): True for debug mode
        Returns:
            - None
        '''
        self.debug_mode=debug_mode
        self.trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
        # self.trans_init = np.asarray([[1,0,0, 2],
        #                      [0,1,0,1.0],
        #                      [0,0,1,0.4], [0.0, 0.0, 0.0, 1.0]])        
        self.counter = 14
    def load_mesh(self, filename):
        '''
        Loads the mesh using Open3D in either ply, stl, obj, 
        off or gltf format
        Args:
            - filename (string): Name of the file with above 
                extensions
        Returns:   
            - mesh in geometry::TriangleMesh
        '''
        return o3d.io.read_triangle_mesh(filename)

    def load_pointcloud(self, filename):
        '''
        Loads the point clouds using Open3D in either xyz, xyzn, xyzrgb, 
        pts, ply or pcd format
        Args:
            - filename (string): Name of the file with above 
                extensions
        Returns:   
            - mesh in geometry::PointClouds
        '''
        return o3d.io.read_point_cloud(filename)
    
    def view(self,data):
        '''
        Displays point cloud or mesh
        Args:
            - data (obj, open3d): the object containing point cloud
                or mesh data
        Returns:   
            - None
        '''
        o3d.visualization.draw_geometries(data)

    def make_pcd(self,np_data):
        '''
        Makes a point cloud from npdata
        Args:
            - np_data (numpy): data in format N x 3
        Returns:
            - pcd (obj, open3d) the object containing point cloud
        '''
        shape_data = np.shape(np_data)
        if shape_data[1] == 3:
            pass
        else:
            np_data = np_data.T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_data)
        return pcd

    def get_matrix(self, rotation, translation):
        def trig(angle):
            r = radians(angle)
            return cos(r), sin(r)

        xC, xS = trig(rotation[0])
        yC, yS = trig(rotation[1])
        zC, zS = trig(rotation[2])
        dX = translation[0]
        dY = translation[1]
        dZ = translation[2]
        Translate_matrix = np.array([[1, 0, 0, dX],
                                    [0, 1, 0, dY],
                                    [0, 0, 1, dZ],
                                    [0, 0, 0, 1]])
        Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                    [0, xC, -xS, 0],
                                    [0, xS, xC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                    [0, 1, 0, 0],
                                    [-yS, 0, yC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                    [zS, zC, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))

    def get_source_target(self,source, transformation=None):
        '''
        Given the source mesh or point cloud, transforms the same using
        transormation matrix.
        Args:
            - source (obj open3d): point cloud or mesh loaded using Open3D
            - transormation (nparray, 4 x 4): transofrmation matrix
        Returns:
            - original and transformed mesh or point cloud.
        '''
        target = source # Using the same image to test
        # If transformation matrix is not available, uses default
        if transformation is None: 
            transformation = self.trans_init
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        # Makes yellow for original and cyan for transformed
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        # applies transformation
        source_temp.transform(transformation)
        if self.debug_mode:
            o3d.visualization.draw_geometries([source_temp, target_temp])
        return source_temp, target_temp
    
    def get_vertices(self, mesh):
        return np.asarray(mesh.vertices)
    
    def save_image(self, source, target, values):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def load_render_option(vis):
            vis.get_render_option().load_from_json(
                "../../TestData/renderoption.json")
            return False

        def capture_depth(vis):
            depth = vis.capture_depth_float_buffer()
            plt.imshow(np.asarray(depth))
            plt.show()
            return False

        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            # plt.text(5,5,'capture')
            plt.imsave(str(self.counter)+'.jpg', np.asarray(image))
            img = Image.open(str(self.counter)+'.jpg')
            draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("arial.ttf", 32)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((50, 50),"EMD: "+str(values[0])+", CD: "+\
                str(values[1])+", HD: "+str(values[2]),(255,0,0),font=font)
            img.save(str(self.counter)+'.jpg')
            # plt.show()
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        key_to_callback[ord("R")] = load_render_option
        key_to_callback[ord(",")] = capture_depth
        key_to_callback[ord(".")] = capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([source, target], key_to_callback)
    
    def get_matrix(self, rotation, translation):
        def trig(angle):
            r = radians(angle)
            return cos(r), sin(r)
        xC, xS = trig(rotation[0])
        yC, yS = trig(rotation[1])
        zC, zS = trig(rotation[2])
        dX = translation[0]
        dY = translation[1]
        dZ = translation[2]
        Translate_matrix = np.array([[1, 0, 0, dX],
                                    [0, 1, 0, dY],
                                    [0, 0, 1, dZ],
                                    [0, 0, 0, 1]])
        Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                    [0, xC, -xS, 0],
                                    [0, xS, xC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                    [0, 1, 0, 0],
                                    [-yS, 0, yC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                    [zS, zC, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        return

SCALE_FACTOR = 0.05
MODE = 'sphere'
DRAW_LINE = True   
def visualize_scene(pc1, pc2, sf, output):

	if pc1.shape[1] != 3:
		pc1 = pc1.T
		pc2 = pc2.T
		sf = sf.T
		output = output.T
	
	gt = pc1 + sf
	pred = pc1 + output
	
	print('pc1, pc2, gt, pred', pc1.shape, pc2.shape, gt.shape, pred.shape)


	fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
	
	if True: #len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
		mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
	
	if False:
		mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0,1,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # cyan

	mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # red
	mlab.points3d(pred[:, 0], pred[:,1], pred[:,2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # green
	
	
	# DRAW LINE
	if True:
		N = 2
		x = list()
		y = list()
		z = list()
		connections = list()

		inner_index = 0
		for i in range(gt.shape[0]):
			x.append(gt[i, 0])
			x.append(pred[i, 0])
			y.append(gt[i, 1])
			y.append(pred[i, 1])
			z.append(gt[i, 2])
			z.append(pred[i, 2])

			connections.append(np.vstack(
				[np.arange(inner_index,   inner_index + N - 1.5),
				np.arange(inner_index + 1,inner_index + N - 0.5)]
			).T)
			inner_index += N

		x = np.hstack(x)
		y = np.hstack(y)
		z = np.hstack(z)

		connections = np.vstack(connections)

		src = mlab.pipeline.scalar_scatter(x, y, z)

		src.mlab_source.dataset.lines = connections
		src.update()
		
		lines= mlab.pipeline.tube(src, tube_radius=0.005, tube_sides=6)
		mlab.pipeline.surface(lines, line_width=2, opacity=.4, color=(1,1,0))
	# DRAW LINE END

	
	mlab.view(90, # azimuth
	         150, # elevation
			 50, # distance
			 [0, -1.4, 18], # focalpoint
			 roll=0)

	mlab.orientation_axes()

	mlab.show()