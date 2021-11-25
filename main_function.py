import os
import sys
import ast
import cv2
import time
import torch
import numpy as np
import io
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(1, os.getcwd()+'/simple-HRNet/models/detectors/yolo')
sys.path.insert(1, os.getcwd()+'/simple-HRNet')
sys.path.insert(1, os.getcwd()+'/VideoPose3D')

import matplotlib.pyplot as plt
from common.arguments import parse_args
from common.camera import *
from common.model import *
from common.jottue_dataset import CustomDataset
# from common.custom_dataset import CustomDataset

import torch
import os
import time
import glob
import cv2

ROOT_dir = os.path.dirname(os.path.abspath(__file__))

def get_img_from_fig(fig, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


class d3_regressor():
    def __init__(self, MODEL_NAME='hr_pt_coco', TRAJ=True,  device=torch.device('cuda')):
        self.MODEL_NAME = MODEL_NAME
        self.dataset = CustomDataset()
        self.TRAJ = TRAJ
        self.cam = self.dataset.cameras()
        self.architecture = '3,3,3,3,3'
        self.causal = True

        self.filter_widths = [int(x) for x in self.architecture.split(',')]

        print('Loading 2D detections...')
        mm = np.load('VideoPose3D/data/metadata.npy', allow_pickle=True)
        self.keypoints_metadata = mm.item()


        self.model_pos = TemporalModel(17, 2, 
                                self.dataset.skeleton().num_joints(),
                                filter_widths= self.filter_widths, 
                                causal= self.causal, 
                                dropout=0.25, 
                                channels=1024,
                                dense=False)

        if self.TRAJ:
            self.model_traj = TemporalModel(17,2, 1,
                                    filter_widths=self.filter_widths, 
                                    causal=self.causal, 
                                    dropout=0.25, 
                                    channels=1024,
                                    dense=False)
        else:
            self.model_traj = None

        
        self.load_model()


        self.model_pos.to(device)
        if self.TRAJ:
            self.model_traj.to(device)




    def load_model(self):
        root = ROOT_dir + '/VideoPose3D/checkpoint/{}'.format(self.MODEL_NAME)


        chk_filename = sorted(glob.glob(os.path.join(root,'epoch_*.bin')))[-1]
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))

        self.model_pos.load_state_dict(checkpoint['model_pos'])
        if self.TRAJ:
            self.model_traj.load_state_dict(checkpoint['model_traj'])


        receptive_field = self.model_pos.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))

        if self.causal:
            print('INFO: Using causal convolutions')



    def predict(self, keypoints):
        keypoint_ = normalize_screen_coordinates(keypoints, w=self.cam['res_w'], h=self.cam['res_h'])

        with torch.no_grad():
            self.model_pos.eval()
            if self.TRAJ:
                self.model_traj.eval()

            inputs_2d = torch.from_numpy(np.array(keypoint_).astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = self.model_pos(inputs_2d[:, :244, :, :])

            if self.TRAJ:
                predicted_3d_traj = self.model_traj(inputs_2d[:, :244, :, :])
            else:
                predicted_3d_traj = 0

            predicted = predicted_3d_pos + predicted_3d_traj

            prediction = predicted.squeeze(1).cpu().numpy()




            rot = self.dataset.cameras()['orientation']
            prediction = camera_to_world(prediction, R=rot, t=0)

        # We don't have the trajectory, but at least we can rebase the height
        # if traj:
        #     prediction[:,1:] += prediction[:,:1]
            # prediction[:, :, 2] -= np.min(prediction[:, :, 2])




        return prediction




class visualization():
    def __init__(self, dataset, keypoints_metadata,):
        plt.ion()
        self.skeleton = dataset.skeleton()
        self.cam = dataset.cameras()
        self.keypoints_metadata = keypoints_metadata
        self.prev_ids = ['']

        self.radius = 1.7
        self.world_radius = 2.5
        self.initialized = False

        self.color_list = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 0)]




    def init_canvas(self, poses):
        size =6
        # fig = plt.figure(1,figsize=(size * (2 + len(poses)),size * 1))
        fig = plt.figure(1,figsize=(1, 2.1), dpi=2021)
        fig.tight_layout(pad=0)
        fig.clf()
        # ax_in = fig.add_subplot(1, 1, 1)

        # ax_in.get_xaxis().set_visible(False)
        # ax_in.get_yaxis().set_visible(False)
        # ax_in.set_axis_off()
        # ax_in.set_title('Input')
        

        # ax_world = fig.add_subplot(1, 2 + len(poses), 2 + len(poses), projection='3d')
        ax_world = fig.add_subplot( projection='3d')
        ax_world.view_init(elev=1., azim=self.cam['azimuth'])
        ax_world.set_xlim3d([-self.world_radius/2, self.world_radius/2])
        ax_world.set_zlim3d([0, self.world_radius])
        ax_world.set_ylim3d([-self.world_radius/2, self.world_radius/2])
        ax_world.set_box_aspect(aspect = (1,1,2))
        


        # ax_world.get_proj = lambda: np.dot(Axes3D.get_proj(ax_world), np.diag([0.3, 0.3, 1.3, 1]))
                      
        try:
            ax_world.set_aspect('equal')
        except NotImplementedError:
            ax_world.set_aspect('auto')
        ax_world.set_xticklabels([])
        ax_world.set_yticklabels([])
        ax_world.set_zticklabels([])
        # ax_world.set_zticks([0, self.world_radius])
        ax_world.dist = 7.5
        ax_world.set_title('world') #, pad=35



        ax_3d = []
        lines_3d = []
        lines_3d_world = []
        # trajectories = []

        for index, (title, data) in enumerate(poses.items()):
        #     ax = fig.add_subplot(1, 2 + len(poses), index+2, projection='3d')
        #     ax.view_init(elev=15., azim=self.cam['azimuth'])
        #     ax.set_xlim3d([-self.radius/2, self.radius/2])
        #     ax.set_zlim3d([0,self.radius])
        #     ax.set_ylim3d([-self.radius/2, self.radius/2])
        #     try:
        #         ax.set_aspect('equal')
        #     except NotImplementedError:
        #         ax.set_aspect('auto')
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_zticklabels([])
        #     ax.dist = 7.5
        #     ax.set_title(title) #, pad=35
            # ax_3d.append(ax)
            ax_3d.append([])
            lines_3d.append([])
            lines_3d_world.append([])


        self.fig = fig
        # self.ax_in = ax_in
        self.ax_3d = ax_3d
        self.ax_world = ax_world
        self.lines_3d = lines_3d
        self.lines_3d_world = lines_3d_world
        self.initialized = False
        self.prev_ids = sorted(poses.keys())
        self.points = []




    def update_video(self, single_frame, keypoints_, predictions, bbox,person_ids):
        """
        single_frmae = [H, W]
        keypoints = [id, 17, 2]
        predictions = [id, 17 ,3] x,y,z array

        """

        temp = np.copy(single_frame)



        if not len(predictions) == 0:

            for identity in range(len(predictions)):
                cv2.rectangle(temp, tuple(bbox[identity, :2]), tuple(bbox[identity, 2:]), self.color_list[(person_ids[identity])%(len(self.color_list))], 3)
                cv2.putText(temp, str(person_ids[identity]), tuple(bbox[identity, :2] + [10,30]), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_list[(person_ids[identity])%(len(self.color_list))], 2)


            trajectories = predictions[:,0,[0,1]]
            min_all_z = np.min(predictions[:,:,2])
            keypoints = keypoints_[..., 1::-1]


            # for n, ax in enumerate(self.ax_3d):
            #     ax.set_xlim3d([-self.radius/2 + trajectories[n][0], self.radius/2 + trajectories[n][0]])
            #     ax.set_ylim3d([-self.radius/2 + trajectories[n][1], self.radius/2 + trajectories[n][1]])

            joints_right_2d = self.keypoints_metadata['keypoints_symmetry'][1]
            colors_2d = np.full(17, 'black')
            colors_2d[joints_right_2d] = 'red'

            if not self.initialized:
                # print(temp[...,::-1].shape)
                # self.image = self.ax_in.imshow(temp[...,::-1], aspect='equal')

                for j, j_parent in enumerate(self.skeleton.parents()):
                    if j_parent == -1:
                        continue

                    col = 'red' if j in self.skeleton.joints_right() else 'black'
                    for n, ax in enumerate(self.ax_3d):
                        pos = predictions[n]
                        # pos[:,2] -= min_all_z
                        pos[:, 2] -= np.min(pos[:, 2])  
                        self.lines_3d_world[n].append(self.ax_world.plot( [pos[j, 0], pos[j_parent, 0]],
                                                    [pos[j, 1], pos[j_parent, 1]],
                                                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                        # self.lines_3d_world[n].append(self.ax_in.plot( [pos[j, 0], pos[j_parent, 0]],
                        #                             [pos[j, 1], pos[j_parent, 1]], zorder=10, c=col))
                        # pos = predictions[n]
                        # pos[:, 2] -= np.min(pos[:, 2])    
                        # self.lines_3d[n].append(ax.plot(
                        #                             [pos[j, 0], pos[j_parent, 0]],
                        #                             [pos[j, 1], pos[j_parent, 1]],
                        #                             [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))



                # for idx in range(len(keypoints)):
                #     self.points.append(self.ax_in.scatter(*keypoints[idx].T, 10, color=colors_2d, edgecolors='white', zorder=10))



                self.initialized = True

            else:
                # self.image.set_data(temp[...,::-1])

                for j, j_parent in enumerate(self.skeleton.parents()):
                    if j_parent == -1:
                        continue
                    

                    for n, ax in enumerate(self.ax_3d):
                        pos = predictions[n]
                        # pos[:,2] -= min_all_z
                        pos[:, 2] -= np.min(pos[:, 2])  
                        self.lines_3d_world[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        self.lines_3d_world[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        self.lines_3d_world[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')


                        # pos = predictions[n]
                        # pos[:, 2] -= np.min(pos[:, 2])  
                        # self.lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        # self.lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        # self.lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')




                # for idx in range(len(keypoints)):
                #     self.points[idx].set_offsets(keypoints[idx])            


        else:
            if not self.initialized:
                self.image = self.ax_in.imshow(temp[...,::-1], aspect='equal')

            else:
                self.image.set_data(temp[...,::-1])



if __name__ == "__main__":
    a = d3_regressor()
    # keypoint = np.zeros((3,243,17,2))
    # for i in range(10):
    #     e=a.predict(keypoint)
    #     print(e.shape)



    input_frame = np.zeros((241,690))
    predicted_3d_joint = np.array([[[-1.5957887 , -3.176806  ,  0.76701665],
            [-1.5637767 , -3.2527926 ,  0.7723584 ],
            [-1.5208877 , -3.2733572 ,  0.42130136],
            [-1.6535568 , -3.332038  ,  0.02567077],
            [-1.6277542 , -3.100742  ,  0.7616198 ],
            [-1.5647541 , -3.1462731 ,  0.40399218],
            [-1.661511  , -3.2281253 ,  0.        ],
            [-1.5561674 , -3.164929  ,  0.988209  ],
            [-1.4707533 , -3.160179  ,  1.2273252 ],
            [-1.4071922 , -3.1388144 ,  1.2869654 ],
            [-1.4087183 , -3.159846  ,  1.3799853 ],
            [-1.5323102 , -3.0686026 ,  1.1749096 ],
            [-1.6204069 , -2.987779  ,  0.9267025 ],
            [-1.5581608 , -2.9888146 ,  0.73592687],
            [-1.44955   , -3.2575305 ,  1.1611171 ],
            [-1.4809456 , -3.352083  ,  0.93147755],
            [-1.4340894 , -3.2984328 ,  0.7676554 ]]], dtype=np.float32)



    predicted_3d_joint_ = predicted_3d_joint
    predicted_3d_joint_[...,1:] += 1

    keypoints = np.ones((2,17,2)) * 300
    poses = {0:predicted_3d_joint, 1: predicted_3d_joint_}
    azim = 70
    prediction = np.concatenate([predicted_3d_joint[None,...],predicted_3d_joint[None,...]],axis=0)
    ###################
    

    vis = visualization(a.dataset, a.keypoints_metadata)
    vis.init_canvas(poses)

    
    vis.update_video(input_frame, keypoints, prediction[:,0,...] )
    vis.fig.canvas.flush_events()
    cv2.waitKey(0)

    
