import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform
import threading
import time
import numpy as np
import smplx
import torch
import math
from enum import Enum, auto
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial

isMacOS = (platform.system() == "Darwin")

class AnimationState():
    PAUSED = "pause"
    PLAYING = "play"
    REVERSE = "reverse"

class MeshState(Enum):
    Trasparent = 0
    Opaque = 1
    Hidden = 2

class ArrowType(Enum):
    Velocity = auto()
    Acceleration = auto()

class Arrow:
    def __init__(self, arrow_type, string_id, is_checked):
        self.arrow_type = arrow_type
        self.string_id = string_id
        self.is_checked = is_checked
        self.material = o3d.visualization.rendering.MaterialRecord()
        
class Pose:
    def __init__(self, joints, vertices):
        self.joints = joints
        self.vertices = vertices

class Joint:
    def __init__(self, name, index, parent_index, child_index, referecne_angle, color):
        self.name = name
        self.index = index
        self.parent_index = parent_index
        self.child_index = child_index
        self.referecne_angle = referecne_angle
        self.color = color
        self.is_checked = True

joints = [
    Joint("Pelvis", 0, None, None, None, None),
    Joint("Left Hip", 1, 0, 4, 26, "#FFD740"),
    Joint("Right Hip", 2, 0, 5, 25, "#BA68C8"),
    Joint("Spine1", 3, 0, 6, 13, "#FF5252"),
    Joint("Left Knee", 4, 1, 7, 15, "#FFFF00"),
    Joint("Right Knee", 5, 2, 8, 9, "#B388FF"),
    Joint("Spine2", 6, 3, 9, 34, "#FF8A80"),
    Joint("Left Ankle", 7, 4, 10, 72, "#FFF176"),
    Joint("Right Ankle", 8, 5, 11, 68, "#E1BEE7"),
    Joint("Spine3", 9, 6, 12, 40, "#FF80AB"),
    Joint("Left Foot", 10, 7, None, None, None),
    Joint("Right Foot", 11, 8, None, None, None),
    Joint("Neck", 12, 9, 15, 22, "#F8BBD0"),
    Joint("Left Collar", 13, 9, 16, 35, "#2979FF"),
    Joint("Right Collar", 14, 9, 17, 33, "#00BFA5"),
    Joint("Head", 15, 12, None, None, None),
    Joint("Left Shoulder", 16, 13, 18, 41, "#00B0FF"),
    Joint("Right Shoulder", 17, 14, 19, 35, "#00E676"),
    Joint("Left Elbow", 18, 16, 20, 23, "#00E5FF"),
    Joint("Right Elbow", 19, 17, 21, 7, "#76FF03"),
    Joint("Left Wrist", 20, 18, 22, 41, "#90CAF9"),
    Joint("Right Wrist", 21, 19, 23, 43, "#B2FF59"),
    Joint("Left Hand", 22, 20, None, None, None),
    Joint("Right Hand", 23, 21, None, None, None)
]

class UserArrowSetting:
    def __init__(self):
        self.is_enabled = False
        self.is_scailing_enabled = False
        self.arrow_size = 1
        self.smoothing_size = 1
        self.color = []

class JointAngleManager:
    @staticmethod
    def calcuate_vector_pointed_from_parent(currrent_position, parent_position):
        vector = currrent_position - parent_position
        norm = np.linalg.norm(vector)
        if norm != 0:
            vector = vector / norm
        return vector
    
    @staticmethod
    def find_joint_angle(vector_pointed_from_parent, vector_pointing_child):
            dot_product = np.dot(vector_pointed_from_parent, vector_pointing_child)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            return np.degrees(angle_rad)
    
class JointAngleData:
    def __init__(self, num_frames):
        self.angles = np.zeros((22, num_frames))
        self.velocities = np.zeros((22, num_frames))
        self.accelertions = np.zeros((22, num_frames))
        self.isEmpty = True

class AMASS_Motion:
    _counter = 0

    def __init__(self, name, max_vel, max_acc, path):
        self.index = AMASS_Motion._counter
        AMASS_Motion._counter += 1
        self.name = name
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.path = path
        self.dataset = np.load(self.path)
        self.num_frames = self.dataset['root_orient'].shape[0]
        self.frame_rate = self.dataset['mocap_frame_rate']


AMASS_Motions = [
    AMASS_Motion('Dance', 5.34, 489.42, './dataset/05_03_stageii.npz'),
    AMASS_Motion('Walking', 5.7, 222.7, './dataset/02_02_stageii.npz'),
    AMASS_Motion('Running', 6.5, 165.407, './dataset/02_03_stageii.npz'),
    AMASS_Motion('Star jump', 4, 144.7, './dataset/Subject_2_F_2_stageii.npz'),
    AMASS_Motion('Basketball - walking dribble', 6.3, 947, './dataset/06_03_stageii.npz'),
    AMASS_Motion('Basketball - side hop dribble', 5.2, 374.4, './dataset/06_09_stageii.npz'), 
    AMASS_Motion('Basketball - jump shoot', 4.3, 365.7, './dataset/06_14_stageii.npz'),
    AMASS_Motion('Basketball - jump shoot 2',  5.7, 619.2, './dataset/06_15_stageii.npz'),
    AMASS_Motion('Football', 11.7, 514.6, './dataset/10_05_stageii.npz'),
    AMASS_Motion('Chicken Wings', 4.8, 295.7, './dataset/50002_chicken_wings_stageii.npz')
]

class VisualizationApp:

    def __init__(self):
        self.selected_amass_data = None
        self.motion_data = AMASS_Motions[0]
        self.animated_state = AnimationState.PAUSED
        self.mesh_state = MeshState.Trasparent
        self.current_frame = 0
        self.joint_size = 1
        self.total_time_point_cloud = 0
        self.total_time_mesh = 0
        self.frame_count = 0

        # self.vel_max = 0
        # self.acc_max = 0

        # --- Window & Scene ---
        self.window = gui.Application.instance.create_window(
            "Human Motion Visualization", 1024, 768)
        w = self.window
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.scene.scene.scene.set_sun_light(
            [0.577, -0.577, -0.577],  # direction
            [1, 1, 1],  # color
            45000)  # intensity
        # self.scene.scene.scene.enable_sun_light(True)
        self.scene.scene.scene.enable_sun_light(False)
        self.scene.scene.show_skybox(True)
        # self.scene.scene.set_background([0, 0, 0, 1])

        bbox = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2],
                                                   [2, 2, 2])
        self.scene.setup_camera(45, bbox, [1,0,0]) #field_of_view, center(pointing toward center), rotation
        self.scene.look_at([0,0,0], [-2, -5, 1], [0,0,1]) #look_at(center, eye, up)
        
        # ---- Settings Panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        indentation_width = 1.5 * em

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        # self._settings_panel.preferred_width = int(560 * self.window.scaling)
        view_ctrls = gui.CollapsableVert("Motion Datasets & Body Mesh", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        
        self._combobox_amass_data = gui.Combobox()
        for motion in AMASS_Motions:
            self._combobox_amass_data.add_item(motion.name)
        self._combobox_amass_data.set_on_selection_changed(self._on_combobox_amass_data)
        view_ctrls.add_child(gui.Label("AMASS Motion Dataset"))
        view_ctrls.add_child(self._combobox_amass_data)

        view_ctrls.add_child(gui.Label("Play Controls"))
        self.play_slider = gui.Slider(gui.Slider.INT)
        self.play_slider.set_limits(1, self.motion_data.num_frames - 1)
        self.play_slider.set_on_value_changed(self._on_play_slider)
        view_ctrls.add_child(self.play_slider)
        
        self._reset_button = gui.Button("<<")
        self._reset_button.horizontal_padding_em = 0.5
        self._reset_button.vertical_padding_em = 0
        self._reset_button.set_on_clicked(self._on_button_reset)
        self._play_button = gui.Button(AnimationState.PLAYING)
        self._play_button.horizontal_padding_em = 0.5
        self._play_button.vertical_padding_em = 0
        self._play_button.set_on_clicked(self._on_button_play)
        self._pause_button = gui.Button(AnimationState.PAUSED)
        self._pause_button.horizontal_padding_em = 0.5
        self._pause_button.vertical_padding_em = 0
        self._pause_button.set_on_clicked(self._on_button_stop)
        self._reverse_button = gui.Button(AnimationState.REVERSE)
        self._reverse_button.horizontal_padding_em = 0.5
        self._reverse_button.vertical_padding_em = 0
        self._reverse_button.set_on_clicked(self._on_button_reverse)
        h = gui.Horiz(0.25 * em) 
        h.add_stretch()
        h.add_child(self._reset_button)
        h.add_child(self._play_button)
        h.add_child(self._pause_button)
        h.add_child(self._reverse_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Mesh"))
        radio_button = gui.RadioButton(gui.RadioButton.VERT)
        radio_button.set_items(["Transparent Mesh", "Opaque Mesh", "Hide Mesh"])
        def on_radio_button_changed(idx):
            self.mesh_state = MeshState(idx)
            if (self.mesh_state == MeshState.Hidden):
                self.scene.scene.remove_geometry('mesh')
                self.scene.scene.remove_geometry('joints')
            else:
                self.draw_mesh(self.current_frame)

        radio_button.set_on_selection_changed(on_radio_button_changed)
        view_ctrls.add_child(radio_button)

        self.joint_size_slider = gui.Slider(gui.Slider.INT)
        self.joint_size_slider.set_limits(1, 5)
        self.joint_size_slider.set_on_value_changed(self._on_joint_size_slide)
        grid = gui.VGrid(2, 6 * em)
        grid.add_child(gui.Label("Joint size"))
        grid.add_child(self.joint_size_slider)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(grid)
        view_ctrls.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        vis_ctrls = gui.CollapsableVert("Velocity & Acceleration", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        vis_ctrls.add_child(gui.Label("Velocity and Acceleration"))

        self._velocity_checkbox = gui.Checkbox("Velocity")
        self._velocity_checkbox.set_on_checked(self._on_velocity_checkbox)
        vis_ctrls.add_child(self._velocity_checkbox)

        self.vel_arrow_setting = UserArrowSetting()

        def on_vel_color(new_color):
            color = [
            new_color.red, new_color.green,
            new_color.blue, new_color.alpha
            ]
            self.arrow_velocity.material.base_color = color
        
        self.vel_color = gui.ColorEdit()
        self.vel_color.set_on_value_changed(on_vel_color)
        self.vel_color.color_value = gui.Color(1.0, 1.0, 0.0, 1.0)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Arrow Color"))
        grid.add_child(self.vel_color)
        indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_item.add_child(grid)
        vis_ctrls.add_child(indented_item)

        self.vel_size_fix_checkbox = gui.Checkbox("Reflect vector magnitude")
        def on_vel_size_fix_checkbox(is_checked):
            self.vel_arrow_setting.is_scailing_enabled = is_checked
        self.vel_size_fix_checkbox.set_on_checked(on_vel_size_fix_checkbox)
        indented_vel1 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel1.add_child(self.vel_size_fix_checkbox)
        vis_ctrls.add_child(indented_vel1)

        self.vel_scailing_slider = gui.Slider(gui.Slider.INT)
        self.vel_scailing_slider.set_limits(1, 5)
        def on_vel_scailing_slider(size):
            self.vel_arrow_setting.arrow_size = int(size)
        self.vel_scailing_slider.set_on_value_changed(on_vel_scailing_slider)
        grid = gui.VGrid(2, 6 * em)
        grid.add_child(gui.Label("Arrow size"))
        grid.add_child(self.vel_scailing_slider)
        indented_vel2 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel2.add_child(grid)
        vis_ctrls.add_child(indented_vel2)

        self.vel_smoothing_slider = gui.Slider(gui.Slider.INT)
        self.vel_smoothing_slider.set_limits(1, 10)
        def on_vel_smoothing_slider(size):
            self.vel_arrow_setting.smoothing_size = int(size)
        self.vel_smoothing_slider.set_on_value_changed(on_vel_smoothing_slider)
        grid = gui.VGrid(2, 0.9 * em)
        grid.add_child(gui.Label("Smoothing window size"))
        grid.add_child(self.vel_smoothing_slider)
        indented_vel3 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel3.add_child(grid)
        vis_ctrls.add_child(indented_vel3)

        self._acceleration_checkbox = gui.Checkbox("Acceleration")
        self._acceleration_checkbox.set_on_checked(self._on_acceleration_checkbox)
        vis_ctrls.add_child(self._acceleration_checkbox)

        self.acc_arrow_setting = UserArrowSetting()

        def on_acc_color(new_color):
            color = [
            new_color.red, new_color.green,
            new_color.blue, new_color.alpha
            ]
            self.arrow_acceleration.material.base_color = color
        
        self.acc_color = gui.ColorEdit()
        self.acc_color.set_on_value_changed(on_acc_color)
        self.acc_color.color_value = gui.Color(1.0, 0.0, 0.0, 1.0)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Arrow Color"))
        grid.add_child(self.acc_color)
        indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_item.add_child(grid)
        vis_ctrls.add_child(indented_item)

        self.acc_size_fix_checkbox = gui.Checkbox("Reflect vector magnitude")
        def on_acc_size_fix_checkbox(is_checked):
            self.acc_arrow_setting.is_scailing_enabled = is_checked
        self.acc_size_fix_checkbox.set_on_checked(on_acc_size_fix_checkbox)
        indented_vel1 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel1.add_child(self.acc_size_fix_checkbox)
        vis_ctrls.add_child(indented_vel1)

        self.acc_scailing_slider = gui.Slider(gui.Slider.INT)
        self.acc_scailing_slider.set_limits(1, 5)
        def on_acc_scailing_slider(size):
            self.acc_arrow_setting.arrow_size = int(size)
        self.acc_scailing_slider.set_on_value_changed(on_acc_scailing_slider)
        grid = gui.VGrid(2, 6 * em)
        grid.add_child(gui.Label("Arrow size"))
        grid.add_child(self.acc_scailing_slider)
        indented_vel2 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel2.add_child(grid)
        vis_ctrls.add_child(indented_vel2)

        self.acc_smoothing_slider = gui.Slider(gui.Slider.INT)
        self.acc_smoothing_slider.set_limits(1, 10)
        def on_acc_smoothing_slider(size):
            self.acc_arrow_setting.smoothing_size = int(size)
        self.acc_smoothing_slider.set_on_value_changed(on_acc_smoothing_slider)
        grid = gui.VGrid(2, 0.9 * em)
        grid.add_child(gui.Label("Smoothing window size"))
        grid.add_child(self.acc_smoothing_slider)
        indented_vel3 = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
        indented_vel3.add_child(grid)
        vis_ctrls.add_child(indented_vel3)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(vis_ctrls)

        joint_ctrls = gui.CollapsableVert("Joint Angle Plot", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        torso_indices = [12, 9, 6, 3]
        right_arm_indices = [14, 17, 19, 21]
        left_arm_indicies = [13, 16, 18, 20]
        right_leg_indices = [2, 5, 8]
        left_leg_indices = [1, 4, 7]
        def on_checkbox_checked(joint_indices, is_checked):
            if (len(joint_indices) > 1): # handle checkbox of super categories
                for i in joint_indices:
                    joints[i].is_checked = is_checked
                    self._joint_checkboxes[i].checked = is_checked
            else: # handle checkbox of each joint
                joints[joint_indices[0]].is_checked = is_checked
                if joint_indices[0] in torso_indices: 
                    selected_joints = [joints[i] for i in torso_indices]
                    self._checkbox_torso.checked = all(joint.is_checked for joint in selected_joints)
                elif joint_indices[0] in right_arm_indices:
                    selected_joints = [joints[i] for i in right_arm_indices]
                    self._checkbox_right_arm.checked = all(joint.is_checked for joint in selected_joints)
                elif joint_indices[0] in left_arm_indicies: 
                    selected_joints = [joints[i] for i in left_arm_indicies]
                    self._checkbox_left_arm.checked = all(joint.is_checked for joint in selected_joints)
                elif joint_indices[0] in right_leg_indices:
                    selected_joints = [joints[i] for i in right_leg_indices]
                    self._checkbox_right_leg.checked = all(joint.is_checked for joint in selected_joints)
                elif joint_indices[0] in left_leg_indices:
                    selected_joints = [joints[i] for i in left_leg_indices]
                    self._checkbox_left_leg.checked = all(joint.is_checked for joint in selected_joints)
            all_indicies_of_checkboxes = torso_indices + right_arm_indices + left_arm_indicies + right_leg_indices + left_leg_indices
            for i in all_indicies_of_checkboxes:
                if (not self._joint_checkboxes[i].checked):
                    self._checkbox_select_all.checked = False
                    break
                self._checkbox_select_all.checked = True

        def _on_checkbox_all_selected(is_checked):
            self._checkbox_torso.checked = is_checked
            self._checkbox_left_arm.checked = is_checked
            self._checkbox_right_arm.checked = is_checked
            self._checkbox_left_leg.checked = is_checked
            self._checkbox_right_leg.checked = is_checked
            all_indicies_of_checkboxes = torso_indices + right_arm_indices + left_arm_indicies + right_leg_indices + left_leg_indices
            for i in all_indicies_of_checkboxes:
                self._joint_checkboxes[i].checked = is_checked
                joints[i].is_checked = is_checked


        self._joint_checkboxes = [gui.Checkbox(f"Joint {i+1}") for i in range(22)]
        
        self._checkbox_select_all = gui.Checkbox("Select All")
        self._checkbox_select_all.set_on_checked(_on_checkbox_all_selected)
        self._checkbox_select_all.checked = True

        self._plot_button = gui.Button("Draw Plot")
        self._plot_button.horizontal_padding_em = 1
        self._plot_button.vertical_padding_em = 0
        self._plot_button.set_on_clicked(self._on_plot_button)

        grid = gui.VGrid(2, 2 * em)
        grid.add_child(self._checkbox_select_all)
        grid.add_child(self._plot_button)
        joint_ctrls.add_child(grid)

        self._checkbox_torso = gui.Checkbox("Torso")
        self._checkbox_torso.set_on_checked(partial(on_checkbox_checked, torso_indices))
        joint_ctrls.add_child(self._checkbox_torso)
        self._checkbox_torso.checked = True

        for i in torso_indices:
            self._joint_checkboxes[i] = gui.Checkbox(joints[i].name)
            self._joint_checkboxes[i].set_on_checked(partial(on_checkbox_checked, [i]))
            indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
            indented_item.add_child(self._joint_checkboxes[i])
            joint_ctrls.add_child(indented_item)
            self._joint_checkboxes[i].checked = True

        self._checkbox_right_arm = gui.Checkbox("Right Arm")
        self._checkbox_right_arm.set_on_checked(partial(on_checkbox_checked, right_arm_indices))
        joint_ctrls.add_child(self._checkbox_right_arm)
        self._checkbox_right_arm.checked = True

        for i in right_arm_indices:
            self._joint_checkboxes[i] = gui.Checkbox(joints[i].name)
            self._joint_checkboxes[i].set_on_checked(partial(on_checkbox_checked, [i]))
            indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
            indented_item.add_child(self._joint_checkboxes[i])
            joint_ctrls.add_child(indented_item)
            self._joint_checkboxes[i].checked = True

        self._checkbox_left_arm = gui.Checkbox("Left Arm")
        self._checkbox_left_arm.set_on_checked(partial(on_checkbox_checked, left_arm_indicies))
        joint_ctrls.add_child(self._checkbox_left_arm)
        self._checkbox_left_arm.checked = True
        
        for i in left_arm_indicies:
            self._joint_checkboxes[i] = gui.Checkbox(joints[i].name)
            self._joint_checkboxes[i].set_on_checked(partial(on_checkbox_checked, [i]))
            indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
            indented_item.add_child(self._joint_checkboxes[i])
            joint_ctrls.add_child(indented_item)
            self._joint_checkboxes[i].checked = True

        self._checkbox_right_leg = gui.Checkbox("Right Leg")
        self._checkbox_right_leg.set_on_checked(partial(on_checkbox_checked, right_leg_indices))
        joint_ctrls.add_child(self._checkbox_right_leg)
        self._checkbox_right_leg.checked = True

        for i in right_leg_indices:
            self._joint_checkboxes[i] = gui.Checkbox(joints[i].name)
            self._joint_checkboxes[i].set_on_checked(partial(on_checkbox_checked, [i]))
            indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
            indented_item.add_child(self._joint_checkboxes[i])
            joint_ctrls.add_child(indented_item)
            self._joint_checkboxes[i].checked = True

        self._checkbox_left_leg = gui.Checkbox("Left Leg")
        self._checkbox_left_leg.set_on_checked(partial(on_checkbox_checked, left_leg_indices))
        joint_ctrls.add_child(self._checkbox_left_leg)
        self._checkbox_left_leg.checked = True

        for i in left_leg_indices:
            self._joint_checkboxes[i] = gui.Checkbox(joints[i].name)
            self._joint_checkboxes[i].set_on_checked(partial(on_checkbox_checked, [i]))
            indented_item = gui.Horiz(0, gui.Margins(indentation_width, 0, 0, 0))
            indented_item.add_child(self._joint_checkboxes[i])
            joint_ctrls.add_child(indented_item)
            self._joint_checkboxes[i].checked = True

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(joint_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self.scene)
        w.add_child(self._settings_panel)

        # ---- Open3D ----
        # SMPLX
        self.initialize_open3d()

        # initialize AMASS dataset
        self.set_motion_data(0)

    def initialize_open3d(self):
        self.model_folder = './dataset/models_lockedhead/'
        self.model = smplx.create(self.model_folder, model_type='smplx',
                                gender='neutral', use_face_contour=False,
                                num_betas=0,
                                num_expression_coeffs=0,
                                ext='npz')
        # Materials
        self.mat_mesh = o3d.visualization.rendering.MaterialRecord()
        self.mat_mesh.shader = "defaultLitTransparency"
        self.mat_mesh.base_roughness = 0.0
        self.mat_mesh.base_reflectance = 0.0
        self.mat_mesh.base_clearcoat = 1.0
        self.mat_mesh.thickness = 1.0
        self.mat_mesh.transmission = 1.0
        self.mat_mesh.absorption_distance = 10
        self.mat_mesh.absorption_color = [0.5, 0.5, 0.5]
        self.mat_mesh.base_color = [0.467, 0.467, 0.467, 0.2]
        self.mat_mesh.base_color = [0.3, 0.3, 0.3, 0.5]

        self.mat_dot = o3d.visualization.rendering.MaterialRecord()
        self.mat_dot.shader = 'defaultLit'
        self.mat_dot.base_color = [0.8, 0, 0, 1.0]
        self.mat_dot.point_size = 5

        self.arrow_velocity = Arrow(ArrowType.Velocity, "arrow_vel", False)
        self.arrow_acceleration = Arrow(ArrowType.Acceleration, "arrow_acc", False)

        self.arrow_velocity.material.shader = 'defaultLit'
        self.arrow_velocity.material.base_color = [1, 1, 0, 1.0]

        self.arrow_acceleration.material.shader = 'defaultLit'
        self.arrow_acceleration.material.base_color = [1, 0, 0, 1.0]

        # Mesh and Joints
        self.mesh = o3d.geometry.TriangleMesh()
        self.joints_pcl = o3d.geometry.PointCloud()
    
    def set_motion_data(self, frame_index): 
        self.current_frame = 0
        self.motion_data = AMASS_Motions[frame_index]
        self.play_slider.set_limits(1, self.motion_data.num_frames - 1)
        self.play_slider.int_value = self.current_frame
        self.extract_pose_from_amass()
        self.remove_vel_arrows()
        self.remove_acc_arrows()
        self.draw_mesh(self.current_frame)
        self.joint_angle_data = JointAngleData(self.motion_data.num_frames)
    
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r
        width = 23 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)
    
    def _on_combobox_amass_data(self, name, index):
        self.animated_state = AnimationState.PAUSED
        self.set_motion_data(index)
    
    def _on_play_slider(self, size):
        self.current_frame = int(size)

        if (self.animated_state == AnimationState.PAUSED):
            self.remove_vel_arrows()
            self.remove_acc_arrows()
            self.draw_mesh(self.current_frame)
            if (self.arrow_velocity.is_checked):
                self.calculate_draw_velocity(self.current_frame)
            if (self.arrow_acceleration.is_checked):
                self.calculate_draw_acceleration(self.current_frame)

    def _on_button_reset(self):
        self.animated_state = AnimationState.PAUSED
        self.current_frame = 0
        self.play_slider.int_value = 0

        self.remove_vel_arrows()
        self.remove_acc_arrows()
        self.draw_mesh(0)

    def _on_button_play(self):
        self.remove_vel_arrows()
        self.remove_acc_arrows()
        if self.animated_state == AnimationState.PAUSED:
            self.animated_state = AnimationState.PLAYING
            def thread_main():
                gui.Application.instance.post_to_main_thread(self.window, self.play_motion)
            threading.Thread(target=thread_main).start()
        else:
            self.animated_state = AnimationState.PLAYING

    def _on_button_stop(self):
        self.animated_state = AnimationState.PAUSED

    def _on_button_reverse(self):
        self.remove_vel_arrows()
        self.remove_acc_arrows()
        if (self.animated_state == AnimationState.PAUSED):
            self.animated_state = AnimationState.REVERSE
            def thread_main():
                gui.Application.instance.post_to_main_thread(self.window, self.play_motion)
            threading.Thread(target=thread_main).start()
        else:
            self.animated_state = AnimationState.REVERSE

    def _on_joint_size_slide(self, size):
        self.mat_dot.point_size = 4 + int(size)

    def _on_velocity_checkbox(self, is_checked):
        self.arrow_velocity.is_checked = is_checked
        if (not is_checked):
            self.remove_vel_arrows()
        if (is_checked and self.animated_state == AnimationState.PAUSED):
            self.calculate_draw_velocity(self.current_frame)

    def _on_acceleration_checkbox(self, is_checked):
        self.arrow_acceleration.is_checked = is_checked
        if (not is_checked):
            self.remove_acc_arrows()
        if (is_checked and self.animated_state == AnimationState.PAUSED):
            self.calculate_draw_acceleration(self.current_frame)

    def _on_plot_button(self):
        if (self.joint_angle_data.isEmpty):
            self.calculate_joint_angle_vel_acc()
        self.draw_plot()

    def remove_vel_arrows(self):
        for j in range(24):
            self.scene.scene.remove_geometry(self.arrow_velocity.string_id + str(j))

    def remove_acc_arrows(self):
        for j in range(24):
            self.scene.scene.remove_geometry(self.arrow_acceleration.string_id + str(j))

    def extract_pose_from_amass(self):
        self.poses = []
        for i in range(0, self.motion_data.num_frames):
            global_orient = torch.tensor(self.motion_data.dataset['root_orient'][i, :3].reshape(1, 3), dtype=torch.float32)
            pose1 = torch.tensor(self.motion_data.dataset['pose_body'][i, :63].reshape(1, 63), dtype=torch.float32)
            model = self.model(global_orient=global_orient, body_pose=pose1, betas=None)
            translation = self.motion_data.dataset['trans'][i, :3].reshape(1, 3)
            joint_positions = model.joints.detach().cpu().numpy().squeeze()
            indices = np.r_[0:22, 37, 52]
            joint_positions = joint_positions[indices] + translation
            vertices = model.vertices.detach().cpu().numpy().squeeze() + translation
            self.poses.append(Pose(joint_positions, vertices))

    def calculate_joint_angle_vel_acc(self):
        for i in range(0, self.motion_data.num_frames):
            vectors = [0] * 24
            for joint in joints:
                if (joint.parent_index != None):
                    parent_position = self.poses[i].joints[joint.parent_index]
                    current_position = self.poses[i].joints[joint.index]
                    vectors[joint.index] = JointAngleManager.calcuate_vector_pointed_from_parent(current_position, parent_position)

            for joint in joints:
                if (joint.child_index != None) :
                    # Joint angle
                    joint_angle = round(JointAngleManager.find_joint_angle(vectors[joint.index], vectors[joint.child_index]))
                    self.joint_angle_data.angles[joint.index, i] = joint_angle - joint.referecne_angle
                    # Joint angular velocity
                    if (i > 0):
                        self.joint_angle_data.velocities[joint.index, i-1] = self.joint_angle_data.angles[joint.index, i] - self.joint_angle_data.angles[joint.index, i-1]
                        self.joint_angle_data.velocities[joint.index, i-1] /= (1 / self.motion_data.frame_rate)
                    # Joint angular acceleeration
                    if (i > 1):
                        self.joint_angle_data.accelertions[joint.index, i-1] = self.joint_angle_data.velocities[joint.index, i-1] - self.joint_angle_data.velocities[joint.index, i-2]
                        self.joint_angle_data.accelertions[joint.index, i-1] /= (1 / self.motion_data.frame_rate)
        self.joint_angle_data.isEmpty = False
                        
    def draw_plot(self):
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Joint Angles", "Velocities", "Accelerations"), vertical_spacing=0.1)

        for i in range(22):
            if (joints[i].child_index != None and joints[i].is_checked):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.motion_data.num_frames)),
                        y=self.joint_angle_data.angles[i, :],
                        mode='lines',
                        name=joints[i].name,
                        line=dict(color=joints[i].color),
                        legendgroup='group1',
                        showlegend=True,
                    ),
                    row=1, col=1
                )

        for i in range(22):
             if (joints[i].child_index != None and joints[i].is_checked):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.motion_data.num_frames)),
                        y=self.joint_angle_data.velocities[i, :],
                        mode='lines',
                        name=joints[i].name,
                        line=dict(color=joints[i].color),
                        showlegend=False
                    ),
                    row=2, col=1
                )

        for i in range(22):
            if (joints[i].child_index != None and joints[i].is_checked):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.motion_data.num_frames)),
                        y=self.joint_angle_data.accelertions[i, :],
                        mode='lines',
                        name=joints[i].name,
                        line=dict(color=joints[i].color),
                        showlegend=False
                    ),
                    row=3, col=1
                )

        fig.update_layout(
            title="Joint Angles Over Time",
            height=1300
        )
        fig.show()

    def calculate_draw_velocity(self, frame):
        """
        Computes and visualizes joint velocities for a specified frame based on smoothing settings
         and passes the computed velocity vectors to 'draw_arrow()'.

        Args:
        - frame (int): Index of the current frame.
        """
        window_size = self.vel_arrow_setting.smoothing_size
        if (frame >= self.motion_data.num_frames - 1):
            return
        start_index = max(0, frame - math.floor(window_size/2))
        end_index = min(self.motion_data.num_frames-1, frame + math.ceil(window_size/2))
        vector_tail = np.zeros((24, 3))
        vector_head = np.zeros((24, 3))
        for j in range(start_index, end_index):
            vector_tail += self.poses[j].joints
            vector_head += self.poses[j+1].joints
        vector_tail /= window_size
        vector_head /= window_size

        # Calculate velocity vectors and magnitudes
        difference_vectors = vector_head - vector_tail 
        vectors = difference_vectors / (1 / self.motion_data.frame_rate)
        magnitudes = np.linalg.norm(vectors, axis=1)
        scaling_factor = magnitudes / self.motion_data.max_vel 
        # self.vel_max = np.max(magnitudes) if self.vel_max < np.max(magnitudes) else self.vel_max
        self.draw_arraw(self.poses[frame].joints, vectors, scaling_factor, self.arrow_velocity, self.vel_arrow_setting)

    def calculate_draw_acceleration(self, frame):
        """
        Computes joint acceleration for a specified frame based on smoothing settings
         and passes the computed acceleration vectors to 'draw_arrow()'.

        Args:
        - frame (int): Index of the current frame.
        """
        window_size = self.acc_arrow_setting.smoothing_size
        if (frame >= self.motion_data.num_frames - 1):
            return
        start_index = max(0, frame - math.floor(window_size/2))
        end_index = min(self.motion_data.num_frames-1, frame + math.ceil(window_size/2) + 1)
        velocity_vectors = []
        for j in range(start_index, end_index):
            velocity = (self.poses[j+1].joints - self.poses[j].joints) / (1 / self.motion_data.frame_rate)
            velocity_vectors.append(velocity)

        vectors = np.zeros((24, 3))
        for j in range(0, len(velocity_vectors)-1):
            vectors += (velocity_vectors[j+1] - velocity_vectors[j]) / (1 / self.motion_data.frame_rate)
        vectors /= (len(velocity_vectors) - 1)
        magnitudes = np.linalg.norm(vectors, axis=1)
        normalized_magnitudes = magnitudes / self.motion_data.max_acc
        # self.acc_max = np.max(magnitudes) if self.acc_max < np.max(magnitudes) else self.acc_max
        self.draw_arraw(self.poses[frame].joints, vectors, normalized_magnitudes, self.arrow_acceleration, self.acc_arrow_setting)

    def draw_mesh(self, frame):
        """
        Visualize a human body mesh in a scene for a specified frame.

        Args:
        - frame (int): Index of the current frame.
        """
        if (frame >= self.motion_data.num_frames - 1): 
            return
        
        self.scene.scene.remove_geometry('mesh')
        self.scene.scene.remove_geometry('joints')
        
        if (self.mesh_state == MeshState.Hidden or self.mesh_state == MeshState.Trasparent):
            self.joints_pcl.points = o3d.utility.Vector3dVector(self.poses[frame].joints)
            self.scene.scene.add_geometry('joints', self.joints_pcl, self.mat_dot)

        if (self.mesh_state != MeshState.Hidden):
            self.mesh.vertices = o3d.utility.Vector3dVector(self.poses[frame].vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(self.model.faces)
            if (self.mesh_state == MeshState.Trasparent):
                self.mat_mesh.shader = 'defaultLitTransparency'
                self.mesh.compute_vertex_normals()
            elif (self.mesh_state == MeshState.Opaque):
                self.mat_mesh.shader = 'defaultLit'
            self.scene.scene.add_geometry('mesh', self.mesh, self.mat_mesh)

    def draw_arraw(self, mesh_joints, vectors, scaling_factor, arrow_info, arrow_setting):
        """
        Draws arrows representing vectors at joint positions in a scene.

        Args:
        - mesh_joints (array): An array of 3D coordinates for the joints where arrows will be placed.
        - vectors (array): An array of 3D vectors representing the directions and magnitudes for the arrows.
        - scaling_factor (array): An array containing scaling values for each arrow, used to adjust the size proportionally.
        - arrow_info (object): An object containing properties used for drawing arrows.
        - arrow_setting (object): An object that contains user-defined settings for arrow properties.

        """
        self.arrows = [0] * 24
        for j in range(24):
            if (np.all(vectors[j] == 0) or np.isnan(vectors[j]).any()):
                continue
            start_position = mesh_joints[j]
            end_position = start_position + vectors[j]
            default_direction = np.array([0, 0, 1]) # arrow along the z-axis
            direction = np.array(end_position) - np.array(start_position)
            direction /= np.linalg.norm(direction)  # normalizations

            # Find a rotational matrix from the direction vector
            v = np.cross(default_direction, direction)
            w = np.sqrt(np.linalg.norm(default_direction)**2 * np.linalg.norm(direction)**2) + np.dot(default_direction, direction)
            quaternion = np.array([w, v[0], v[1], v[2]])
            quaternion /= np.linalg.norm(quaternion)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

            # Transformation matrices
            rotation_matrix = np.identity(4)
            rotation_matrix[:3, :3] = R
            translation_matrix = np.identity(4)
            translation_matrix[:3, 3] = start_position
            scailing = arrow_setting.arrow_size
            if arrow_setting.is_scailing_enabled:
                scailing *= scaling_factor[j]
            scailing = min(1.5, scailing) # if scailing > 1.5, only the length of the arrows gets scaled

            height_scailing = arrow_setting.arrow_size 
            if arrow_setting.is_scailing_enabled:
                height_scailing *= scaling_factor[j]

            self.arrows[j] = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.007 * scailing,
                                                    cone_radius=0.015 * scailing,
                                                    cylinder_height=0.1 * height_scailing,
                                                    cone_height=0.04 * scailing)
            self.arrows[j].compute_vertex_normals()

            self.arrows[j].transform(rotation_matrix)
            self.arrows[j].transform(translation_matrix)

            self.scene.scene.remove_geometry(arrow_info.string_id + str(j))
            self.scene.scene.add_geometry(arrow_info.string_id + str(j), self.arrows[j], arrow_info.material)

    def play_motion(self):
        if (self.motion_data == None):
            return
        
        # self.vel_max = 0
        # self.acc_max = 0

        running = True
        while running:
            if (self.arrow_velocity.is_checked):
                self.calculate_draw_velocity(self.current_frame)

            if (self.arrow_acceleration.is_checked):
                self.calculate_draw_acceleration(self.current_frame)

            self.draw_mesh(self.current_frame)

            self.current_frame = self.current_frame + 1 if self.animated_state == AnimationState.PLAYING else self.current_frame - 1
            tick_return = gui.Application.instance.run_one_tick()
            if tick_return:
                self.window.post_redraw()
            self.play_slider.int_value = self.current_frame
                
            if (self.animated_state == AnimationState.PAUSED):
                running = False
            if (self.animated_state == AnimationState.PLAYING
                and self.current_frame >= (self.motion_data.num_frames - 1)): 
                running = False
                self.animated_state = AnimationState.PAUSED
            if (self.animated_state == AnimationState.REVERSE and self.current_frame < 1):
                running = False
                self.animated_state = AnimationState.PAUSED

        # print ("vel_max: ", self.vel_max)
        # print ("acc_max: ", self.acc_max)

def main():

    gui.Application.instance.initialize()
    VisualizationApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()