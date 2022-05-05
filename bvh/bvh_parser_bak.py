import collections

import torch
import bvh.BVH as BVH
import numpy as np
from bvh.Quaternions import Quaternions
from models.kinematics import ForwardKinematicsJoint
from models.transforms import quat2repr6d, quat2mat
from models.contact import foot_contact
from bvh.bvh_writer import WriterWrapper


# Mixamo 1, Mixamo 2, Dogs, CMU, Monkey
corps_name_1 = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_3 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_2_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_boss2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'Left_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Right_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_cmu = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_monkey = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms = ['Three_Arms_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms_split = ['Three_Arms_split_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_split', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_split']
corps_name_Prisoner = ['HipsPrisoner', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm']
corps_name_mixamo2_m = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_cylinder = ['Bone1', 'Bone2']
coprs_name_SMPL = ['f_avg_root', 'f_avg_Pelvis', 'f_avg_L_Hip', 'f_avg_L_Knee', 'f_avg_L_Ankle', 'f_avg_L_Foot', 'f_avg_R_Hip', 'f_avg_R_Knee', 'f_avg_R_Ankle', 'f_avg_R_Foot', 'f_avg_Spine1', 'f_avg_Spine2', 'f_avg_Spine3', 'f_avg_Neck', 'f_avg_Head', 'f_avg_L_Collar', 'f_avg_L_Shoulder', 'f_avg_L_Elbow', 'f_avg_L_Wrist', 'f_avg_L_Hand', 'f_avg_R_Collar', 'f_avg_R_Shoulder', 'f_avg_R_Elbow', 'f_avg_R_Wrist', 'f_avg_R_Hand']
corps_name_xia = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightFingerBase', 'RightHandIndex1', 'RThumb']
corps_crab = ['Hips', 'BN_Bip01_Pelvis', 'BN_leg_R_01', 'BN_leg_R_02', 'BN_leg_R_03', 'BN_leg_R_04', 'BN_leg_R_05', 'BN_Leg_R_05Nub', 'BN_leg_R_06', 'BN_Leg_R_07', 'BN_Leg_R_08', 'BN_Leg_R_09', 'BN_Leg_R_10', 'BN_Leg_R_10Nub', 'BN_Leg_R_11', 'BN_Leg_R_12', 'BN_Leg_R_13', 'BN_Leg_R_14', 'BN_Leg_R_15', 'BN_Leg_R_15Nub', 'BN_Leg_L_11', 'BN_Leg_L_12', 'BN_Leg_L_13', 'BN_Leg_L_14', 'BN_Leg_L_15', 'BN_Leg_L_15Nub', 'BN_leg_L_06', 'BN_Leg_L_07', 'BN_Leg_L_08', 'BN_Leg_L_09', 'BN_Leg_L_10', 'BN_Leg_L_10Nub', 'BN_leg_L_01', 'BN_leg_L_02', 'BN_leg_L_03', 'BN_leg_L_04', 'BN_leg_L_05', 'BN_Leg_L_05Nub', 'BN_Eye_L_01', 'BN_Eye_L_02', 'BN_Eye_L_03', 'BN_Eye_L_04', 'BN_Eye_R_01', 'BN_Eye_R_02', 'BN_Eye_R_03', 'BN_Eye_R_04', 'BN_Arm_L_01', 'BN_Arm_L_02', 'BN_Arm_L_03', 'BN_Arm_L_04', 'BN_Arm_R_01', 'BN_Arm_R_02', 'BN_Arm_R_03', 'BN_Arm_R_04']
corps_anaconda = ['Hips', 'BN_Tail_01', 'BN_Tail_02', 'BN_Tail_03', 'BN_Tail_04', 'BN_Tail_05', 'BN_Tail_06', 'BN_Tail_07', 'BN_Tail_08', 'BN_Tail_09', 'BN_Tail_10', 'BN_Tail_11', 'BN_Tail_12', 'BN_Tail_13', 'BN_Spline_01', 'BN_Spline_02', 'BN_Spline_03', 'BN_Spline_04', 'BN_Spline_05', 'BN_Spline_06', 'BN_Neck', 'BN_Head', 'BN_Jaw', 'BN_Tone_01', 'BN_Tone_02', 'BN_Tone_03', 'BN_Tone_04']
corps_name_eagle = ['Hips', 'Bip01_Pelvis', 'BN_Tai_01', 'BN_Tai_R_01', 'BN_Tai_R_02', 'BN_Tai_L_01', 'BN_Tai_L_02', 'Bip01_Spine', 'Bip01_R_Thigh', 'Bip01_R_Calf', 'Bip01_R_HorseLink', 'Bip01_R_Foot', 'BN_Toe_R_30', 'BN_Toe_R_31', 'BN_Toe_R_00', 'BN_Toe_R_01', 'BN_Toe_R_20', 'BN_Toe_R_21', 'BN_Toe_R_10', 'BN_Toe_R_11', 'Bip01_R_Toe0', 'Bip01_L_Thigh', 'Bip01_L_Calf', 'Bip01_L_HorseLink', 'Bip01_L_Foot', 'BN_Toe_L_00', 'BN_Toe_L_01', 'BN_Toe_L_10', 'BN_Toe_L_11', 'BN_Toe_L_30', 'BN_Toe_L_31', 'BN_Toe_L_20', 'BN_Toe_L_21', 'Bip01_L_Toe0', 'Bip01_Spine1', 'BN_Wing_R_01', 'BN_Wing_R_02', 'BN_Wing_R_03', 'BN_Wing_R_05', 'BN_Wing_R_06', 'BN_Wing_R_04', 'BN_Wing_L_01', 'BN_Wing_L_02', 'BN_Wing_L_03', 'BN_Wing_L_05', 'BN_Wing_L_06', 'BN_Wing_L_04', 'Bip01_Neck', 'Bip01_Neck1', 'Bip01_Neck2', 'Bip01_Head', 'BN_Jaw']
corps_name_elephant = ['Hips', 'Bip01_Pelvis', 'BN_Tail_01', 'BN_Tail_02', 'BN_Tail_03', 'BN_Tail_04', 'Bip01_Spine', 'Bip01_R_Thigh', 'Bip01_R_Calf', 'Bip01_R_Foot', 'Bip01_R_Toe0', 'Bip01_L_Thigh', 'Bip01_L_Calf', 'Bip01_L_Foot', 'Bip01_L_Toe0', 'Bip01_Spine1', 'Bip01_Spine2', 'Bip01_Neck', 'Bip01_Head', 'BN_Eyebrow_L', 'BN_Eyebrow_R', 'BN_Ear_L_01', 'BN_Ear_L_02', 'BN_Mouth_01', 'BN_Ear_R_01', 'BN_Ear_R_02', 'BN_Nose_01', 'BN_Nose_02', 'BN_Nose_03', 'BN_Nose_04', 'BN_Nose_05', 'BN_Nose_06', 'Bip01_R_Clavicle', 'Bip01_R_UpperArm', 'Bip01_R_Forearm', 'Bip01_R_Hand', 'Bip01_L_Clavicle', 'Bip01_L_UpperArm', 'Bip01_L_Forearm', 'Bip01_L_Hand']
corps_name_crabnew = ['ORG_Hips', 'ORG_BN_Bip01_Pelvis', 'DEF_BN_Eye_L_01', 'DEF_BN_Eye_L_02', 'DEF_BN_Eye_L_03', 'DEF_BN_Eye_L_03_end', 'DEF_BN_Eye_R_01', 'DEF_BN_Eye_R_02', 'DEF_BN_Eye_R_03', 'DEF_BN_Eye_R_03_end', 'DEF_BN_Leg_L_11', 'DEF_BN_Leg_L_12', 'DEF_BN_Leg_L_13', 'DEF_BN_Leg_L_14', 'DEF_BN_Leg_L_15', 'DEF_BN_Leg_L_15_end', 'DEF_BN_Leg_R_11', 'DEF_BN_Leg_R_12', 'DEF_BN_Leg_R_13', 'DEF_BN_Leg_R_14', 'DEF_BN_Leg_R_15', 'DEF_BN_Leg_R_15_end', 'DEF_BN_leg_L_01', 'DEF_BN_leg_L_02', 'DEF_BN_leg_L_03', 'DEF_BN_leg_L_04', 'DEF_BN_leg_L_05', 'DEF_BN_leg_L_05_end', 'DEF_BN_leg_L_06', 'DEF_BN_Leg_L_07', 'DEF_BN_Leg_L_08', 'DEF_BN_Leg_L_09', 'DEF_BN_Leg_L_10', 'DEF_BN_Leg_L_10_end', 'DEF_BN_leg_R_01', 'DEF_BN_leg_R_02', 'DEF_BN_leg_R_03', 'DEF_BN_leg_R_04', 'DEF_BN_leg_R_05', 'DEF_BN_leg_R_05_end', 'DEF_BN_leg_R_06', 'DEF_BN_Leg_R_07', 'DEF_BN_Leg_R_08', 'DEF_BN_Leg_R_09', 'DEF_BN_Leg_R_10', 'DEF_BN_Leg_R_10_end', 'DEF_BN_Bip01_Pelvis', 'DEF_BN_Bip01_Pelvis_end', 'DEF_BN_Arm_L_01', 'DEF_BN_Arm_L_02', 'DEF_BN_Arm_L_03', 'DEF_BN_Arm_L_03_end', 'DEF_BN_Arm_R_01', 'DEF_BN_Arm_R_02', 'DEF_BN_Arm_R_03', 'DEF_BN_Arm_R_03_end']
corps_name_crabnew2 = ['DEF_BN_Bip01_Pelvis', 'DEF_BN_Eye_L_01', 'DEF_BN_Eye_L_02', 'DEF_BN_Eye_L_03', 'DEF_BN_Eye_L_03_end', 'DEF_BN_Eye_R_01', 'DEF_BN_Eye_R_02', 'DEF_BN_Eye_R_03', 'DEF_BN_Eye_R_03_end', 'DEF_BN_Leg_L_11', 'DEF_BN_Leg_L_12', 'DEF_BN_Leg_L_13', 'DEF_BN_Leg_L_14', 'DEF_BN_Leg_L_15', 'DEF_BN_Leg_L_15_end', 'DEF_BN_Leg_R_11', 'DEF_BN_Leg_R_12', 'DEF_BN_Leg_R_13', 'DEF_BN_Leg_R_14', 'DEF_BN_Leg_R_15', 'DEF_BN_Leg_R_15_end', 'DEF_BN_leg_L_01', 'DEF_BN_leg_L_02', 'DEF_BN_leg_L_03', 'DEF_BN_leg_L_04', 'DEF_BN_leg_L_05', 'DEF_BN_leg_L_05_end', 'DEF_BN_leg_L_06', 'DEF_BN_Leg_L_07', 'DEF_BN_Leg_L_08', 'DEF_BN_Leg_L_09', 'DEF_BN_Leg_L_10', 'DEF_BN_Leg_L_10_end', 'DEF_BN_leg_R_01', 'DEF_BN_leg_R_02', 'DEF_BN_leg_R_03', 'DEF_BN_leg_R_04', 'DEF_BN_leg_R_05', 'DEF_BN_leg_R_05_end', 'DEF_BN_leg_R_06', 'DEF_BN_Leg_R_07', 'DEF_BN_Leg_R_08', 'DEF_BN_Leg_R_09', 'DEF_BN_Leg_R_10', 'DEF_BN_Leg_R_10_end', 'DEF_BN_Arm_L_01', 'DEF_BN_Arm_L_02', 'DEF_BN_Arm_L_03', 'DEF_BN_Arm_L_03_end', 'DEF_BN_Arm_R_01', 'DEF_BN_Arm_R_02', 'DEF_BN_Arm_R_03', 'DEF_BN_Arm_R_03_end']
corps_name_smpl2 = ['pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'jaw', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_eye_smplhf']
corps_name_dog = ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_end', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_end', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_end', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_end', 'Tail', 'Tail1']
corps_name_bear = ['RigPelvis', 'RigSpine1', 'RigSpine2', 'RigChest', 'RigNeck1', 'RigNeck2', 'RigHead', 'RigJaw', 'RigTongue_1', 'RigTongue_2', 'RigTongue_3', 'RigTongue_3_end', 'RigREyelid_R', 'RigREyelid', 'RigRigREar1_R', 'RigRigREar1_R_end', 'RigREyelid_L', 'RigREyelid', 'RigRigREar1_L', 'RigRigREar1_L_end', 'RigRFLegCollarbone_R', 'RigRFLeg1_R', 'RigRFLeg2_R', 'RigRFLegAnkle_R', 'RigRFLegDigit11_R', 'RigRFLegDigit11_R_end', 'RigRFLegCollarbone_L', 'RigRFLeg1_L', 'RigRFLeg2_L', 'RigRFLegAnkle_L', 'RigRFLegDigit11_L', 'RigRFLegDigit11_L_end', 'RigTail1', 'RigTail2', 'RigTail2_end', 'RigRBLeg1_R', 'RigRBLeg2_R', 'RigRBLegAnkle_R', 'RigRBLegDigit11_R', 'RigRBLegDigit11_R_end', 'RigRBLeg1_L', 'RigRBLeg2_L', 'RigRBLegAnkle_L', 'RigRBLegDigit11_L', 'RigRBLegDigit11_L_end']
corps_name_bear = ['RigPelvis', 'RigSpine1', 'RigSpine2', 'RigChest', 'RigNeck1', 'RigNeck2', 'RigHead', 'RigJaw', 'RigTongue_1', 'RigTongue_2', 'RigTongue_3', 'RigTongue_3_end', 'RigREyelid_R', 'RigREyelid_R_end', 'RigRigREar1_R', 'RigRigREar1_R_end', 'RigREyelid_L', 'RigREyelid_L_end', 'RigRigREar1_L', 'RigRigREar1_L_end', 'RigRFLegCollarbone_R', 'RigRFLeg1_R', 'RigRFLeg2_R', 'RigRFLegAnkle_R', 'RigRFLegDigit11_R', 'RigRFLegDigit11_R_end', 'RigRFLegCollarbone_L', 'RigRFLeg1_L', 'RigRFLeg2_L', 'RigRFLegAnkle_L', 'RigRFLegDigit11_L', 'RigRFLegDigit11_L_end', 'RigTail1', 'RigTail2', 'RigTail2_end', 'RigRBLeg1_R', 'RigRBLeg2_R', 'RigRBLegAnkle_R', 'RigRBLegDigit11_R', 'RigRBLegDigit11_R_end', 'RigRBLeg1_L', 'RigRBLeg2_L', 'RigRBLegAnkle_L', 'RigRBLegDigit11_L', 'RigRBLegDigit11_L_end']

ee_name_1 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_2 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_3 = ['LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
ee_name_cmu = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_monkey = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_three_arms_split = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand_split', 'RightHand_split']
ee_name_Prisoner = ['LeftToe_End', 'RightToe_End', 'HeadTop_End', 'LeftHand', 'RightForeArm']
ee_name_cylinder = ['Bone1', 'Bone2']
ee_name_SMPL = ['f_avg_L_Foot', 'f_avg_R_Foot', 'f_avg_Head']
ee_name_2_2 = ['LeftToe_End', 'RightToe_End', 'Head', 'LeftHand', 'RightHand']
ee_name_acrnn = ['lFoot', 'rFoot', 'head', 'lHand', 'rHand']
ee_crab = [name for name in corps_crab if 'Nub' in name]
ee_anaconda = corps_anaconda[:4]
ee_name_eagle = corps_name_eagle[:4]
ee_name_elephant = [name for name in corps_name_elephant if 'Hand' in name or 'Toe' in name or 'Foot' in name]
# ee_name_crabnew = [name for name in corps_name_crabnew if '05' in name or '10' in name or '15' in name]
ee_name_crabnew = [name for name in corps_name_crabnew if 'end' in name and ('05' in name or '10' in name or '15' in name)]
ee_name_crabnew2 = ee_name_crabnew
ee_name_smpl2 = ['left_foot', 'right_foot', 'head', 'left_eye_smplhf', 'jaw']
ee_names_dog = [name for name in corps_name_dog if '_end' in name]
ee_name_bear = [name for name in corps_name_bear if 'LegDigit11' in name]


crab_contact_name = ee_crab
crabnew_contact_name = ee_name_crabnew
crabnew2_contact_name = crabnew_contact_name
dog_contact_name = ee_names_dog
smpl2_contact_name = ['left_ankle', 'left_foot', 'right_ankle', 'right_foot']
bear_contact_name = ee_name_bear


skeleton_class_name = ['Mixamo 1', 'Mixamo 2', 'Dogs', 'CMU', 'Monkey']

corps_names = [corps_name_1, corps_name_2, corps_name_3, corps_name_cmu, corps_name_monkey, corps_name_2_2,
               corps_name_boss2, corps_name_three_arms, corps_name_three_arms_split, corps_name_Prisoner,
               corps_name_mixamo2_m, corps_name_cylinder, coprs_name_SMPL]
ee_names = [ee_name_1, ee_name_2, ee_name_3, ee_name_cmu, ee_name_monkey, ee_name_2_2, ee_name_1, ee_name_1,
            ee_name_three_arms_split, ee_name_Prisoner, ee_name_2, ee_name_cylinder, ee_name_SMPL]


animal_corps_names = [corps_crab, corps_anaconda, corps_name_eagle, corps_name_elephant, corps_name_crabnew, corps_name_crabnew2, corps_name_smpl2, corps_name_dog, corps_name_bear]
animal_ee_names = [ee_crab, ee_anaconda, ee_name_eagle, ee_name_elephant, ee_name_crabnew, ee_name_crabnew2, ee_name_smpl2, ee_names_dog, ee_name_bear]
animal_contact_names = [crab_contact_name, ee_anaconda, ee_name_eagle, ee_name_elephant, crabnew_contact_name, crabnew2_contact_name, smpl2_contact_name, dog_contact_name, bear_contact_name]
animal_names = ['crab', 'anaconda', 'eagle', 'elephant', 'crabnew', 'crabnew2', 'smpl2', 'dog', 'bear']

contact_thresholds = {'default': 0.018, 'crab': 0.006, 'anaconda': -1., 'eagle': -1, 'elephant': 0.02, 'crabnew': 0.006,
                      'crabnew2': 0.006, 'smpl2': 0.018, 'dog': 0.018, 'bear': 0.005}


height_base = 3.7175894766954465


class BVH_file:
    def __init__(self, file_path, no_scale=False, requires_contact=False, heel_contact=False, joint_reduction=True,
                 use_toe_end=False, passive_detect=False, scale_to_height=False, ideal_traj='', contact_threshold_ratio=-1.,
                 use_beat=False):
        self.anim, self._names, self.frametime = BVH.load(file_path)
        if self.frametime < 0.0084:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]
        if self.frametime < 0.017:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]
        self.skeleton_type = -1
        self._topology = None
        self.ideal_traj = ideal_traj
        self.ee_length = []
        self.requires_contact = requires_contact
        self.animal_id = -1
        contact_threshold = contact_thresholds['default']

        if use_beat:
            self.beat = np.load(file_path[:-4] + '_beat.npy')
            self.anim.positions = self.anim.positions[:self.beat.shape[0]]
            self.anim.rotations = self.anim.rotations[:self.beat.shape[0]]
        else:
            self.beat = None

        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                self._names[i] = name

        full_fill = [1] * len(corps_names)
        for i, ref_names in enumerate(corps_names):
            for ref_name in ref_names:
                if ref_name not in self._names:
                    full_fill[i] = 0
                    break

        if full_fill[3]:
            self.skeleton_type = 3
        else:
            for i, _ in enumerate(full_fill):
                if full_fill[i]:
                    self.skeleton_type = i
                    break

        if self.skeleton_type == 2 and full_fill[4]:
            self.skeleton_type = 4

        if 'Neck1' in self._names:
            self.skeleton_type = 5
        if 'Left_End' in self._names:
            self.skeleton_type = 6
        if 'Three_Arms_Hips' in self._names:
            self.skeleton_type = 7
        if 'Three_Arms_Hips_split' in self._names:
            self.skeleton_type = 8

        if 'LHipJoint' in self._names:
            self.skeleton_type = 3

        if 'HipsPrisoner' in self._names:
            self.skeleton_type = 9

        if 'Spine1_split' in self._names:
            self.skeleton_type = 10
            
        if 'Bone1' in self._names:
            self.skeleton_type = 11

        if 'f_avg_root' in self._names:
            self.skeleton_type = 12

        if '00' in self._names:
            corps_names.append(sorted(self._names))
            ee_names.append([])
            self.skeleton_type = len(corps_names) - 1

        if 'LHipJoint' in self._names:
            self.skeleton_type = 14
            corps_names.append([])  # placeholder
            corps_names.append(corps_name_xia)
            ee_names.append([])
            ee_names.append(ee_name_1)

        if 'rShin' in self._names: # acrnn
            self.skeleton_type = 15
            corps_names.append([])
            corps_names.append([])
            corps_names.append(self._names)
            ee_names.append([])
            ee_names.append([])
            ee_names.append(ee_name_acrnn)

        # for i in range(len(self._names)):
        #     print(self._names[i] == animal_corps_names[-1][i], self._names[i], animal_corps_names[-1][i])

        for i, name in enumerate(animal_names):
            def list_equal(a, b):
                return collections.Counter(a) == collections.Counter(b)
            if list_equal(animal_corps_names[i], self._names):
                if corps_names[-1] != animal_corps_names[i]:
                    corps_names.append(animal_corps_names[i])
                if ee_names[-1] != animal_ee_names[i]:
                    ee_names.append(animal_ee_names[i])
                self.skeleton_type = len(corps_names) - 1
                self.animal_id = i
                contact_threshold = contact_thresholds[name]
                # print('Found Animal:', name)
                break

        if not joint_reduction and self.animal_id == -1:
            self.skeleton_type = len(corps_names)
            corps_names.append(self._names)
            if 'LeftToe_End' in self._names:
                ee_names.append(ee_name_2_2)
            else:
                # Those are placeholders
                idx = [0, 1, 2, 3]
                ee_name = [self._names[i] for i in idx]
                ee_names.append(ee_name)

        if joint_reduction and (use_toe_end or (passive_detect and "LeftToe_End" in self._names)):
            if self.skeleton_type == 1:
                self.skeleton_type = 5
            else:
                raise Exception('Something is wrong with the bvh file!')

        if self.skeleton_type == -1:
            print(self._names)
            raise Exception('Unknown skeleton')

        if self.skeleton_type in [0, 1, 5] and not no_scale: # Mixamo
            self.anim.offsets /= 100
            self.anim.positions /= 100

        if self.skeleton_type == 14 and not no_scale:  # Xia et al.
            self.anim.offsets *= 0.14
            self.anim.positions *= 0.14
            # self.anim.rotations = self.anim.rotations[::4]
            # self.anim.positions = self.anim.positions[::4]

        if self.skeleton_type == 15 and not no_scale:
            self.anim.offsets *= 0.022
            self.anim.positions *= 0.022
            self.anim.rotations = self.anim.rotations[::2]
            self.anim.positions = self.anim.positions[::2]

        if requires_contact:
            if self.skeleton_type == 5:
                self.contact_names = ['LeftToe_End', 'RightToe_End']
                if heel_contact:
                    self.contact_names += ['LeftToeBase', 'RightToeBase']
            elif self.skeleton_type == 15:
                self.contact_names = ['lFoot', 'rFoot']
                if heel_contact:
                    pass
            elif self.animal_id != -1:
                self.contact_names = animal_contact_names[self.animal_id]
            else:
                # Xia et al.
                self.contact_names = ['LeftToeBase', 'RightToeBase']
                if heel_contact:
                    self.contact_names += ['LeftFoot', 'RightFoot']
        else:
            self.contact_names = []

        self.details = []
        for i, name in enumerate(self._names):
            if ':' in name: name = name[name.find(':')+1:]
            if name not in corps_names[self.skeleton_type]: self.details.append(i)
        self.joint_num = self.anim.shape[1]
        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        for name in corps_names[self.skeleton_type]:
            for j in range(self.anim.shape[1]):
                if name in self._names[j]:
                    self.corps.append(j)
                    break

        if len(self.corps) != len(corps_names[self.skeleton_type]):
            for i in self.corps: print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in file', file_path)

        self.ee_id = []
        for i in ee_names[self.skeleton_type]:
            self.ee_id.append(corps_names[self.skeleton_type].index(i))

        self.contact_id = []
        for i in self.contact_names:
            self.contact_id.append(corps_names[self.skeleton_type].index(i))

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(self.anim.shape[1]):
            if i in self.details:
                self.simplify_map[i] = -1

        if scale_to_height:
            ratio = height_base / self.get_height()
            self.anim.offsets *= ratio
            self.anim.positions *= ratio

        if contact_threshold_ratio > 0:
            contact_threshold *= contact_threshold_ratio

        self.fk = ForwardKinematicsJoint(self.topology, self.offset)
        self.writer = WriterWrapper(self.topology, self.offset)
        if self.requires_contact:
            gl_pos = self.joint_position()
            self.contact_label = foot_contact(gl_pos[:, self.contact_id], threshold=contact_threshold)
            if self.animal_id == 8:
                self.contact_label[250:252] = 1
            self.gl_pos = gl_pos

    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]
            self._topology = tuple(self._topology)
        return self._topology

    def local_pos(self):
        gl_pos = self.joint_position()
        local_pos = gl_pos - gl_pos[:, 0:1, :]
        return local_pos[:, 1:]

    def get_ee_id(self):
        return self.ee_id

    def get_lowest_parent(self, names):
        res = []

        for i, name in enumerate(names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                names[i] = name
            names[i] = self._names.index(name)

        for i in names:
            while self.simplify_map[i] == -1:
                i = self.anim.parents[i]
            res.append(self.simplify_map[i])
        return res

    def to_tensor(self, repr='euler', rot_only=False):
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
            raise Exception('Unknown rotation representation')
        positions = self.get_position()
        rotations = self.get_rotation(repr=repr)

        if rot_only:
            return rotations.reshape(rotations.shape[0], -1)

        if self.requires_contact:
            virtual_contact = torch.zeros_like(rotations[:, :len(self.contact_id)])
            virtual_contact[..., 0] = self.contact_label
            rotations = torch.cat([rotations, virtual_contact], dim=1)

        rotations = rotations.reshape(rotations.shape[0], -1)
        return torch.cat((rotations, positions), dim=-1)

    def joint_position(self):
        positions = torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)
        rotations = self.anim.rotations[:, self.corps, :]
        rotations = Quaternions.from_euler(np.radians(rotations)).qs
        rotations = torch.tensor(rotations, dtype=torch.float)
        j_loc = self.fk.forward(rotations, positions)
        return j_loc

    def get_rotation(self, repr='quat'):
        rotations = self.anim.rotations[:, self.corps, :]
        if repr == 'quaternion' or repr == 'quat' or repr == 'repr6d':
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            rotations = torch.tensor(rotations, dtype=torch.float)
        if repr == 'repr6d':
            rotations = quat2repr6d(rotations)
        if repr == 'euler':
            rotations = torch.tensor(rotations, dtype=torch.float)
        return rotations

    def get_position(self):
        return torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)

    @property
    def offset(self):
        return torch.tensor(self.anim.offsets[self.corps], dtype=torch.float)

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology

        res = 0
        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        return res

    # def write(self, file_path):
    #     motion = self.to_numpy(quater=False, edge=False)
    #     rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
    #     positions = motion[..., -3:]
    #     write_bvh(self.topology, self.offset, rotations, positions, self.names, 1.0 / 30, 'xyz', file_path)

    # def from_numpy(self, motions, frametime=None, quater=False):
    #     if frametime is not None:
    #         self.frametime = frametime
    #     motions = motions.copy()
    #     positions = motions[:, -3:]
    #     self.anim.positions = positions[:, np.newaxis, :]
    #     if quater:
    #         rotations = motions[:, :-3].reshape(motions.shape[0], -1, 4)
    #         norm = rotations[:, :, 0]**2 + rotations[:, :, 1]**2 + rotations[:, :, 2]**2 + rotations[:, :, 3]**2
    #         norm = np.repeat(norm[:, :, np.newaxis], 4, axis=2)
    #         rotations /= norm
    #         rotations = Quaternions(rotations)
    #         rotations = np.degrees(rotations.euler())
    #     else:
    #         rotations = motions[:, -3:].reshape(motions.shape[0], -1, 3)
    #     rotations_full = np.zeros((rotations.shape[0], self.anim.shape[1], 3))
    #
    #     for i in range(self.anim.shape[1]):
    #         if i in self.corps:
    #             pt = self.corps.index(i)
    #             rotations_full[:, i, :] = rotations[:, pt, :]
    #     self.anim.rotations = rotations_full

    def get_ee_length(self):
        if len(self.ee_length): return self.ee_length
        degree = [0] * len(self.topology)
        for i in self.topology:
            if i < 0: continue
            degree[i] += 1

        for j in self.ee_id:
            length = 0
            while degree[j] <= 1:
                t = self.offset[j]
                length += np.dot(t, t) ** 0.5
                j = self.topology[j]

            self.ee_length.append(length)

        height = self.get_height()
        ee_group = [[0, 1], [2], [3, 4]]
        for group in ee_group:
            maxv = 0
            for j in group:
                maxv = max(maxv, self.ee_length[j])
            for j in group:
                self.ee_length[j] *= height / maxv

        return self.ee_length

    def dfs(self, x, vis, dist):
        fa = self.topology
        vis[x] = 1
        for y in range(len(fa)):
            if (fa[y] == x or fa[x] == y) and vis[y] == 0:
                dist[y] = dist[x] + 1
                self.dfs(y, vis, dist)

    def get_neighbor(self, threshold, enforce_lower=False, enforce_contact=False):
        fa = self.topology
        neighbor_list = []
        for x in range(0, len(fa)):
            vis = [0 for _ in range(len(fa))]
            dist = [0 for _ in range(len(fa))]
            self.dfs(x, vis, dist)
            neighbor = []
            for j in range(0, len(fa)):
                if dist[j] <= threshold:
                    neighbor.append(j)
            neighbor_list.append(neighbor)

        contact_list = []
        if self.requires_contact:
            for i, p_id in enumerate(self.contact_id):
                v_id = len(neighbor_list)
                neighbor_list[p_id].append(v_id)
                neighbor_list.append(neighbor_list[p_id])
                contact_list.append(v_id)

        root_neighbor = neighbor_list[0]
        lower_body_list = list(range(0, self.ee_id[1]))

        id_ideal_rot = -1
        id_ideal_pos = -1
        id_root = len(neighbor_list) + int('rot' in self.ideal_traj) + int('loc' in self.ideal_traj)
        if 'rot' in self.ideal_traj:
            id_ideal_rot = len(neighbor_list)
            neighbor_list.append([0, id_root])
        if 'loc' in self.ideal_traj:
            id_ideal_pos = len(neighbor_list)
            neighbor_list.append([0, id_root])

        if self.beat is not None:
            id_beat = len(neighbor_list)
            neighbor_list.append(list(range(0, id_beat + 2)))

        id_root = len(neighbor_list)

        if enforce_contact:
            root_neighbor = root_neighbor + contact_list
            for j in contact_list:
                if enforce_lower:
                    neighbor_list[j] += lower_body_list
                neighbor_list[j] = list(set(neighbor_list[j]))
        if enforce_lower:
            root_neighbor = root_neighbor + lower_body_list + contact_list

        root_neighbor = list(set(root_neighbor))
        for j in root_neighbor:
            neighbor_list[j].append(id_root)
        root_neighbor.append(id_root)
        neighbor_list.append(root_neighbor)  # Neighbor for root position
        if id_ideal_rot != -1:
            neighbor_list[0].append(id_ideal_rot)
            neighbor_list[id_root].append(id_ideal_rot)
        if id_ideal_pos != -1:
            neighbor_list[0].append(id_ideal_pos)
            neighbor_list[id_root].append(id_ideal_pos)

        if self.beat is not None:
            for i in range(len(neighbor_list)):
                if i != id_beat:
                    neighbor_list[i].append(id_beat)
        return neighbor_list
