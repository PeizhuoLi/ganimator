import copy


mixamo_corps_name = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
elephant_corps_name = ['Hips', 'Bip01_Pelvis', 'BN_Tail_01', 'BN_Tail_02', 'BN_Tail_03', 'BN_Tail_04', 'Bip01_Spine', 'Bip01_R_Thigh', 'Bip01_R_Calf', 'Bip01_R_Foot', 'Bip01_R_Toe0', 'Bip01_L_Thigh', 'Bip01_L_Calf', 'Bip01_L_Foot', 'Bip01_L_Toe0', 'Bip01_Spine1', 'Bip01_Spine2', 'Bip01_Neck', 'Bip01_Head', 'BN_Eyebrow_L', 'BN_Eyebrow_R', 'BN_Ear_L_01', 'BN_Ear_L_02', 'BN_Mouth_01', 'BN_Ear_R_01', 'BN_Ear_R_02', 'BN_Nose_01', 'BN_Nose_02', 'BN_Nose_03', 'BN_Nose_04', 'BN_Nose_05', 'BN_Nose_06', 'Bip01_R_Clavicle', 'Bip01_R_UpperArm', 'Bip01_R_Forearm', 'Bip01_R_Hand', 'Bip01_L_Clavicle', 'Bip01_L_UpperArm', 'Bip01_L_Forearm', 'Bip01_L_Hand']
crab_dance_corps_name = ['ORG_Hips', 'ORG_BN_Bip01_Pelvis', 'DEF_BN_Eye_L_01', 'DEF_BN_Eye_L_02', 'DEF_BN_Eye_L_03', 'DEF_BN_Eye_L_03_end', 'DEF_BN_Eye_R_01', 'DEF_BN_Eye_R_02', 'DEF_BN_Eye_R_03', 'DEF_BN_Eye_R_03_end', 'DEF_BN_Leg_L_11', 'DEF_BN_Leg_L_12', 'DEF_BN_Leg_L_13', 'DEF_BN_Leg_L_14', 'DEF_BN_Leg_L_15', 'DEF_BN_Leg_L_15_end', 'DEF_BN_Leg_R_11', 'DEF_BN_Leg_R_12', 'DEF_BN_Leg_R_13', 'DEF_BN_Leg_R_14', 'DEF_BN_Leg_R_15', 'DEF_BN_Leg_R_15_end', 'DEF_BN_leg_L_01', 'DEF_BN_leg_L_02', 'DEF_BN_leg_L_03', 'DEF_BN_leg_L_04', 'DEF_BN_leg_L_05', 'DEF_BN_leg_L_05_end', 'DEF_BN_leg_L_06', 'DEF_BN_Leg_L_07', 'DEF_BN_Leg_L_08', 'DEF_BN_Leg_L_09', 'DEF_BN_Leg_L_10', 'DEF_BN_Leg_L_10_end', 'DEF_BN_leg_R_01', 'DEF_BN_leg_R_02', 'DEF_BN_leg_R_03', 'DEF_BN_leg_R_04', 'DEF_BN_leg_R_05', 'DEF_BN_leg_R_05_end', 'DEF_BN_leg_R_06', 'DEF_BN_Leg_R_07', 'DEF_BN_Leg_R_08', 'DEF_BN_Leg_R_09', 'DEF_BN_Leg_R_10', 'DEF_BN_Leg_R_10_end', 'DEF_BN_Bip01_Pelvis', 'DEF_BN_Bip01_Pelvis_end', 'DEF_BN_Arm_L_01', 'DEF_BN_Arm_L_02', 'DEF_BN_Arm_L_03', 'DEF_BN_Arm_L_03_end', 'DEF_BN_Arm_R_01', 'DEF_BN_Arm_R_02', 'DEF_BN_Arm_R_03', 'DEF_BN_Arm_R_03_end']
xia_corps_name = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightFingerBase', 'RightHandIndex1', 'RThumb']
walking_corps_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End']
eight_corps_name = ['hip', 'abdomen', 'chest', 'neck', 'head', 'leftEye', 'rightEye', 'rCollar', 'rShldr', 'rForeArm', 'rHand', 'rThumb1', 'rThumb2', 'rIndex1', 'rIndex2', 'rMid1', 'rMid2', 'rRing1', 'rRing2', 'rPinky1', 'rPinky2', 'lCollar', 'lShldr', 'lForeArm', 'lHand', 'lThumb1', 'lThumb2', 'lIndex1', 'lIndex2', 'lMid1', 'lMid2', 'lRing1', 'lRing2', 'lPinky1', 'lPinky2', 'rButtock', 'rThigh', 'rShin', 'rFoot', 'lButtock', 'lThigh', 'lShin', 'lFoot']


mixamo_contact_name = ['LeftToe_End', 'RightToe_End', 'LeftToeBase', 'RightToeBase']
elephant_contact_name = [name for name in elephant_corps_name if 'Hand' in name or 'Toe' in name or 'Foot' in name]
crab_dance_contact_name = [name for name in crab_dance_corps_name if 'end' in name and ('05' in name or '10' in name or '15' in name)]
xia_contact_name = ['LeftToeBase', 'RightToeBase']
walking_contact_name = copy.deepcopy(mixamo_contact_name)
eight_contact_name = ['lShin', 'lFoot', 'rShin', 'rFoot']


class SkeletonDatabase:
    names = ['Mixamo', 'Elephant', 'Crab_dance', 'Xia', 'Walking', 'Eight']
    corps_names = [mixamo_corps_name, elephant_corps_name, crab_dance_corps_name, xia_corps_name, walking_corps_name, eight_corps_name]
    contact_names = [mixamo_contact_name, elephant_contact_name, crab_dance_contact_name, xia_contact_name, walking_contact_name, eight_contact_name]
    contact_thresholds = [0.018, 0.018, 0.006, 0.018, 0.018, 0.018]

    @classmethod
    def match(cls, joint_names):
        n_match = []
        for idx, class_name in enumerate(cls.names):
            res = 0
            for j in cls.corps_names[idx]:
                if j in joint_names:
                    res += 1
            n_match.append(res)
        max_match = max(n_match)
        max_match_id = n_match.index(max_match)
        return max_match_id, max_match

