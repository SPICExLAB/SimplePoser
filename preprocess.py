r"""
    Preprocess DIP-IMU, TotalCapture, IMUPoser, and Nymeria datasets.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""


import articulate as art
import torch
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import glob

from config import paths, amass_data


# fix to a stupid problem
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray


# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

body_model = art.ParametricModel(paths.smpl_file)


def _syn_acc(v, fps=60, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * (fps**2) for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * (fps**2) / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def _foot_ground_probs(joint):
    r"""
    Compute foot-ground contact probabilities.
    """
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)        


def process_amass():
    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(p.clone())  # N, 24, 3, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        out_contact.append(_foot_ground_probs(joint).clone())  # N, 2
        b += l

    print('Saving')
    data = {
        'joint': out_joint,
        'pose': out_pose,
        'shape': out_shape,
        'tran': out_tran,
        'acc': out_vacc,
        'ori': out_vrot,
        'contact': out_contact
    }
    torch.save(data, os.path.join(paths.amass_dir, 'AMASS.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)


def process_dipimu(split='train'):
    imu_mask = [7, 8, 9, 10, 0, 2]

    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    test_split = ['s_09', 's_10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans, joints, shapes = [], [], [], [], [], []

    body_model = art.ParametricModel(paths.smpl_file)

    for subject_name in subjects:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            shape = torch.zeros(10)
            tran = torch.zeros(pose.shape[0], 3)
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                # forward kinematics to get the joint position
                pose = art.math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                _, joint, _ = body_model.forward_kinematics(pose, shape, tran, calc_mesh=True)

                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(tran.clone())  # dip-imu does not contain translations
                shapes.append(shape.clone())
                joints.append(joint.clone())
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    print('Saving')
    os.makedirs(paths.dipimu_dir, exist_ok=True)
    data = {
        'pose': poses,
        'tran': trans,
        'acc': accs,
        'ori': oris,
        'joint': joints,
        'shape': shapes,
    }
    torch.save(data, os.path.join(paths.dipimu_dir, f'DIP_{split}.pt'))
    print(f'Preprocessed DIP-IMU {split} dataset is saved at', paths.dipimu_dir)


def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)


def process_imuposer(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    test_split = ['P9', 'P10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans, contacts, joints = [], [], [], [], [], []
    syn_acc, syn_ori = [], []
    for subject_name in sorted(os.listdir(paths.raw_imuposer_dir)):
        if subject_name not in subjects:
            continue

        print(f"Processing: {subject_name}")
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_imuposer_dir, subject_name))):
            with open(os.path.join(paths.raw_imuposer_dir, subject_name, motion_name), "rb") as f: 
                fdata = pickle.load(f)
                
                acc = fdata['imu'][:, :5*3].view(-1, 5, 3)
                ori = fdata['imu'][:, 5*3:].view(-1, 5, 3, 3)
                pose = art.math.axis_angle_to_rotation_matrix(fdata['pose']).view(-1, 24, 3, 3)
                tran = fdata['trans'].to(torch.float32)
                
                 # align IMUPoser global fame with DIP
                rot = torch.tensor([[[-1, 0, 0], [0, 0, 1], [0, 1, 0.]]])
                pose[:, 0] = rot.matmul(pose[:, 0])
                tran = tran.matmul(rot.squeeze())

                # obtain joints positions and foot-ground contact probabilities
                grot, joint, vert = body_model.forward_kinematics(pose, None, tran, calc_mesh=True)
                contact = _foot_ground_probs(joint)

                # ensure sizes are consistent
                assert tran.shape[0] == pose.shape[0]

                accs.append(acc.clone())          # N, 5, 3
                oris.append(ori.clone())          # N, 5, 3, 3
                contacts.append(contact.clone())  # N, 2
                joints.append(joint.clone())      # N, 24, 3
                poses.append(pose.clone())        # N, 24, 3, 3
                trans.append(tran.clone())        # N, 3

                # synthesize IMU accelerations and orientations
                syn_acc.append(_syn_acc(vert[:, vi_mask]).clone()) # N, 6, 3
                syn_ori.append(grot[:, ji_mask].clone())           # N, 6, 3, 3

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans,
        'syn_acc': syn_acc,
        'syn_ori': syn_ori,
        'contact': contacts,
        'joint': joints
    }
    os.makedirs(paths.imuposer_dir, exist_ok=True)
    data_path = os.path.join(paths.imuposer_dir, f'{split}.pt')
    torch.save(data, data_path)
    print(f'Preprocessed IMUPoser {split} dataset is saved at', data_path)



def process_nymeria():
    r"""
    Preprocess Nymeria dataset. Converts Xsens poses to SMPL and synthesizes virtual IMU data.
    """
    XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
                     5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]

    def xsens_to_smpl(xsens_rotmat):
        perm = [1, 2, 0]
        xsens_rotmat = xsens_rotmat[:, :, perm, :]
        xsens_rotmat = xsens_rotmat[:, :, :, perm]
        if not isinstance(xsens_rotmat, torch.Tensor):
            xsens_rotmat = torch.from_numpy(xsens_rotmat).float()
        else:
            xsens_rotmat = xsens_rotmat.float()
        Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        xsens_rotmat = Rz90 @ (Rx90 @ xsens_rotmat)
        smpl_rotmat = torch.eye(3).repeat(xsens_rotmat.shape[0], 24, 1, 1)
        for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
            smpl_rotmat[:, smpl_idx] = xsens_rotmat[:, xsens_idx]
        return smpl_rotmat

    os.makedirs(paths.nymeria_dir, exist_ok=True)
    for pkl_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_nymeria_dir, '*.pkl')))):
        print('\rReading', pkl_fname)
        with open(pkl_fname, 'rb') as f:
            raw = pickle.load(f)

        if 'gt_data' not in raw or 'xsens_pose' not in raw['gt_data']:
            print('\tskipping (no xsens_pose)'); continue

        xsens_pose = raw['gt_data']['xsens_pose']  # (N, 23, 4, 4)
        if isinstance(xsens_pose, np.ndarray):
            xsens_pose = torch.from_numpy(xsens_pose).float()

        N = xsens_pose.shape[0]
        if N <= 12: print('\tdiscard one sequence with length', N); continue

        # extract root translation and joint rotations
        tran = xsens_pose[:, 0, :3, 3].contiguous().clone()
        xsens_rot = xsens_pose[:, :, :3, :3].contiguous().clone()

        # Xsens -> SMPL global rotations -> local rotations via IK
        global_rot = xsens_to_smpl(xsens_rot)
        pose = body_model.inverse_kinematics_R(global_rot).view(N, 24, 3, 3)

        # forward kinematics in chunks to avoid OOM
        shape = torch.zeros(10)
        CHUNK = 1000
        all_grot, all_joint, all_vert = [], [], []
        for i in range(0, N, CHUNK):
            g, j, v = body_model.forward_kinematics(
                pose[i:i+CHUNK], shape, tran[i:i+CHUNK], calc_mesh=True
            )
            all_grot.append(g)
            all_joint.append(j)
            all_vert.append(v[:, vi_mask])
            del g, j, v
        grot = torch.cat(all_grot)
        joint = torch.cat(all_joint)
        vert = torch.cat(all_vert)
        del all_grot, all_joint, all_vert

        # synthesize IMU signals (same as AMASS)
        acc = _syn_acc(vert, fps=50)
        ori = grot[:, ji_mask]

        data = {
            'pose': pose.clone(),                          # N, 24, 3, 3
            'shape': shape.clone(),                        # 10
            'tran': tran.clone(),                          # N, 3
            'joint': joint[:, :24].contiguous().clone(),   # N, 24, 3
            'acc': acc,                                    # N, 6, 3
            'ori': ori,                                    # N, 6, 3, 3
            'contact': _foot_ground_probs(joint).clone(),  # N, 2
        }
        out_name = os.path.splitext(os.path.basename(pkl_fname))[0] + '.pt'
        torch.save(data, os.path.join(paths.nymeria_dir, out_name))
        print(f'\tSaved {out_name} ({N} frames)')

    print('Preprocessed Nymeria dataset is saved at', paths.nymeria_dir)


if __name__ == '__main__':
    # process_amass()
    # process_dipimu(split='train')
    # process_totalcapture()
    # process_imuposer(split='test')
    process_nymeria()