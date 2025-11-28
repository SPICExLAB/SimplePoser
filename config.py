r"""
    Config for paths, joint set, and normalizing scales.
"""


# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
amass_data = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 
              'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
              'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SOMA',
              'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

# dataset 
datasets = {
    'AMASS':        {'fps': 60},
    'DIP_IMU':      {'fps': 60},
    'IMUPoser':     {'fps': 30},
    'TotalCapture': {'fps': 60},
}


# device-location combinations
combos = {
    # 'global': [0, 1, 2, 3, 4],
    # 'lw_rp_h': [0, 3, 4],
    # 'rw_rp_h': [1, 3, 4],
    # 'lw_lp_h': [0, 2, 4],
    # 'rw_lp_h': [1, 2, 4],
    # 'lw_h': [0, 4],
    # 'lw': [0],
    # 'lw_lp': [0, 2],
    # 'lw_rp': [0, 3],
    # 'h': [4],
    # 'rw_lp': [1, 2],
    'rw_rp': [1, 3],
    # 'lp_h': [2, 4],
    # 'rp_h': [3, 4],
    # 'lp': [2],
    # 'rp': [3],
}


class paths:
    data_dir = '/data/projects/Pose/dataset_work' # directory of processed datasets

    raw_amass_dir = '/data/projects/Pose/raw/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = '/data/projects/Pose/dataset_work/AMASS'              # output path for the synthetic AMASS dataset

    raw_dipimu_dir = '/data/projects/Pose/raw/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = '/data/projects/Pose/dataset_work/DIP_IMU'     # output path for the preprocessed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP website.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/dataset_raw/TotalCapture/official'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = '/data/projects/Pose/dataset_work/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    raw_imuposer_dir = '/data/projects/Pose/raw/IMUPoser'          # raw IMUPoser dataset path (raw_imuposer_dir/P1/*.pkl)
    imuposer_dir = '/data/projects/Pose/dataset_work/IMUPoser'          # output path for the preprocessed IMUPoser dataset

    example_dir = 'data/example'                    # example IMU measurements
    smpl_file = 'models/SMPL_male.pkl'              # official SMPL model path
    weights_file = 'data/weights.pt'                # network weight file


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(0, 24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_imu =  5 * (3 + 9) # 5 sensors * (3 acc + 9 ori)

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


fps = 30
acc_scale = 30
vel_scale = 2
gravity_velocity = -0.018
