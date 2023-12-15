# -*-coding:utf-8 -*-
"""
This script is used to execute board-side inference.
Copyright Shenshu Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import argparse
import configparser
import time
import os
import sys
import stat
import socket
import shutil
import gc
import logging
from pathlib import Path
import cv2
import numpy as np

LOGGER = logging.getLogger(__file__)


def os_command_inject_check(cmd: str) -> tuple:
    inject_character_list = ['|', '&', '$', '>', '<', '`', '\\', '!', '\n']
    for ch in inject_character_list:
        if ch in cmd:
            return False, ch
    return True, cmd


def bgr2yuv420sp(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = yuv.shape[0:2]
    h_plane = h // 3
    yuv_sp = yuv.copy()
    uv_sp = yuv_sp[-h_plane:].reshape(2, -1).transpose(1, 0).reshape(h_plane, -1)
    yuv_sp[-h_plane:] = uv_sp
    return yuv_sp


def rgb_chw2yuv420sp(rgb_chw):
    rgb_chw_np = rgb_chw.cpu().numpy()
    c, h, w = rgb_chw_np.shape
    rgb = rgb_chw_np.reshape(c, h, w).transpose(1, 2, 0)
    bgr = rgb[:, :, ::-1]
    return bgr2yuv420sp(bgr)


class SSHBoard:

    def __init__(self, host_ip, username, pwd, port):
        try:
            import paramiko
        except ImportError as ex:
            raise RuntimeError("Please pip install paramiko.") from ex

        self.paramiko = paramiko
        self.ssh = self.paramiko.SSHClient()
        self.host_ip = host_ip
        self.username = username
        self.pwd = pwd
        self.port = port
        self.transport = self.paramiko.Transport(sock=(self.host_ip, self.port))

    @staticmethod
    def adapter_board_address(*paths):
        return os.path.join(*paths).replace('\\', '/')

    @staticmethod
    def parse_host_mount_path(ssh_cfg_path: str, ssh_conf: configparser.ConfigParser) -> str:
        host_mount_path = ssh_conf.get('host_mount_path', '').replace('\\', '/').rstrip('/')
        if not host_mount_path or not os.path.isdir(host_mount_path):
            raise RuntimeError(r'Invalid "HOST_MOUNT_PATH" configured in {0}.'.format(ssh_cfg_path))
        return host_mount_path

    @staticmethod
    def parse_board_mount_path(ssh_cfg_path: str, ssh_conf: configparser.ConfigParser) -> str:
        board_mount_path = SSHBoard.adapter_board_address(ssh_conf.get('board_mount_path', '')).rstrip('/')
        if not board_mount_path:
            raise RuntimeError(r'Invalid "BOARD_MOUNT_PATH" configured in {0}.'.format(ssh_cfg_path))
        return board_mount_path

    def login_host(self):
        self.ssh.set_missing_host_key_policy(self.paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(self.host_ip, self.port, self.username, self.pwd, timeout=30)
        except Exception as err:
            raise RuntimeError('SSH connect error, please check whether ssh configuration is correct.') from err
        self.transport.connect(username=self.username, password=self.pwd)
        # Explicitly clear after use
        del self.pwd
        gc.collect()

    def execute_command(self, command):
        cmd_valid, command = os_command_inject_check(command)
        if not cmd_valid:
            RuntimeError('Invalid command input.')
        stdin, stdout, stderr = self.ssh.exec_command(command)
        command_result = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        return command_result, err

    def logout_host(self):
        self.ssh.close()
        self.transport.close()

    def mkdir(self, path):
        sftp = self.paramiko.SFTPClient.from_transport(self.transport)
        dir_path = str()
        for dir_folder in path.split("/"):
            if dir_folder == "":
                continue
            dir_path += r"/{0}".format(dir_folder)
            try:
                sftp.listdir(dir_path)
            except IOError:
                sftp.mkdir(dir_path)

    def mount(self, mnt_cmd):
        """
        Mounting board side path, tmp folder is created by default
        """
        # mnt_result tuple (stdout, stderr)
        mnt_result = self.execute_command(mnt_cmd)
        if not mnt_result[1]:
            # stderr is empty means mount success
            return True, mnt_result[0]
        else:
            return False, mnt_result[1]

    def umount(self, mnt_dir):
        """
        Umounting board side file path
        """
        umount_cmd = 'umount ' + mnt_dir
        # umount_result tuple (stdout, stderr)
        umount_result = self.execute_command(umount_cmd)
        if not umount_result[1]:
            # stderr is empty means unmount success
            return True
        else:
            LOGGER.info(f'[Umount] umount {mnt_dir} failed: {umount_result[1]}')
            return False

    def invoke_shell_execute_main(self, execute_file_path, args, is_stdout=True):
        """
        Execute the board operation and execute the corresponding main method
        """
        # set the maximum width (in characters) of the terminal window
        conn = self.ssh.invoke_shell(width=999)
        # set the timeout on blocking read/write operations.
        conn.settimeout(3000)
        conn.send(f"cd {os.path.dirname(execute_file_path)}\r")
        while not conn.recv_ready():
            # delay 2 seconds
            time.sleep(2)
        buff_first = ''
        while not buff_first.endswith('# '):
            # set the maximum number of bytes to read.
            resp = conn.recv(9999)
            buff_first += resp.decode('utf-8')
        if is_stdout:
            print(buff_first)
        conn.send(f"./{os.path.basename(execute_file_path)} {args}\r")
        while not conn.recv_ready():
            # delay 2 seconds
            time.sleep(2)
        buff_second = ''
        while not buff_second.endswith('# '):
            # set the maximum number of bytes to read.
            resp = conn.recv(9999)
            buff_second = resp.decode('utf-8')
            if is_stdout:
                print(resp.decode('utf-8'), end="")
        if is_stdout:
            print(os.linesep)
        conn.close()

    def delete(self, path):
        """
        Delete all content in the corresponding folder
        """
        sftp = self.paramiko.SFTPClient.from_transport(self.transport)
        files = sftp.listdir_attr(path=path)
        for file in files:
            if stat.S_ISDIR(file.st_mode):
                self.delete(self.adapter_board_address(path, file.filename))
            else:
                sftp.remove(self.adapter_board_address(path, file.filename))
        sftp.rmdir(path)

    def get_output_from_path(self, path):
        """
        Find the file name corresponding to the existing job name
        Use “/” because the board side is linux system
        """
        sftp = self.paramiko.SFTPClient.from_transport(self.transport)
        files_attr = sftp.listdir_attr(path)
        for file_attr in files_attr:
            mode = file_attr.st_mode
            if stat.S_ISDIR(mode):
                return file_attr.filename
        return ''


class BoardInfer(object):

    def __init__(self: any, sys_argv: any) -> None:
        parser = argparse.ArgumentParser(description='This script is used to execute board-side invocation '
                                         'using the specified Ascend App path.',
                                         allow_abbrev=False)
        self.construct_args_parser(parser)
        if not sys_argv:
            parser.print_help()
            return
        args = parser.parse_args(sys_argv)
        self.ssh_cfg_path = args.ssh_config
        self.ssh_conf = self.parse_ssh_conf(args.ssh_config)
        self.host_mount_path = SSHBoard.parse_host_mount_path(self.ssh_cfg_path, self.ssh_conf)
        self.board_mount_path = SSHBoard.parse_board_mount_path(self.ssh_cfg_path, self.ssh_conf)
        self.project_path = str(Path(self.host_mount_path) / args.project_name)
        self.work_dir = self.project_path
        self.exchange_data_dir = Path(self.work_dir) / "exchange_data"
        self.exchange_data_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_data_dir = str(self.exchange_data_dir)
        self.exe_loc = None
        self.args = args
        self.output_path = None
        self.stop_signal = args.stop_signal

        # initialize the ssh client
        ip = self.ssh_conf.get('board_ip', '')
        username = self.ssh_conf.get('user', '')
        user_pd = self.ssh_conf.get('password', '')
        port = int(self.ssh_conf.get('port', 22))

        # Step 0. Initialize the SSH client
        self.ssh = SSHBoard(host_ip=ip, username=username, pwd=user_pd, port=port)
        self.ssh.login_host()

    @staticmethod
    def construct_args_parser(parser: argparse.ArgumentParser) -> None:
        """
        construct board inference argument parser
        Args:
            parser: the initial parser
        Returns:
            None
        """
        parser.add_argument('-s',
                            '--ssh_config',
                            dest='ssh_config',
                            default='',
                            type=str,
                            required=False,
                            help='Specify a separate ssh and mount configuration of the board, '
                            'use the CFG configuration file format.')
        parser.add_argument("--om",
                            default="/home/liuyang/haisi/ai-sd3403/models/yolov8n_640x640_2_mix_original.om",
                            help="path of om model")
        parser.add_argument("--project_name", default="test_ssh", help="work space")
        parser.add_argument("--stop_signal", default="stop", type=str, help="stop_signal filename")
        parser.add_argument("--exe_path",
                            default="/home/liuyang/haisi/ai-sd3403/mpp_inference/out/svp_pingpong_rpc",
                            type=str)

    @staticmethod
    def parse_ssh_conf(ssh_conf_file: str) -> dict:
        if os.path.exists(ssh_conf_file) and os.path.isfile(ssh_conf_file):
            ssh_conf = configparser.ConfigParser()
            ssh_conf.read(ssh_conf_file)
            ssh_conf = dict(ssh_conf['ssh_config'])
        else:
            raise RuntimeError('Invalid ssh configuration file: {0}, not an existing file.'.format(ssh_conf_file))
        return ssh_conf

    @staticmethod
    def _host_path_transfer(src: str, host_mount_path: str, board_mount_path: str) -> str:
        """
        convert the source host path to the mount path on the board.
        Args:
            src: source path
            host_mount_path: the host mount path
            board_mount_path: the board mount path
        Returns:
            a path under the board mount path
        """
        board_src_loc = os.path.realpath(src).replace(os.path.realpath(host_mount_path),
                                                      os.path.realpath(board_mount_path))
        LOGGER.debug(r'[Path convert] {0} ------> {1}'.format(src, board_src_loc))
        return board_src_loc

    @staticmethod
    def _gen_mount_commands(host_mount_path: str, board_mount_path: str) -> str:
        return r'mount -t nfs -o rsize=32768,wsize=32768 -o nolock -o tcp ' \
               r'{HOST_IP}:{HOST_MOUNT_PATH} {BOARD_MOUNT_PATH}' \
            .format(HOST_IP=BoardInfer._get_host_ip(), HOST_MOUNT_PATH=host_mount_path,
                    BOARD_MOUNT_PATH=board_mount_path)

    @staticmethod
    def _check_mount_status(ssh: SSHBoard, host_mount_path: str, board_mount_path: str) -> bool:
        """
        check whether the device has been mounted.
        Args:
            ssh: the ssh connection
            host_mount_path: host mount path
            board_mount_path: board mount path
        Returns:
            True if mount successful
        """
        all_mount = ssh.execute_command('df -h')[0].split('\n')
        device = r'{HOST_IP}:{HOST_MOUNT_PATH}' \
            .format(HOST_IP=BoardInfer._get_host_ip(), HOST_MOUNT_PATH=host_mount_path)
        for index, host in enumerate(all_mount):
            if device == ssh.adapter_board_address(host).rstrip('/'):
                if index + 1 < len(all_mount) \
                        and all_mount[index + 1].find(ssh.adapter_board_address(board_mount_path)) != -1:
                    return True
        return False

    @staticmethod
    def _get_host_ip():
        return socket.gethostbyname(socket.gethostname())
        # return "192.168.0.207"

    def run(self: any, is_stdout: bool = True, output_path: str = '') -> str:
        """
        board-side execution
        Args:
            output_path: oneclick output path
            is_stdout: whether print standard output to console
        Returns:
            result of dump or profiling if enable
        """

        self._prepare_paths(output_path)

        # initialize the ssh client
        ip = self.ssh_conf.get('board_ip', '')
        username = self.ssh_conf.get('user', '')
        user_pd = self.ssh_conf.get('password', '')
        port = int(self.ssh_conf.get('port', 22))

        # Step 0. Initialize the SSH client
        ssh = SSHBoard(host_ip=ip, username=username, pwd=user_pd, port=port)
        ssh.login_host()
        del user_pd
        del self.ssh_conf

        # Step 1. mount
        board_mount_path, host_mount_path = self._do_mount(ssh)

        # Step2. invoke the executable program on the board
        board_run_args = self._gen_board_run_args()
        ssh.execute_command(f"chmod -R u+x {ssh.adapter_board_address(self.exe_loc)}")
        ssh.invoke_shell_execute_main(self.exe_loc, board_run_args, is_stdout)

        # ssh.delete(board_tmp_output)

        # Step5. umount and login out
        ssh.umount(board_mount_path)
        if not self._check_mount_status(ssh, host_mount_path, board_mount_path):
            LOGGER.info(f'[Umount] umount {board_mount_path} successfully.')

        ssh.logout_host()
        return ""

    def change_file_mode(self, file_path, mode="666"):
        file_path = self._host_path_transfer(file_path, self.host_mount_path, self.board_mount_path)
        self.ssh.execute_command(f"chmod {mode} {file_path}")

    def rm_file(self, file_path):
        file_path = self._host_path_transfer(file_path, self.host_mount_path, self.board_mount_path)
        self.ssh.execute_command(f"rm {file_path}")

    def stop_board_process(self):
        self.ssh.execute_command(f"touch {str(Path(self.exchange_data_dir) / self.stop_signal)}")

    def _path_transfer(self: any) -> None:
        self.exe_loc = self._host_path_transfer(self.exe_loc, self.host_mount_path, self.board_mount_path)
        self.om = self._host_path_transfer(self.om, self.host_mount_path, self.board_mount_path)
        self.exchange_data_dir = self._host_path_transfer(self.exchange_data_dir, self.host_mount_path,
                                                          self.board_mount_path)

    def _gen_board_run_args(self: any) -> str:
        cmd_npu = f"{self.om} {self.exchange_data_dir} {self.stop_signal}"
        return cmd_npu

    def _do_mount(self: any, ssh: SSHBoard) -> tuple:
        # error message not empty
        if len(ssh.execute_command(r'cd {0}'.format(self.board_mount_path))[1]) != 0:
            raise RuntimeError('Invalid board mount path, '
                               f'please check configured in {self.ssh_cfg_path}.')
        mount_cmd = self._gen_mount_commands(self.host_mount_path, self.board_mount_path)
        mount_ret, msg = ssh.mount(mount_cmd)
        if not mount_ret and not self._check_mount_status(ssh, self.host_mount_path, self.board_mount_path):
            raise RuntimeError(r'[Mount] Mount failed: {0}. execute cmd: {1}'.format(msg, mount_cmd))
        else:
            LOGGER.info(r'[Mount] Mount success by cmd: {0}'.format(mount_cmd))
        return self.board_mount_path, os.path.realpath(self.host_mount_path)

    def _prepare_paths(self, output_path: str) -> None:
        """
        prepare paths before run board inference
        Args:
            output_path: oneclick output path
        Returns:
            result of dump or profiling if enable
        """
        self.output_path = output_path
        # pre-compiled binary
        self.exe_loc = self.copy_file_to_dir(self.args.exe_path, os.path.join(self.project_path, 'bin'))
        # copy executable to work directory.
        self.om = self.copy_file_to_dir(self.args.om, str(Path(self.project_path) / "model"))

        # check whether the working directory is valid.
        if self.project_path.find(self.host_mount_path) != 0:
            raise RuntimeError(r'Invalid work directory: {0}, '
                               r'The work directory is required must be under the HOST_MOUNT_PATH for run the NPU.'
                               r'configured in {1}.'.format(self.project_path, self.ssh_cfg_path))
        self._path_transfer()

    def copy_file_to_dir(self, file_path: str, target_dir: str) -> str:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        file_path_p = Path(file_path)
        if not file_path_p.is_file():
            raise RuntimeError(f"{file_path} is not a file!")
        shutil.copy(file_path, target_path)
        return str(target_path / file_path_p.name)


def dfl(box_p, box_first=False):
    '''
    box_first: box_p's order is bx4xregxa, else bxregx4xa. box_first need transpose in yolov8's dfl.
    default is False
    '''
    b, c, a = box_p.shape  # batch, channel, anchors
    reg_max = int(c / 4)
    if box_first:
        box_p = box_p.reshape(b, 4, reg_max, a)
        # (1, 4, 16, 8400)
        box_exp = np.exp(box_p)
        # (1, 4, 1, 8400)
        box_sum = np.sum(box_exp, 2, keepdims=True)
        box_p = box_exp / box_sum

        local_encode = np.arange(reg_max).reshape((-1, 1))
        box_p = box_p.transpose((0, 1, 3, 2)).reshape((-1, reg_max))
        box_p = box_p.dot(local_encode)
    else:
        box_p = box_p.reshape(b, reg_max, 4, a)
        # (1, 16, 4, 8400)
        box_exp = np.exp(box_p)
        # (1, 1, 4, 8400)
        box_sum = np.sum(box_exp, 1, keepdims=True)
        box_p = box_exp / box_sum

        local_encode = np.arange(reg_max).reshape((-1, 1))
        box_p = box_p.transpose((0, 2, 3, 1)).reshape((-1, reg_max))
        box_p = box_p.dot(local_encode)
    return box_p.reshape((b, 4, a))

def meshgrid(i_s, j_s):
    '''
    i_s: np.array
    j_s: np.array
    '''
    rows = len(i_s)
    cols = len(j_s)
    i_grid = np.empty((rows, cols), dtype=type(i_s[0]))
    j_grid = i_grid.copy()
    for i_n, i in enumerate(i_s):
        for j_n, j in enumerate(j_s):
            i_grid[i_n, j_n] = i
            j_grid[i_n, j_n] = j
    return i_grid, j_grid


def make_anchors_numpy(hs, ws, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    for h, w, stride in zip(hs, ws, strides):
        sx = np.arange(w) + grid_cell_offset
        sy = np.arange(h) + grid_cell_offset
        sy, sx = meshgrid(sy, sx)
        anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(np.ones((h * w, 1)) * stride)
    anchors = np.concatenate(anchor_points).T
    return anchors[np.newaxis, ...], np.concatenate(stride_tensor).T


def dist2bbox_numpy(box_p, anchors, xywh=False):
    '''
    box_p: shape = [batch, 4, 8400]
    anchors: shape = [1, 2, 8400]
    '''
    lt, rb = box_p[:, 0:2], box_p[:, 2:]

    x1y1 = anchors - lt
    x2y2 = anchors + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), 1)
    return np.concatenate((x1y1, x2y2), 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_on_2output_numpy(preds):
    if preds[0].shape[1] == 4:
        bbox = preds[0]
        cls = preds[1]
    else:
        bbox = preds[1]
        cls = preds[0]
    # cls: (1, 80, 8400)
    cls = sigmoid(cls)
    # y: (1, 84, 8400)
    y = np.concatenate((bbox, cls), 1)
    return y


def forward_on_3output_numpy(preds, batch, no, reg_max, anchors, strides_sq, xywh=False, box_first=False):
    # x_cat shape: (1, 144, 8400)
    x_cat = np.concatenate([preds_i.reshape(batch, no, -1) for preds_i in preds], 2)
    box, cls = x_cat[:, 0:reg_max * 4], x_cat[:, reg_max * 4:]
    box = dfl(box, box_first=box_first)
    # dbox:(1, 4, 8400)
    dbox = dist2bbox_numpy(box, anchors, xywh=xywh) * strides_sq
    # cls: (1, 80, 8400)
    cls = sigmoid(cls)
    # y: (1, 84, 8400)
    y = np.concatenate((dbox, cls), 1)
    return y


if __name__ == '__main__':
    # board = BoardInfer(sys.argv[1:])
    board = BoardInfer(["-s", "../ssh.cfg"])
    board.run()
