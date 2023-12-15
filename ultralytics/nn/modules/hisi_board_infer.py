from pathlib import Path
import numpy as np

from ultralytics.utils.torch_utils import model_info
from .hisi_utils import BoardInfer, bgr2yuv420sp, make_anchors_numpy, rgb_chw2yuv420sp, forward_on_2output_numpy, forward_on_3output_numpy
from multiprocessing import Process
import time
import json
import yaml


default_ssh_cfg = "/mnt/root2204/home/liuyang/Documents/YOLO/ultralytics-compare/ssh.cfg"
default_om_exe = "/home/liuyang/haisi/ai-sd3403/mpp_inference/out/svp_pingpong_rpc"
dtype_map = {"S8": np.int8, "F16": np.float16}

class hisi_board_infer(object):

    def __init__(self, om: str):
        self.om = om
        self.output_id = 0
        self.input_id = 0
        self.exchange_data_dir = Path("")
        self.board_started = False
        self.get_stop_signal = False

    def input_one_image(self, img: np.ndarray):
        if self.board_started:
            # rgb_chw2yuv
            yuv = rgb_chw2yuv420sp(img)
            # exchange to board by nfs
            yuv.tofile(str(self.exchange_data_dir / f"img_{self.input_id}.bin"))
            # ready signal
            f = open(str(self.exchange_data_dir / f"img_{self.input_id}.bin.ready"), "w")
            f.close()
            self.input_id += 1
        else:
            print("input: board process has not started")

    def get_one_output(self):
        if self.board_started:
            while True:
                if self.get_stop_signal:
                    break
                ready_signals = sorted(self.exchange_data_dir.glob(f"output_{self.output_id}*.ready"))
                if len(ready_signals) > 0:
                    break
                time.sleep(0.003)
            output_file_prefix = f"output_{self.output_id}_"
            output_names = sorted(self.exchange_data_dir.glob(f"{output_file_prefix}*.bin"))
            if len(output_names) > 0:
                outputs = self.parse_output_bin(output_names, output_file_prefix)
                self.output_id += 1
                return outputs
            elif self.get_stop_signal:
                return []
            else:
                raise RuntimeError("get empty output")
        else:
            print("output: board process has not started")

    def parse_output_bin(self, bin_files:list, output_file_prefix):
        preds = [None] * self.output_num
        for file_path in bin_files:
            file_stem = file_path.stem
            output_node = file_stem[len(output_file_prefix):]
            node_index = self.output_nodes.index(output_node)
            with open(str(file_path), 'rb') as f:
                data = f.read()
                pred = np.ndarray(self.model_info[output_node]['dims'], dtype=self.model_info[output_node]['dtype'], buffer=data)
                # rescale
                preds[node_index] = (pred - self.model_info[output_node]['offset']) / self.model_info[output_node]['scale']

        y = None
        if self.output_num == 1:
            y = preds[0]
        elif self.output_num == 2:
            y = forward_on_2output_numpy(preds)
        else:
            y = forward_on_3output_numpy(preds, self.model_info['batch'], self.no, self.reg_max, self.anchors, self.strides_sq, xywh=True)
        return y

    def get_outputs(self, batch_num):
        output_batch = []
        if batch_num == 1:
            return self.get_one_output()
        else:
            for _ in range(batch_num):
                output_batch.append(self.get_one_output())
            output_batch = [np.concatenate(input_i, axis=0) for input_i in zip(*output_batch)]
        return output_batch

    def get_model_info(self):
        '''
        1. get scales and offsets from quant_file in weight dir,
        2. get om_model info from model_info_file in exchange dir.
        '''
        weight_dir = Path(self.om).parent
        quant_file = weight_dir / "om_config" / f"{Path(self.om).stem}.json"
        scale_offsets = json.load(open(quant_file, "r"))

        om_info_file = self.exchange_data_dir / Path(self.om).stem
        om_model_info = yaml.safe_load(open(om_info_file, "r"))

        info_needed = {}
        info_needed['batch'] = om_model_info['batch']
        info_needed['cl_num'] = om_model_info['cl_num']  # only used with output_num == 3
        output_num = 0
        for out_name, dims in zip(om_model_info['v_output_names'],
                                  om_model_info['v_output_dims']):
            output_num += 1
            info_needed[out_name] = {}
            info_needed[out_name]['dims'] = dims
            scale_off = scale_offsets[out_name]
            if len(scale_off) == 0:
                info_needed[out_name]['scale'] = 1.0
                info_needed[out_name]['offset'] = 0.0
                info_needed[out_name]['dtype'] = np.float16
            else:
                info_needed[out_name]['scale'] = scale_off[0]['scale']
                info_needed[out_name]['offset'] = scale_off[0]['offset']
                info_needed[out_name]['dtype'] = dtype_map[scale_off[0]
                                                           ['data_type']]
        self.output_num = output_num
        self.output_nodes = om_model_info['v_output_names']
        self.model_info = info_needed
        self.reg_max = om_model_info['reg_max']
        self.no = info_needed['cl_num'] + self.reg_max * 4
        self.anchors, self.strides_sq = None, None
        if len(om_model_info['strides']) > 0:
            # hs = sorted([om_model_info['h'] // s_i for s_i in om_model_info['strides']], reverse=True)
            # ws = sorted([om_model_info['w'] // s_i for s_i in om_model_info['strides']], reverse=True)
            # strides = sorted(om_model_info['strides'])
            hs = [om_model_info['h'] // s_i for s_i in om_model_info['strides']]
            ws = [om_model_info['w'] // s_i for s_i in om_model_info['strides']]
            strides = om_model_info['strides']
            self.anchors, self.strides_sq = make_anchors_numpy(hs, ws, strides)


    def start_board(self, ssh_cfg: str = default_ssh_cfg, om_exe: str = default_om_exe, std_out=False):
        self.board = BoardInfer(["-s", ssh_cfg, "--om", self.om, "--exe_path", om_exe])
        self.exchange_data_dir = Path(self.board.exchange_data_dir)
        # cleaned signal
        cleaned_signal_path = self.exchange_data_dir / "cleaned"
        if cleaned_signal_path.is_file():
            self.board.rm_file(str(cleaned_signal_path))

        self.board_process = Process(target=self.board.run, args=(std_out, ))
        self.board_process.start()

        # wait for starting
        print("waiting for board process starting")
        while True:
            if cleaned_signal_path.is_file():
                break
            else:
                time.sleep(0.5)
        print("board process started!")

        # get om model info file
        model_info_signal_path = self.exchange_data_dir / "modelinfo.ready"
        while True:
            if model_info_signal_path.is_file():
                break
            else:
                time.sleep(0.1)
        self.get_model_info()

        self.board_started = True

    def stop_board(self):
        f = open(str(self.exchange_data_dir / self.board.stop_signal), "w")
        f.close()
        self.board_process.join()
