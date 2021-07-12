import datetime
import json
import os
import socket
import time
from typing import List
from typing import Union

import torch
import torch.distributed as tdist
from colorama import Fore
from torch.multiprocessing import set_start_method


class TorchDistManager:
    WORLD_GROUP = tdist.group.WORLD if torch.cuda.is_available() else None

    @staticmethod
    def time_str():
        return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')

    def __init__(self, exp_dirname: str, node0_addr: Union[int, str] = 'auto', node0_port: Union[int, str] = 'auto', mp_start_method: str = 'fork', backend: str = 'nccl'):
        set_start_method(mp_start_method, force=True)
        self.backend = backend
        # if multi_nodes:   # if $2 > ntasks-per-node
        #     os.environ[f'{backend.upper()}_SOCKET_IFNAME'] = 'eth0'
        world_size: int = int(os.environ['SLURM_NTASKS'])
        rank: int = int(os.environ['SLURM_PROCID'])

        temp_f_path = exp_dirname + f'.temp_{os.environ["SLURM_NODELIST"]}.json'
        self.temp_f_path = temp_f_path
        if rank == 0:
            node0_addr = str(node0_addr).lower()
            if node0_addr == 'auto':
                node0_addr = f'{socket.gethostbyname(socket.gethostname())}'

            node0_port = str(node0_port).lower()
            if node0_port == 'auto':
                sock = socket.socket()
                sock.bind(('', 0))
                _, node0_port = sock.getsockname()
                sock.close()

            node0_addr_port = f'tcp://{node0_addr}:{node0_port}'
            with open(temp_f_path, 'w') as fp:
                json.dump(node0_addr_port, fp)
            print(f'{TorchDistManager.time_str()}[rk00] node0_addr_port: {node0_addr_port} (saved at \'{temp_f_path}\')')
        else:
            time.sleep(3 + rank * 0.1)
            while not os.path.exists(temp_f_path):
                print(f'{TorchDistManager.time_str()}[rk{rank:02d}] try to read node0_addr_port')
                time.sleep(0.5)
            with open(temp_f_path, 'r') as fp:
                node0_addr_port = json.load(fp)
            print(f'{TorchDistManager.time_str()}[rk{rank:02d}] node0_addr_port obtained')

        self.backend, self.node0_addr_port, self.world_size, self.rank = backend, node0_addr_port, world_size, rank
        self.node0_addr = self.node0_addr_port[self.node0_addr_port.find('//') + 2:self.node0_addr_port.rfind(':')]
        self.ngpus_per_node, self.dev_idx = torch.cuda.device_count(), ...

    # def initialize(self):
        tdist.init_process_group(
            backend=self.backend, init_method=self.node0_addr_port,
            world_size=self.world_size, rank=self.rank
        )
        self.dev_idx: int = int(os.environ['SLURM_LOCALID'])            # equals to rank % gres_gpu
        torch.cuda.set_device(self.dev_idx)

        print(f'{TorchDistManager.time_str()}[dist init] rank[{self.rank:02d}]: node0_addr_port={self.node0_addr_port}, gres_gpu(ngpus_per_node)={self.ngpus_per_node}, dev_idx={self.dev_idx}')

        assert torch.distributed.is_initialized()

        self.barrier()
        if self.is_master():
            os.remove(self.temp_f_path)
            print(Fore.LIGHTBLUE_EX + f'{TorchDistManager.time_str()}[rk00] removed temp file: \'{self.temp_f_path}\'')

    def finalize(self) -> None:
        print(Fore.CYAN + f'{TorchDistManager.time_str()}[dist finalize] rank[{self.rank:02d}]')

    def is_master(self):
        return self.rank == self.world_size // 2

    def get_world_group(self):
        return TorchDistManager.WORLD_GROUP

    def new_group(self, ranks: List[int]):
        return tdist.new_group(ranks=ranks)

    def barrier(self) -> None:
        tdist.barrier()

    def allreduce(self, t: torch.Tensor, group_idx_or_handler=None) -> None:
        if group_idx_or_handler is None:
            group_idx_or_handler = TorchDistManager.WORLD_GROUP
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.all_reduce(cu, group=group_idx_or_handler)
            t.copy_(cu.cpu())
        else:
            tdist.all_reduce(t, group=group_idx_or_handler)

    def allgather(self, t: torch.Tensor, cat=False, group_idx_or_handler=None, group_size=None) -> Union[List[torch.Tensor], torch.Tensor]:
        assert (group_idx_or_handler is None) == (group_size is None)
        
        if group_idx_or_handler is None:
            group_idx_or_handler = TorchDistManager.WORLD_GROUP
            group_size = self.world_size
        if not t.is_cuda:
            t = t.cuda()
        
        ls = [torch.empty_like(t) for _ in range(group_size)]
        tdist.all_gather(ls, t, group=group_idx_or_handler)
        if cat:
            ls = torch.cat(ls, dim=0)
        return ls

    def broadcast(self, t: torch.Tensor, rank_in_the_group: int, group_idx_or_handler=None) -> None:
        if group_idx_or_handler is None:
            group_idx_or_handler = TorchDistManager.WORLD_GROUP
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.broadcast(cu, src=rank_in_the_group, group=group_idx_or_handler)
            t.copy_(cu.cpu())
        else:
            tdist.broadcast(t, src=rank_in_the_group, group=group_idx_or_handler)

    def dist_fmt_vals(self, val, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
        ts = torch.zeros(self.world_size)
        ts[self.rank] = val
        self.allreduce(ts)
        if fmt is None:
            return ts
        return [fmt % v for v in ts.numpy().tolist()]
