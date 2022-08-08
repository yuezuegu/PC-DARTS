

import time 
import subprocess

from datetime import datetime
import nvsmi

class ProcQueue:
    def __init__(self, gpus, max_procs) -> None:
        self.gpus = gpus
        self.max_procs = max_procs

        self.wait_queue = []
        self.exec_queue = {i: [] for i in gpus}

    def start(self):
        while True:
            for i in self.exec_queue:
                for p in self.exec_queue[i]:
                    if self.is_done(p):
                        self.exec_queue[i].remove(p)

            if self.total_no_running() < self.max_procs:
                self.start_proc()

            if len(self.wait_queue) == 0 and self.total_no_running() == 0:
                print("All processes are completed.")
                exit()

            time.sleep(1.0)


    def push(self, cmd):
        self.wait_queue.append(cmd)

    def start_proc(self):
        if len(self.wait_queue) > 0:
            cmd = self.wait_queue.pop(0)
            #gpu_id = self.least_busy_gpu()
            gpu_id = self.load_balancer()
            
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + cmd
            print("Starting cmd: {}".format(cmd))

            p = subprocess.Popen(cmd, shell=True)
            self.exec_queue[gpu_id].append(p)

    def total_no_running(self):
        return sum([len(self.exec_queue[i]) for i in self.exec_queue])

    def least_busy_gpu(self):
        least_gpu_id = self.gpus[0]
        min_procs = len(self.exec_queue[least_gpu_id])
        for i in self.gpus[1:]:
            if len(self.exec_queue[i]) < min_procs:
                min_procs = len(self.exec_queue[i])
                least_gpu_id = i
        return least_gpu_id

    def load_balancer(self):
        no_procs = {gpu: 0 for gpu in self.gpus}

        for proc in nvsmi.get_gpu_processes():
            gpu_id = int(proc.gpu_id)
            if gpu_id not in self.gpus:
                continue
            no_procs[gpu_id] += 1
        
        least_procs = list(no_procs.values())[0]
        best_gpu = list(no_procs.keys())[0]
        for gpu_id in no_procs:
            if no_procs[gpu_id] < least_procs:
                least_procs = no_procs[gpu_id]
                best_gpu = gpu_id
        return best_gpu

    def is_done(self, proc):
        retcode = proc.poll()
        if retcode is not None: # Process finished.
            print("Process {} ended with code {}".format(proc.pid, retcode))
            if retcode != 0:
                print("FAILED: Return code is not 0")
            return True
        else:
            return False


def date_minute():
	return datetime.now().strftime("%Y_%m_%d-%H_%M")

def date_second():
	return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

def date_millisecond():
	return datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")[:-3] #-3 to convert us to ms

