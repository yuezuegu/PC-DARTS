
from process_queue import ProcQueue

if __name__=="__main__":
	gpus = [2]
	queue = ProcQueue(gpus=gpus, max_procs=4)

	seed = 0

	cmd = f"python train_search.py \
	--lambda_jr=0 \
	--adv_epsilon=0 \
	--seed={seed}"
	queue.push(cmd)

	cmd = f"python train_search.py \
	--lambda_jr=0.01 \
	--adv_epsilon=0 \
	--seed={seed}"
	queue.push(cmd)

	cmd = f"python train_search.py \
	--lambda_jr=0 \
	--adv_epsilon=8 \
	--seed={seed}"
	queue.push(cmd)

	queue.start()