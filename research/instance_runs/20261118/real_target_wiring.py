"""Verify B on one real labeled target attempt using ephemeral initial weights."""
import json
from pathlib import Path
import torch
from suction20.instance_data import InstanceAttemptDataset
from suction20.instance_training import attempt_loss, create_instance_model, load_instance_config, seed_all

base = Path("/home/ubuntu/work/instance-routes-20261118")
config = load_instance_config(base/"B/repo/configs/instance_b.yaml")
seed_all(config["seed"])
model = create_instance_model(config, "cuda")
data = InstanceAttemptDataset(base/"attempts_rgbd.jsonl", "train", cache_size=0)
for i in range(min(16, len(data))):
    item = data[i]
    loss = attempt_loss(model, item)
    if loss is None:
        continue
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.norm() for p in grads]))
    assert norm.isfinite() and norm > 0
    print(json.dumps(dict(index=i, attempt_ids=item['attempts']['attempt_ids'],
                       support=len(item['observation']['support_points']),
                       gradient_norm=norm.item(), loss=loss.item(),
                       cuda_model=torch.cuda.get_device_name(0), phase='partial_wiring',
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())))
    break
else:
    raise RuntimeError('No supported attempt among first sixteen samples')
