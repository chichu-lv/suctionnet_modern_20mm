import json
import platform
import subprocess
from pathlib import Path
import torch
import spconv
import suction20
from models.instance_route_a import InstanceSuction3D
from suction20.instance_data import InstanceAttemptDataset
root = Path(__file__).resolve().parent
case = InstanceAttemptDataset(root.parent/'attempts_rgbd.jsonl', 'train')[0]
backbone = root/'upstream'/'graspness_modern'
model = InstanceSuction3D(backbone_root=backbone).cuda().train()
result = model(case['observation'],case['attempts']['point_cam_m'],case['attempts']['normal_out_cam'])
valid = result['valid']
assert valid.any()
loss = torch.nn.functional.binary_cross_entropy_with_logits(result['task_logit'][valid].reshape(-1), torch.as_tensor(case['attempts']['success'], device='cuda').float().reshape(-1)[valid])
loss.backward()
grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
torch.cuda.synchronize()
print(json.dumps(dict(role='train-only target example; untrained A forward/backward probe',commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(), module=suction20.__file__, python=platform.python_version(), torch=torch.__version__,spconv=spconv.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),allocated_peak=torch.cuda.max_memory_allocated(),observation_id=case['attempts']['attempt_ids'],support=len(case['observation']['support_points']), backbone=str(backbone), gradient_tensors=len(grads), loss=loss.item())))
