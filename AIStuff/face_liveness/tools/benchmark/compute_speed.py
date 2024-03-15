"""
compute speed of each leaf module
"""
# import time
import logging
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_speed(model, input_size, device, iteration):
#     device = torch.device("cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model = model.module.to(device)
    model.eval()

    x = torch.randn(*input_size, device=device)
    x.to(device)

    for _ in range(10):
        model(x.float())

    logger.info('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    # t_start = time.time()
    for _ in range(iteration):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    # elapsed_time = time.time() - t_start
    # logger.info(
    #     f'Elapsed time: [{elapsed_time} s / {iteration} iter]')
    # logger.info(f'Speed Time: {elapsed_time / iteration * 1000} ms / iter    FPS: {iteration / elapsed_time}')
