import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    '''if accu['t1'] > target_accu and accu['fdg'] > target_accu:
        log('\tboth above {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}t1_{0:.4f}fdg_dict.pth').format(accu['t1'], accu['fdg'])))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}t1_{0:.4f}fdg.pth').format(accu['t1'], accu['fdg'])))
    elif accu['t1'] > target_accu:
        log('\tt1 above {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}t1_dict.pth').format(accu['t1'])))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}_t1.pth').format(accu['t1'])))
    elif accu['fdg'] > target_accu:
        log('\tfdg above {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}_fdg_dict.pth').format(accu['fdg'])))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}_fdg.pth').format(accu['fdg'])))'''
    if accu['sum'] > target_accu:
        log('\multimodal above {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}_dict.pth').format(accu['sum'])))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu['sum'])))