def adjust_learning_rate(optimizer, iteration_count, lr_decay):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count, lr):
    """Imitating the original implementation"""
    lr = lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr