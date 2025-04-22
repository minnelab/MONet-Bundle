
import os

def subfiles(directory, prefix=None, suffix=None, join=True, sort=True):
    """
    List files in a directory with optional filtering by prefix and/or suffix.
    
    Parameters
    ----------
    directory : str
        The path to the directory to list files from.
    prefix : str, optional
        If specified, only files starting with this prefix will be included.
    suffix : str, optional
        If specified, only files ending with this suffix will be included.
    join : bool, optional
        If True, the directory path will be joined with the filenames. Default is True.
    sort : bool, optional
        If True, the list of files will be sorted. Default is True.
    
    Returns
    -------
    list of str
        A list of filenames (with full paths if `join` is True) that match the specified criteria.
    """

    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]
    if suffix is not None:
        files = [f for f in files if f.endswith(suffix)]
    if join:
        files = [os.path.join(directory, f) for f in files]
    if sort:
        files.sort()
    return files

def prepare_nnunet_batch(batch, device, non_blocking):
    """
    Prepares a batch of data and targets for nnU-Net training by transferring them to the specified device.

    Parameters
    ----------
    batch : dict
        A dictionary containing the data and target tensors. The key "data" corresponds to the input data tensor,
        and the key "target" corresponds to the target tensor or a list of target tensors.
    device : torch.device
        The device to which the data and target tensors should be transferred (e.g., 'cuda' or 'cpu').
    non_blocking : bool
        If True, allows non-blocking data transfer to the device.

    Returns
    -------
    tuple
        A tuple containing the data tensor and the target tensor(s) after being transferred to the specified device.
    """
    data = batch["data"].to(device, non_blocking=non_blocking)
    if isinstance(batch["target"], list):
        target = [i.to(device, non_blocking=non_blocking) for i in batch["target"]]
    else:
        target = batch["target"].to(device, non_blocking=non_blocking)
    return data, target

def get_checkpoint(epoch, ckpt_dir):
    """
    Retrieves the checkpoint for a given epoch from the checkpoint directory.

    Parameters
    ----------
    epoch : int or str
        The epoch number to retrieve. If 'latest', the function will return the latest checkpoint.
    ckpt_dir : str
        The directory where checkpoints are stored.

    Returns
    -------
    int
        The epoch number of the checkpoint to be retrieved. If 'latest', returns the latest epoch number.
    """
    if epoch == "latest":

        latest_checkpoints = subfiles(ckpt_dir, prefix="checkpoint_epoch", sort=True,
                                      join=False)
        epochs = []
        for latest_checkpoint in latest_checkpoints:
            epochs.append(int(latest_checkpoint[len("checkpoint_epoch="):-len(".pt")]))

        epochs.sort()
        latest_epoch = epochs[-1]
        return latest_epoch
    else:
        return epoch

def reload_checkpoint(trainer, epoch, num_train_batches_per_epoch, ckpt_dir, lr_scheduler=None):
    """
    Reloads the checkpoint for a given epoch and updates the trainer's state.

    Parameters
    ----------
    trainer : object
        The trainer object whose state needs to be updated.
    epoch : int
        The epoch number to load the checkpoint from.
    num_train_batches_per_epoch : int
        The number of training batches per epoch.
    ckpt_dir : str
        The directory where the checkpoints are stored.
    lr_scheduler : object, optional
        The learning rate scheduler to be updated (default is None).

    Returns
    -------
    None
    """

    epoch_to_load = get_checkpoint(epoch, ckpt_dir)
    trainer.state.epoch = epoch_to_load
    trainer.state.iteration = (epoch_to_load* num_train_batches_per_epoch) +1
    if lr_scheduler is not None:
        lr_scheduler.ctr = epoch_to_load
        lr_scheduler.step(epoch_to_load)
