import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import dgcn
import sklearn
import copy
from tqdm import tqdm
import pathlib
import os
from importlib.machinery import SourceFileLoader
import time
# curr_path = pathlib.Path().resolve()
# curr_path=
# sys_path = os.path.join(os.path.dirname(curr_path), "sysconf.py")
sys_path="/home/n/nguyenpk/clone_gnn/GNN_ERC/sysconf.py"
conf = (SourceFileLoader("sysconf", sys_path).load_module()).conf

# torch.backends.cudnn.enabled=False # my setting having error
# ----------------------------------------
log = dgcn.utils.get_logger()
def evaluation(model, dataset,  device='cuda:0'):
    model.eval()
    y_true = []
    y_pred = []
    logits = []
    #-----------------------
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            y_true.append(data['label_tensor'])
            for k, v in data.items():
                data[k] = v.to(device)
            logits = model(data)
            y_hat = torch.argmax(logits, dim=-1).detach().cpu()
            y_pred.append(y_hat)
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        f1  = sklearn.metrics.f1_score(y_true, y_pred,average="weighted")
        return f1


def main(args):
    dgcn.utils.set_seed(args.seed)
    if torch.cuda.is_available():
        generator = generator = torch.Generator('cuda').manual_seed(args.seed)
    else:
        print("Because of package problem, this one is running on CPU")
        generator = torch.Generator().manual_seed(args.seed)
        args.device="cpu"
    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = dgcn.utils.load_pkl(args.data)
    log.info("Loaded data.")
    
    
    trainset = dgcn.Dataset(data["train"], args.batch_size)
    devset = dgcn.Dataset(data["dev"], args.batch_size)
    testset = dgcn.Dataset(data["test"], args.batch_size)

    log.debug("Building model...")
    model_file = conf["model_file"].format(args.Rtype+str(time.time()))
    model = dgcn.DialogueGCN(args).to(args.device)
    
    if not args.from_begin:
        ckpt = torch.load(model_file)
        best_dev_f1 = ckpt["best_dev_f1"]
        best_epoch = ckpt["best_epoch"]
        best_state = ckpt["best_state"]
        model.load_state_dict(self.best_state)

    # --------------------------------- begin to train -------------------------
    log.info("Start training...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.0001
    )
     # Can change as the paper itself
    # loss_fcn = nn.NLLLoss()
    
    if args.lossfunc == 'entropy':
        loss_fcn = nn.CrossEntropyLoss()
    else:
        if args.class_weight:
            loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                                          1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(args.device)
            loss_fcn = nn.NLLLoss(loss_weights)
        else:
            loss_fcn = nn.NLLLoss()
    

    best_state = None
    best_dev_f1 = None
    best_epoch = None
    test_f1 = None
    
    for epoch in range(args.epochs + 1):
        model.train()
        total_loss = 0
        for idx in tqdm(range(len(trainset)), desc="train epoch {}".format(epoch)):
        # for idx in range(len(trainset)):
            optimizer.zero_grad()
            idata = trainset[idx]
            label = idata['label_tensor'].to(args.device)
            for k, v in idata.items():
                idata[k] = v.to(args.device)
            logits = model(idata)
            loss = loss_fcn(logits, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_loss = total_loss/len(trainset)
        ev_train = evaluation(model, trainset)
        dev1 = ev_train
        if best_epoch is None or dev1 > best_dev_f1:
            best_dev_f1 = dev1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        ev_test = evaluation(model, testset)
        log.info('[Epochs {}: , loss: {},  f1_train:{}, f1_test: {}]'
              .format(epoch, avg_loss,  ev_train, ev_test)
             )
    # Save.
    checkpoint = {
        "best_dev_f1": best_dev_f1,
        "best_epoch": best_epoch,
        "best_state": best_state,
    }
    torch.save(checkpoint, model_file)
    ## Reload file and check!!!!!!!!!!!!!!
    model.load_state_dict(best_state)
    log.info("Best in epoch {}:".format(best_epoch))
    dev_f1 = evaluation(model, trainset)
    log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
    test_f1 = evaluation(model, testset)
    log.info("[Test set] [f1 {:.4f}]".format(test_f1))
    log.info("setting: last_layer, epoch, learning_rate, drop_rate, optimizer, weight_decay, f1: {}, {},{},{},{},{}".format(args.last_layer, args.epochs, args.learning_rate, args.drop_rate, args.optimizer, args.weight_decay, test_f1))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")

    # Training parameters
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")

    # Model parameters
    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")
    parser.add_argument("--lossfunc", type=str, default="ntll",
                        choices=["entropy", "ntll"], help="Type of  Loss function")
    parser.add_argument("--Rtype", type=str, default="MHA",
                        choices=["Final", "MHA"], help="Type of  relation graph transform type")
    
    parser.add_argument("--last_layer", type=str, default="h",
                        choices=["h", "add_X", "add_X_att"], help="If adding the last layer or not")

    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()
    log.debug(args)

    main(args)