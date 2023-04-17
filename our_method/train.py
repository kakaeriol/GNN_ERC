import argparse
import torch
import dgcn
from torch.utils.data import DataLoader
import tensorflow
from tensorflow import keras
log = dgcn.utils.get_logger()


def main(args):
    dgcn.utils.set_seed(args.seed)
    if torch.cuda.is_available():
        generator = generator = torch.Generator('cuda').manual_seed(args.seed)
    else:

        print("ERROR LOADING CUDA, USING CPU")
        args.device = "cpu"
        generator = torch.Generator().manual_seed(args.seed)
        args.device="cpu"
    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = dgcn.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = dgcn.my_DataSet(args.data, "train")
    devset = dgcn.my_DataSet(args.data, "test")
    testset = dgcn.my_DataSet(args.data, "dev")
    
    ###
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=0,
                              pin_memory=False
                             )

    valid_loader = DataLoader(devset,
                              batch_size=args.batch_size,
                              collate_fn=devset.collate_fn,
                              num_workers=0,
                              pin_memory=False
                              )

    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=0,
                              pin_memory=False
                             )

    log.debug("Building model...")
    model_file = "./save/model_{}.pt".format(args.data.split('/')[-1][:len('token_fts.pkl')])
    model = dgcn.DialogueGCN(args).to(args.device)
    opt = dgcn.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = dgcn.Coach(train_loader, valid_loader, test_loader, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


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
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    parser.add_argument("--lossfunc", type=str, default='nll',
                        help=" possible losses: entropy, nll.")
    # Model parameters
    
    
    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")
    parser.add_argument("--pretrained_word_vectors",  type=str, default="",
                        help="Place to store the vector")
    
    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()
    log.debug(args)

    main(args)

