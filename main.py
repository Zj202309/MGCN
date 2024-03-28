from utils.misc import *
from train.train import Train
from datasets.data_utils import load_dataset
from model.Creat_model import creat_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, adj= load_dataset(args)
    args = load_configs(args, "config/configs.yml")
    set_random_seed(args.seed)

    pca = PCA(n_components=args.n_input)
    X_pca = pca.fit_transform(x)
    dataset = LoadDataset(X_pca)

    adj = adj.to(device)
    data = torch.Tensor(dataset.x).to(device)
    label = y

    model = creat_model('mgcn', args).to(device)
    Train(args.epoch, model, data, adj, label, device, args)

if __name__ == "__main__":
    args = build_args()
    main(args)