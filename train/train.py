import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils.misc import *

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []

def Train(epoch, model, data, adj, label, device, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(torch.load(f'save/{args.dataset}/pretrain.pkl', map_location='cpu'))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model.init(data, adj)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(epoch):
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(data, adj)

        tmp_q = q.data
        p = target_distribution(tmp_q).detach()

        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_s = F.mse_loss(z_igae, z_ae)
        loss_igae = args.loss_w * loss_w + args.loss_a * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + args.loss_s * loss_s + args.loss_kl * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(data, adj)
                print('{:3d} loss: {}'.format(epoch, loss))
                kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

                acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
                acc_reuslt.append(acc)
                nmi_result.append(nmi)
                ari_result.append(ari)
                f1_result.append(f1)

                if acc > args.acc:
                    args.acc = acc
                    # torch.save(model.state_dict(), path)
    print_results(acc_reuslt, nmi_result, ari_result, f1_result, args)