from model.AE import AE
from model.IGAE import IGAE
from model.MGCN import MGCN

def creat_model(model_name, args):
    model = None
    if model_name =='ae':
        model = AE(
            ae_n_enc_1=128,
            ae_n_enc_2=256,
            ae_n_enc_3=20,
            ae_n_dec_1=20,
            ae_n_dec_2=256,
            ae_n_dec_3=128,
            n_input=args.n_input,
            n_z=args.n_z)
    
    elif model_name =='igae':
        model = IGAE(
            gae_n_enc_1=128,
            gae_n_enc_2=256,
            gae_n_enc_3=20,
            gae_n_dec_1=20,
            gae_n_dec_2=256,
            gae_n_dec_3=128,
            n_input=args.n_input,
        )
    
    elif model_name =='mgcn':
        model = MGCN(
            ae_n_enc_1=128,
            ae_n_enc_2=256,
            ae_n_enc_3=20,
            ae_n_dec_1=20,
            ae_n_dec_2=256,
            ae_n_dec_3=128,
            gae_n_enc_1=128,
            gae_n_enc_2=256,
            gae_n_enc_3=20,
            gae_n_dec_1=20,
            gae_n_dec_2=256,
            gae_n_dec_3=128,
            n_input=args.n_input, sigma=args.sigma,
            n_z=args.n_z,
            n_clusters=args.n_clusters,
            v=args.freedom_degree)
    return model

    

