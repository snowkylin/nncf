from utils import CiteULikeDataLoader

data_loader = CiteULikeDataLoader()
x_u_all, x_v_all = data_loader.fetch_batch()