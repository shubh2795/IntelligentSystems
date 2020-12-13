from find_usenet_posts import *
usr_q = 'is fuel injector cleaning necessary?'
usenet_data = load_usenet_data()
vectorizer, feat_mat = vocab_normalize_usenet_data(usenet_data)
find_top_n_posts(vectorizer, usr_q, feat_mat, norm_euclid_dist, top_n=5)