from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE


def reduce(x, method, dims):
    if method == "pca":
        pca = PCA(n_components=dims)
        reduced = pca.fit_transform(x)

        return reduced.transpose()
    if method == "ica":
        ica = FastICA(n_components=dims, random_state=0)
        reduced = ica.fit_transform(x)

        return reduced.transpose()
    if method == "tsne":
        tsne = TSNE(n_components=dims)
        reduced = tsne.fit_transform(x)

        return reduced.transpose()
