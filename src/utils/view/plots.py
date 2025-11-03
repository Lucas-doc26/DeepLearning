import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.image_metrics import calculate_ssim
   
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pathlib
from pathlib import Path

def plot_autoencoder_with_ssim(dataset, autoencoder, width=128, height=128, save_path=None):

    #converte em um batch
    x_batch = next(iter(dataset))
    # Caso o dataset tenha (x, y)
    if isinstance(x_batch, tuple):
        x_batch = x_batch[0]

    # Converte para numpy
    x = x_batch.numpy()

    reconstructions = autoencoder.predict(x)

    n = min(8, len(x))

    plt.figure(figsize=(16, 4))  

    for i in range(n):
        img = x[i]
        recon = reconstructions[i]
        ssim = calculate_ssim(img, recon)

        #originais
        plt.subplot(2, n, i + 1)
        plt.imshow(img.reshape(width, height, 3), cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')

        #reconstruções
        plt.subplot(2, n, n + i + 1)
        plt.imshow(recon.reshape(width, height, 3), cmap='gray')
        plt.title(f"Recon {i+1}\nSSIM: {ssim:.3f}")
        plt.axis('off')

    plt.tight_layout()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_history(history, type='Classifier', save_fig=None):
    # Função auxiliar pra evitar erro com chave inexistente
    def get_hist(key):
        return history.history.get(key, [])

    if type == 'Classifier':
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        accuracy = get_hist('accuracy')
        val_accuracy = get_hist('val_accuracy')
        epochs = range(len(loss))

        plt.figure(figsize=(15, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        #plt.xticks(epochs)
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        #plt.xticks(epochs)
        plt.legend()

    elif type == 'Autoencoder':
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        epochs = range(len(loss))

        print(epochs, loss, val_loss)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        #plt.xticks(epochs)
        plt.legend()

    else:  # CVAE ou CCVAE
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        kl_loss = get_hist('kl_loss')
        val_kl_loss = get_hist('val_kl_loss')
        reconstruction_loss = get_hist('reconstruction_loss')
        val_reconstruction_loss = get_hist('val_reconstruction_loss')
        epochs = range(len(loss))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        #plt.xticks(epochs)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, kl_loss, label='kl_loss')
        plt.plot(epochs, val_kl_loss, label='val_kl_loss')
        plt.title('KL Loss')
        #plt.xticks(epochs)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, reconstruction_loss, label='reconstruction_loss')
        plt.plot(epochs, val_reconstruction_loss, label='val_reconstruction_loss')
        plt.title('Reconstruction Loss')
        #plt.xticks(epochs)
        plt.legend()

    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=['Empty', 'Occupied'], legend:str=None , save_path=None):
    """
    Plota uma matriz de confusão.

    Args:
        y_true: Array numpy com os rótulos verdadeiros
        y_pred: Array numpy com as previsões do modelo
        labels: Lista de rótulos das classes
        title: Título da figura (opcional)
        save_path: Caminho para salvar a figura (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    plt.title(f"{accuracy_text}")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if legend:
        if isinstance(legend, str):
            legend = [legend]
        patches = [mpatches.Patch(color='lightblue', label=text) for text in legend]
        plt.legend(handles=patches, loc='lower right', fontsize=10, frameon=True)

    # Salvar ou exibir a figura
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_pca_CVAE(dimensionality=2, z_vectors=None, labels=None):
    pca = PCA(n_components=dimensionality)
    z_pca = pca.fit_transform(z_vectors)
    save_path=f'Images/CVAE-PCA-{dimensionality}D.png'

    def pca_2d(z_pca, labels):
        plt.figure(figsize=(8,6))
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(z_pca[idx,0], z_pca[idx,1], label=f'Classe {cls}', alpha=0.6)
        plt.legend()
        plt.title("Espaço latente do CVAE (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(save_path)
        plt.close()
    
    def pca_3d(z_pca, labels):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(labels):
            idx = labels == cls
            ax.scatter(z_pca[idx,0], z_pca[idx,1], z_pca[idx,2], label=f'Classe {cls}', alpha=0.6)
        ax.set_title("Espaço latente do CVAE (PCA 3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.savefig(save_path)
        plt.close()

    if dimensionality == 2: 
        pca_2d(z_pca, labels) 
    else: 
        pca_3d(z_pca, labels)

def plot_tsne_CVAE(dimensionality=2, z_vectors=None, labels=None):
    tsne = TSNE(n_components=dimensionality, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(z_vectors)
    save_path = f'Images/CVAE-TSNE-{dimensionality}D.png'

    def tsne_2d(z_tsne, labels):
        plt.figure(figsize=(8,6))
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(z_tsne[idx,0], z_tsne[idx,1], label=f'Classe {cls}', alpha=0.6)
        plt.legend()
        plt.title("Espaço latente do CVAE (t-SNE)")
        plt.savefig(save_path)
        plt.close()
    
    def tsne_3d(z_tsne, labels):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(labels):
            idx = labels == cls
            ax.scatter(z_tsne[idx,0], z_tsne[idx,1], z_tsne[idx,2], label=f'Classe {cls}', alpha=0.6)
        ax.set_title("Espaço latente do CVAE (t-SNE 3D)")
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
        ax.set_zlabel("Dim3")
        ax.legend()
        plt.savefig(save_path)
        plt.close()

    if dimensionality == 2: 
        tsne_2d(z_tsne, labels) 
    else: 
        tsne_3d(z_tsne, labels)

def plot_tsne_plotply(dimensionality=3, z_vectors=None, labels=None, save_path="tsne_plot.html"):
    labels_binary = np.array([0 if l == 0 else 1 for l in labels])

    labels_names = np.array(['Classe 0', 'Classe 1'])[labels_binary]

    tsne = TSNE(n_components=dimensionality, random_state=0)
    projections = tsne.fit_transform(z_vectors)

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=labels_names,
        labels={'color': 'Classes'},  
        color_discrete_map={'Classe 0': 'blue', 'Classe 1': 'green'}
    )
    fig.update_traces(marker_size=8)
    fig.write_html(save_path)
    