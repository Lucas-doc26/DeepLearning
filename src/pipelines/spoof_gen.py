import os
import pandas as pd

def make_csv(dataset, num_per_class, save):
    df = pd.read_csv(
        f'/home/lucas/DeepLearning/CSV/Spoof/{dataset}.txt',
        sep=' ',
        names=['path_image', 'class']
    )

    df['class'] = df['class'].astype(str)

    df['path_image'] = df['path_image'].apply(
        lambda x: os.path.join('/datasets/asvspoof2019', x.split("/files/")[-1])
    )

    print(f"\n[{dataset}] Distribuição original:")
    print(df['class'].value_counts())

    dfs = []
    for class_value in df['class'].unique():
        subset = df[df['class'] == class_value]
        if subset.empty:
            print(f"⚠️ Classe {class_value} não encontrada, pulando...")
            continue

        # Garante que não vai pedir mais do que existe
        sample_size = min(num_per_class, len(subset))
        dfs.append(subset.sample(n=sample_size, random_state=42))
        print(f"→ Classe {class_value}: usando {sample_size} exemplos.")

    df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced.to_csv(f'/home/lucas/DeepLearning/CSV/Spoof/Spoof_{save}.csv', index=False)

    print(f"\n✅ Dataset balanceado salvo em Spoof_{save}.csv")
    print(df_balanced['class'].value_counts())


make_csv('train', 5000, 'train')
make_csv('val', 2500, 'valid')

test = pd.read_csv("/home/lucas/DeepLearning/CSV/Spoof/test.txt", sep=" ", names=['path_image', 'class'])

test['path_image'] = test['path_image'].apply(
    lambda x:
    os.path.join("/datasets/asvspoof2019", x.split("/files/")[-1])
)

test.to_csv(f"/home/lucas/DeepLearning/CSV/Spoof/Spoof_test.csv", index=False)
