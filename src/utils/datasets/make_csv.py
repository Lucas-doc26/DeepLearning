import pandas as pd
import os
import random
import cv2
import albumentations as A
from utils import albumentations_tf, salt_and_pepper
import yaml

SEED = 42
random.seed(SEED)
map_classes = {'occupied': 1, 'empty': 0}

class BaseDataset:
    def __init__(self, yaml):
        self.dataset_name = None
        self.csv_dir = None
        self.dataset_path = None
        self.classes = None
        self.split = None 
        self.data_augmentation = None
        self.config(yaml)

    def config(self, data):
        self.dataset_name = data.get('name')
        self.csv_dir = data.get('paths').get('csv')
        self.dataset_path = data.get('paths').get('processed')
        self.classes = data.get('classes')
        self.split = data.get('split_subjects') or data.get('split')
        self.data_augmentation = data.get('augmentations')

    def to_csv(self):
        os.makedirs(self.csv_dir, exist_ok=True)

        if not self.classes:
            images = [os.path.join(self.dataset_path, img) for img in os.listdir(self.dataset_path)]
            df = pd.DataFrame(images, columns=["path_image"])

            df.to_csv(os.path.join(self.csv_dir, f'{self.dataset_name}.csv'), index=False)
        else:
            if self.dataset_name == 'CNR':
                path_labels = os.path.join(self.dataset_path, 'LABELS')
                labels = os.listdir(path_labels)
                for label in labels:
                    df = pd.read_csv(os.path.join(path_labels, label), sep=' ', header=None, names=['path_image', 'class'])
                    df['path_image'] = df['path_image'].apply(lambda x: os.path.join(self.dataset_path, 'PATCHES', x))
                    #df['class'] = df['class'].map(map_classes)
                    label = label.split('.')[0]
                    if label == 'all':
                        label = 'CNR'
                    df.to_csv(os.path.join(self.csv_dir, f'{label}.csv'), index=False)

def create_datasets_from_yaml():                                                                                        
    yaml_data = yaml.safe_load(open('/home/c.oliveira25/Desktop/DeepLearning/config/dataset.yaml', 'r'))
    download = yaml_data['datasets'].get('download_dir')
    csv_dir = yaml_data['datasets'].get('csv_dir')
    dataset = DatasetManager(dataset_path=str(download), csv_dir=str(csv_dir))
    print(dataset.dataset_path)
    dataset.create_csvs()


class DatasetManager:
    def __init__(self, dataset_path=None, csv_dir=None):
        self.dataset_path = dataset_path
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)
        self.create_important_directories()

    def create_important_directories(self, 
        directories=['PKLot', 'Kyoto', 'CNR', 
                     'PUC', 'UFPR04', 'UFPR05', 
                     'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']):
        """
        Cria diretórios importantes para o gerenciamento dos datasets
        """
        for dir in directories:
            dir_path = os.path.join(self.csv_dir, dir)
            os.makedirs(dir_path, exist_ok=True)

    def create_csvs(self):
        """
        Cria arquivos CSV com os caminhos das imagens e seus rótulos
        """
        CNRDataset(self.dataset_path, self.csv_dir).create_csv_cnr()
        #PKLotDataset(self.dataset_path, self.csv_dir).create_csv_PKLot()
        #KyotoDataset(self.dataset_path, self.csv_dir)
        #ASVSPOOFDataset(self.dataset_path, self.csv_dir).create_csv_ASVSPOOF()
        shuffle_dataset(self.csv_dir)

    def _return_days(self, days, dict_days_values, n_images, n_days): 
        total_images = 0
        days_selected = []
        images_per_day_total_list = []

        for day in days[:]:  # copia para não quebrar o original
            if total_images >= n_images:
                break

            empty, occupied = dict_days_values[day]
            images_per_class = min(empty, occupied)

            if images_per_class == 0:
                continue  # pula o dia se não houver amostras

            # quantas ainda faltam
            missing_images = n_images - total_images  

            # se passar do limite, corta proporcionalmente
            if 2 * images_per_class > missing_images:
                images_per_class = missing_images // 2

            # se mesmo assim não couber nada, pula
            if images_per_class == 0:
                continue  

            # adiciona este dia
            total_images += 2 * images_per_class
            days_selected.append(day)
            images_per_day_total_list.append(2 * images_per_class)

            if len(days_selected) >= n_days:
                break

        # dias restantes que não entraram
        days_left = sorted(list(set(days) - set(days_selected)))

        return days_selected, days_left, images_per_day_total_list

    def _create_df_per_days(self, df, days, images_per_day=None):
        df_final = pd.DataFrame(columns=['path_image', 'class'])

        if images_per_day is None:
            for day in days:
                df_day = df[df['day'] == day]
                df_final = pd.concat([df_final, df_day], ignore_index=True)
        else:
            for day, n_images in zip(days, images_per_day):
                df_day = df[df['day'] == day]
                empty = df_day[df_day['class'] == 0]
                occupied = df_day[df_day['class'] == 1]

                half = n_images // 2
                half_plus = half + (n_images % 2)  # caso seja ímpar, a classe 0 recebe a imagem a mais

                empty_sample = empty.sample(n=min(half_plus, len(empty)), random_state=SEED, replace=(len(empty) < half_plus))
                occupied_sample = occupied.sample(n=min(half, len(occupied)), random_state=SEED, replace=(len(occupied) < half))

                df_final = pd.concat([df_final, empty_sample, occupied_sample], ignore_index=True)

        return df_final

    def split_train_valid_test(self, df, n_days_train, n_days_valid, n_images_train, n_images_valid):
        days = sorted(df['day'].unique().tolist())

        list_df_per_day = [df[df['day'] == day] for day in days]
        number_of_images_per_day = {
            #pegando o dia
            day_df['day'].iloc[0]: (
                len(day_df[day_df['class'] == 0]), #conta a quantidade de cada classe
                len(day_df[day_df['class'] == 1])
            )
            for day_df in list_df_per_day
        } #retorno um dic -> 2023-10-19: (678, 123), 2023-10-20: (876, 456)
        days_to_train, days_left, images_per_class_train = self._return_days(days, number_of_images_per_day, n_images_train, n_days_train)

        days_to_valid, days_to_test, images_per_class_valid = self._return_days(days_left, number_of_images_per_day, n_images_valid, n_days_valid)

        df_train = self._create_df_per_days(df, days_to_train, images_per_class_train)
        df_valid = self._create_df_per_days(df, days_to_valid, images_per_class_valid)
        df_test = self._create_df_per_days(df, days_to_test, None)

        return df_train, df_valid, df_test
    
    def create_weather_dataset(self,df):
        weather = df["weather"].str.upper()

        df_sunny = df[weather == "SUNNY"]
        df_rainy = df[weather == "RAINY"]

        if "CLOUDY" in weather.unique():
            df_cloudy = df[weather == "CLOUDY"]
        else:
            df_cloudy = df[weather.isin(["CLOUDY", "OVERCAST"])].copy()
            df_cloudy["weather"] = "CLOUDY"

        return df_sunny, df_rainy, df_cloudy

    def save_datasets(self, df, train, valid, test, base, weather=None):
        if weather:
            weather_dir = os.path.join(self.csv_dir,base, weather)
            os.makedirs(weather_dir, exist_ok=True )

            df.to_csv(os.path.join(weather_dir, f'{base}_{weather}.csv'), index=False, columns=["path_image", "class", "weather"])
            train.to_csv(os.path.join(weather_dir, f'{base}_{weather}_train.csv'), index=False, columns=["path_image", "class", "weather"])
            valid.to_csv(os.path.join(weather_dir, f'{base}_{weather}_valid.csv'), index=False, columns=["path_image", "class", "weather"])
            test.to_csv(os.path.join(weather_dir, f'{base}_{weather}_test.csv'), index=False, columns=["path_image", "class", "weather"])
        else:
            df.to_csv(os.path.join(self.csv_dir,f'{base}/{base}.csv'), index=False, columns=["path_image", "class"])
            train.to_csv(os.path.join(self.csv_dir,f'{base}/{base}_train.csv'), index=False, columns=["path_image", "class"])
            valid.to_csv(os.path.join(self.csv_dir,f'{base}/{base}_valid.csv'), index=False, columns=["path_image", "class"])
            test.to_csv(os.path.join(self.csv_dir,f'{base}/{base}_test.csv'), index=False, columns=["path_image", "class"])

class CNRDataset(DatasetManager):
    def __init__(self, dataset_path=None, csv_dir=None):
        super().__init__(dataset_path, csv_dir)
        self.dataset_name = "CNR"

    def create_csv_cnr(self, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria um arquivo CSV com os caminhos das imagens e seus rótulos
        """
        cnr_dir = os.path.join(self.dataset_path, self.dataset_name)
        all_cnr = os.path.join(cnr_dir, 'LABELS', 'all.txt')

        df = pd.read_csv(all_cnr, sep=' ', header=None, names=['path_image', 'class'])

        path_split = df["path_image"].str.split("/", expand=True)

        df["weather"] = path_split[0]
        df["day"] = path_split[1]
        df["camera"] = path_split[2]
        df['path_image'] = df['path_image'].apply(lambda x: os.path.join(cnr_dir, 'PATCHES', x))

        self.create_cameras(df)

        train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df,train, valid, test, 'CNR', None)

        df_sunny, df_rainy, df_cloudy = self.create_weather_dataset(df)

        train_sunny, valid_sunny, test_sunny = self.split_train_valid_test(df_sunny, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_sunny,train_sunny, valid_sunny, test_sunny, 'CNR', 'Sunny')

        train_rainy, valid_rainy, test_rainy = self.split_train_valid_test(df_rainy, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_rainy, train_rainy, valid_rainy, test_rainy, 'CNR', 'Rainy')

        train_cloudy, valid_cloudy, test_cloudy = self.split_train_valid_test(df_cloudy, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_cloudy, train_cloudy,  valid_cloudy,  test_cloudy,  'CNR', 'Cloudy')

    def create_cameras(self, df, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o csv para cada câmera
        """
        cameras = df["camera"].unique().tolist()
        for cam in sorted(cameras):
            df_cam = df[df["camera"] == cam]

            train, valid, test = self.split_train_valid_test(df_cam, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_cam, train, valid, test, cam)

            df_sunny, df_rainy, df_cloudy = self.create_weather_dataset(df)

            train_sunny, valid_sunny, test_sunny = self.split_train_valid_test(df_sunny, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_sunny, train_sunny, valid_sunny, test_sunny, cam, 'Sunny')

            train_rainy, valid_rainy, test_rainy = self.split_train_valid_test(df_rainy, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_rainy, train_rainy, valid_rainy, test_rainy, cam, 'Rainy')

            train_cloudy, valid_cloudy, test_cloudy = self.split_train_valid_test(df_cloudy, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_cloudy, train_cloudy, valid_cloudy, test_cloudy, cam, 'Cloudy')

class PKLotDataset(DatasetManager):
    def __init__(self, dataset_path=None, csv_dir=None):
        super().__init__(dataset_path, csv_dir)
        self.dataset_name = 'PKLot/PKLotSegmented'
        self.universities = ["PUC", "UFPR04", "UFPR05"]
        self.weathers = ["Cloudy", "Sunny", "Rainy"]

    def create_csv_PKLot(self, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o arquivo CSV da PKLot
        """
        data = []
        for university in self.universities:
            for weather in self.weathers:
                days_dir = os.path.join(self.dataset_path, self.dataset_name, university, weather)
                for day in os.listdir(days_dir):
                    day_dir = os.path.join(days_dir, day)
                    for label in ['Empty', 'Occupied']:
                        dir_imgs = os.path.join(day_dir, label)
                        if os.path.isdir(dir_imgs):
                            path_imgs = os.listdir(dir_imgs)
                            data.extend([[university, weather, day, os.path.join(day_dir, label, img), label] 
                                         for img in path_imgs])
        df = pd.DataFrame(data=data, columns=['university', 'weather', 'day', 'path_image', 'class'])
        df['class'] = df['class'].map(map_classes)

        self.create_universities(df)

        train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df, train, valid, test, 'PKLot')

        df_sunny, df_rainy, df_cloudy = self.create_weather_dataset(df)
        train_sunny, valid_sunny, test_sunny = self.split_train_valid_test(df_sunny, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_sunny, train_sunny, valid_sunny, test_sunny, 'PKLot', 'Sunny')

        train_rainy, valid_rainy, test_rainy = self.split_train_valid_test(df_rainy, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_rainy, train_rainy, valid_rainy, test_rainy, 'PKLot', 'Rainy')

        train_cloudy, valid_cloudy, test_cloudy = self.split_train_valid_test(df_cloudy, days_train, days_valid, n_images_train, n_images_valid)
        self.save_datasets(df_cloudy, train_cloudy, valid_cloudy, test_cloudy, 'PKLot', 'Cloudy')

    def create_universities(self, df, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o arquivo csv de cada uma das faculdades
        """
        for university in self.universities:
            df_university = df[df["university"] == university]

            train, valid, test = self.split_train_valid_test(df_university, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_university, train, valid, test, university)

            df_sunny, df_rainy, df_cloudy = self.create_weather_dataset(df_university)

            train_sunny, valid_sunny, test_sunny = self.split_train_valid_test(df_sunny, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_sunny, train_sunny, valid_sunny, test_sunny, university, 'Sunny')

            train_rainy, valid_rainy, test_rainy = self.split_train_valid_test(df_rainy, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_rainy, train_rainy, valid_rainy, test_rainy, university, 'Rainy')

            train_cloudy, valid_cloudy, test_cloudy = self.split_train_valid_test(df_cloudy, days_train, days_valid, n_images_train, n_images_valid)
            self.save_datasets(df_cloudy, train_cloudy, valid_cloudy, test_cloudy, university, 'Cloudy')

class KyotoDataset(DatasetManager):
    def __init__(self, dataset_path=None, csv_dir=None):
        super().__init__(dataset_path, csv_dir)
        self.dataset_name = 'kyoto'
        self.dataset_path_kyoto = os.path.join(self.dataset_path, self.dataset_name)
        self.base_imgs = os.listdir(self.dataset_path_kyoto)
        print(self.base_imgs)
        self.data_aug()
        self.create_csv_kyoto()

    def data_aug(self):
        self.apply_dataaug(salt_and_pepper(), 1)

        transform2 = A.Compose([
            A.RandomRain(
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180),  blur_value=5,brightness_coefficient=0.8, p=0.15
            ),
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=0.15),
            A.ChannelShuffle(p=0.15),
            A.Rotate(limit=40, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=0.15),
            A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=0.15),
            #A.Resize(height=256, width=256)
        ])
        self.apply_dataaug(transform2, 2)
        
        transform3 = A.Compose([
                    A.ChannelShuffle(p=1),
        ])

        self.apply_dataaug(transform3, 3)

        transform4 = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=1),
                    A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=1),
        ])

        self.apply_dataaug(transform4, 4)

    def apply_dataaug(self, transform, id):
        for img_path in self.base_imgs:
            img = cv2.imread(os.path.join(self.dataset_path_kyoto, img_path), cv2.IMREAD_COLOR)
            img_augmented = albumentations_tf(img, transform=transform)
            img_name = img_path.split("/")[-1]
            new_name = f"{img_name.split('.')[0]}-{id}.jpg"
            cv2.imwrite(os.path.join(self.dataset_path, new_name), img_augmented)

    def create_csv_kyoto(self):
        images = os.listdir(self.dataset_path_kyoto)
        df = pd.DataFrame(images, columns=["path_image"])
        kyoto_dir = os.path.join(self.csv_dir, 'kyoto')
        os.makedirs(kyoto_dir, exist_ok=True)

        df.to_csv(os.path.join(kyoto_dir,'kyoto.csv'), index=False)

class ASVSPOOFDataset(DatasetManager):
    def __init__(self, dataset_path=None, csv_dir=None):
        super().__init__(dataset_path, csv_dir)
        self.dataset_name = 'ASVSPOOF'
        self.dataset_path_asvspoof = os.path.join(self.dataset_path, self.dataset_name)

    def create_csv_ASVSPOOF(self):
        images = []
        labels = []
        for label in os.listdir(self.dataset_path_asvspoof):
            dataset_path_label = os.path.join(self.dataset_path_asvspoof, label)
            for img in os.listdir(dataset_path_label):
                images.append(os.path.join(dataset_path_label, img))
                labels.append(label)
        df = pd.DataFrame({'path_image':images, 'class':labels})
        asvsopoof_dir = os.path.join(self.csv_dir, 'asvsopoof')
        os.makedirs(asvsopoof_dir, exist_ok=True)

        df.to_csv(os.path.join(asvsopoof_dir, 'asvsopoof.csv'), index=False)
 
def shuffle_dataset(path_datasets):
    for root, dir, csv in os.walk(path_datasets):
         for fname in csv:
            if fname.lower().endswith('.csv'):
                csv_path = os.path.join(root, fname)
                # lê, embaralha, reseta índice e grava de volta sem índice extra
                df = pd.read_csv(csv_path)
                df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
                df_shuffled.to_csv(csv_path, index=False)
                print(f'Embaralhado: {csv_path} ({len(df_shuffled)} linhas)')
