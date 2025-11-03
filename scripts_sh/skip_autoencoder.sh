
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder.py --train CNR --epochs 100
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder.py --train kyoto --epochs 100
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder.py --train PKLot --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder.py --train CNR --epochs 100
python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder.py --train kyoto --epochs 100
python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder.py --train PKLot --epochs 100

#Treinando o skip
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder kyoto --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder kyoto --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder kyoto --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder CNR --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder CNR --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder CNR --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder PKLot --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder PKLot --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_fully_connected.py --autoencoder PKLot --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

#Treinando o SVM
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder kyoto --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder kyoto --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder kyoto --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder CNR --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder CNR --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder CNR --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder PKLot --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder PKLot --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_svm.py --autoencoder PKLot --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 
###########
# Aqui é ##
# VAE    ##
###########
#Treinando o skip
python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder kyoto --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder kyoto --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder kyoto --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder CNR --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder CNR --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder CNR --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder PKLot --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder PKLot --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_fully_connected.py --autoencoder PKLot --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 --epochs 100



#Treinando o SVM
python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder kyoto --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder kyoto --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder kyoto --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder CNR --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder CNR --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder CNR --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder PKLot --train UFPR05 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder PKLot --train UFPR04 --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 

python /home/lucas/DeepLearning/src/pipelines/train_variational_autoencoder_svm.py --autoencoder PKLot --train PUC --test UFPR05 UFPR04 PUC camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 