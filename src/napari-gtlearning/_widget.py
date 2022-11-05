"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from napari.types import ImageData, LabelsData
import os
import tensorflow as tf
from magicgui.tqdm import trange
import tempfile
from zipfile import ZipFile
from napari import Viewer
from tensorflow.keras import backend as K
import napari
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from qtpy.QtWidgets import QFileDialog, QListWidget
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from napari.utils.notifications import show_info
from focal_loss import BinaryFocalLoss
import pathlib
import shutil

import napari_gtlearning.path as paths

zip_dir = tempfile.TemporaryDirectory()

def get_mosaic(img):
    A = []
    h,l,z = img.shape
    #longueur
    L1 = [ i for i in range(0,l-255,255)]+[l-255]
    L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

    #hauteur
    R1 = [ i for i in range(0,h-255,255)]+[h-255]
    R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]

    for h1,h2 in zip(R1,R2):
        for l1,l2 in zip(L1,L2):
            A.append(img[h1:h2,l1:l2])
    return A

def reconstruire(img1,K):
    ex,ey,ez=img1.shape
    A = np.zeros((ex,ey,1), dtype=np.bool)
    h=ex
    l=ey
    z=1
  
    #longueur
    L1 = [ i for i in range(0,l-255,255)]+[l-255]
    L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

    #hauteur
    R1 = [ i for i in range(0,h-255,255)]+[h-255]
    R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]
  
    n = 0
    for h1,h2 in zip(R1,R2):
        for l1,l2 in zip(L1,L2):
            if A[h1:h2,l1:l2].shape == K[n].shape:
                A[h1:h2,l1:l2] = K[n]
            else:
                A[h1:h2,l1:l2] = np.zeros(A[h1:h2,l1:l2].shape, dtype=np.bool)
            n+=1
    return A

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 

def do_image_segmentation(layer: ImageData) -> ImageData:
    
    img1_list = get_mosaic(layer)

    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),'best_model_W_BCE_chpping.h5'),custom_objects={'dice_coefficient': dice_coefficient})

    taille_p = 256
    X_ensemble = np.zeros((len(img1_list), taille_p, taille_p, 3), dtype=np.uint8)
    for n in range(len(img1_list)):
      sz1_x,sz2_x,sz3_x = img1_list[n].shape
      if (sz1_x,sz2_x)==(256,256):
        X_ensemble[n]=img1_list[n]

    preds_test = model_New.predict(X_ensemble, verbose=1)
    preds_test_opt = (preds_test > 0.2).astype(np.uint8)
    output_image = reconstruire(layer,preds_test_opt)
    return np.squeeze(output_image[:,:,0])

@magic_factory(call_button="Load images",filename={"label": "Pick a file:"})    
def process_function_load(napari_viewer : Viewer,filename=pathlib.Path.cwd()):
    
    with ZipFile(filename,'r') as zipObject:
        listOfFileNames = zipObject.namelist()
        for i in range(len(listOfFileNames)):
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)

    path_folder = zip_dir.name.replace("\\","/")
    folder = os.listdir(zip_dir.name.replace("\\","/"))[0]
    files_of_folder = os.listdir(path_folder+'/'+folder)

    A = [path_folder+'/'+folder+'/'+files_of_folder[i] for i in range(len(files_of_folder))]

    os.mkdir(path_folder+'/mask')

    MASK = []
    
    names = []
    for ix in range(len(A)):
        names.append(A[ix].split('/')[-1][:-4])
        image_data = imread(A[ix])
        show_info(f'Image processed ... {ix}/{len(A)}')
        MASK.append(do_image_segmentation(image_data))
    
    for iy in range(len(MASK)):
        path_mask = path_folder+'/mask/mask_'+A[iy].split('/')[-1][:-4]+'.png'
        plt.imsave(path_mask, MASK[iy], cmap = plt.cm.gray)
        
    names = [A[ix].split('/')[-1][:-4] for ix in range(len(A))]
    B = [path_folder+'/mask/mask_'+A[iy].split('/')[-1][:-4]+'.png' for iy in range(len(MASK))]
    
    for i,j in zip(A,B):
        new_folder_image = i.split('/')[-1][:-4]
    
        os.mkdir(path_folder+'/'+new_folder_image)
        
        new_file = path_folder+'/'+new_folder_image+'/'+i.split('/')[-1]
        os.replace(i,new_file)
        
        new_mask = path_folder+'/'+new_folder_image+'/'+j.split('/')[-1]
        os.replace(j,new_mask)

    os.rmdir(path_folder+'/'+folder)
    os.rmdir(path_folder+'/mask')

    def open_name(item):
        
        name = item.text()
        name_folder = name[:-4]
        print('name :',name)
        print('name_folder :',name_folder)
        print('Loading', name, '...')

        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
        print('zip_dir.name :',zip_dir.name)
        print('name :',name)
        fname = f'{zip_dir.name}\{name}'
        print('fname :',fname)
        for fname_i in os.listdir(fname):
            if fname_i.find('mask')!=-1:
                data_label = imread(f'{fname}\{fname_i}')
                if len(data_label.shape)==3:
                    data_label_output = data_label[:,:,0]
                else:
                    data_label_output = data_label[:,:]
                data_label1 = np.array(data_label_output)
                non_fleur=np.where(data_label1==0)
                fleur=np.where(data_label1==255)
                data_label1[non_fleur]=0
                data_label1[fleur]=1
                
                napari_viewer.add_labels(data_label1,name=f'{fname_i[:-4]}')                
            else:
                napari_viewer.add_image(imread(f'{fname}\{fname_i}'),name=f'{fname_i[:-4]}')

        print('... done.')
    
    list_widget = QListWidget()
    print('names :',names)
    for n in names:
        list_widget.addItem(n)    
    list_widget.currentItemChanged.connect(open_name)   
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)