import json
import uuid
import math
import numpy as np
import rasterio
import cv2
import tqdm
from sklearn.model_selection import train_test_split


class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.category = -1
        self.score = -1


class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.objects = list([])
        
    def load(self):
        src_raster = rasterio.open('xview_recognition/'+self.filename, 'r')
        input_type = src_raster.profile['dtype']
        input_channels = src_raster.count
        img = np.zeros((src_raster.height, src_raster.width, src_raster.count), dtype=input_type)
        for band in range(input_channels):
            img[:, :, band] = src_raster.read(band+1)
        self.img = img
        
    def scale(self, scale):
        if scale != 1:
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale)

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)
    

class Dataset:

    def __init__(self, json_file='xview_recognition/xview_ann_train.json'):
    
        self.json_file = json_file
        self.categories = {0: 'Cargo plane', 1: 'Helicopter', 2: 'Small car', 3: 'Bus', 4: 'Truck', 5: 'Motorboat', 6: 'Fishing vessel', 7: 'Dump truck', 8: 'Excavator', 9: 'Building', 10: 'Storage tank', 11: 'Shipping container'}        
        
    def load_data(self, subset=0):
        
        with open(self.json_file) as ifs:
            json_data = json.load(ifs)

        counts = dict.fromkeys(self.categories.values(), 0)
        
        self.anns = []
        
        if subset != 0:
            subset = list(zip(json_data['images'].values(), json_data['annotations'].values()))[:subset]
        else:
            subset = list(zip(json_data['images'].values(), json_data['annotations'].values()))
        
        for json_img, json_ann in tqdm.tqdm(subset):
            image = GenericImage(json_img['filename'])
            image.tile = np.array([0, 0, json_img['width'], json_img['height']])
            image.load()
            
            obj = GenericObject()
            obj.bb = (int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
            obj.category = json_ann['category_id']
            
            counts[obj.category] += 1
            image.add_object(obj)
            
            self.anns.append(image)
            
    def scale_dataset(self, scale=1):
        for image in self.anns:
            image.scale(scale)        
            
    def split_train_test(self, test_size=0.1):

        anns_train, anns_valid = train_test_split(self.anns, test_size=test_size, random_state=1, shuffle=True)
        
        print('Number of training images: ' + str(len(anns_train)))
        print('Number of validation images: ' + str(len(anns_valid)))
        
        self.anns_train = anns_train
        self.anns_valid = anns_valid
        
        self.objs_train = [(ann.img, obj) for ann in anns_train for obj in ann.objects]
        self.objs_valid = [(ann.img, obj) for ann in anns_valid for obj in ann.objects]

    def train_generator(self, batch_size, do_shuffle=True):
        
        while True:
            if do_shuffle:
                np.random.shuffle(self.objs_train)
            groups = [self.objs_train[i:i+batch_size] for i in range(0, len(self.objs_train), batch_size)]
            for group in groups:
                images, labels = [], []
                for (img, obj) in group:
                    images.append(img)
                    #images.append(apply_histogram_equalization(random_data_augmentation(load_geoimage(filename)), method='clahe'))
                    #images.append(load_geoimage(filename))
                    probabilities = np.zeros(len(self.categories))
                    probabilities[list(self.categories.values()).index(obj.category)] = 1
                    labels.append(probabilities)
                images = np.array(images).astype(np.float32)
                labels = np.array(labels).astype(np.float32)
                #weights = np.array([class_weights_dict[np.argmax(label)] for label in labels]).astype(np.float32)

                yield images, labels#, weights
                
    def train_data(self, batch_size):
        steps = math.ceil(len(self.objs_train)/batch_size)
        return self.train_generator(batch_size=batch_size), steps
                
    def valid_generator(self, batch_size, do_shuffle=False):
        while True:
            if do_shuffle:
                np.random.shuffle(self.objs_valid)
            groups = [self.objs_valid[i:i+batch_size] for i in range(0, len(self.objs_valid), batch_size)]
            for group in groups:
                images, labels = [], []
                for (img, obj) in group:
                    # Load image
                    images.append(img)
                    #images.append(apply_histogram_equalization(random_data_augmentation(load_geoimage(filename)), method='clahe'))
                    #images.append(load_geoimage(filename))
                    probabilities = np.zeros(len(self.categories))
                    probabilities[list(self.categories.values()).index(obj.category)] = 1
                    labels.append(probabilities)
                images = np.array(images).astype(np.float32)
                labels = np.array(labels).astype(np.float32)
                #weights = np.array([class_weights_dict[np.argmax(label)] for label in labels]).astype(np.float32)

                yield images, labels#, weights
                
    def valid_data(self, batch_size):
        steps = math.ceil(len(self.objs_valid)/batch_size)
        return self.valid_generator(batch_size=batch_size), steps

"""

# Generate the list of objects from annotations




# Generators
train_generator = generator_images(objs_train, batch_size, do_shuffle=True)
valid_generator = generator_images(objs_valid, batch_size, do_shuffle=False)


"""