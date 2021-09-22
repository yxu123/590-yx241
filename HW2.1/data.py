import json
import matplotlib.pyplot as plt
import numpy as np
import random

class Data:
    def __init__(self,file_name):
        with open(file_name,'r') as load_f:
            self.load_dict = json.load(load_f)
        
        data_len = len(self.load_dict['x'])

        npx = np.array(self.load_dict['x'])
        npy = np.array(self.load_dict['y'])
        npis_adult = np.array(self.load_dict['is_adult'])



        # Normalize
        npx = (npx - npx.mean())/npx.std()
        npy = (npy - npy.mean())/npy.std()
        
        test_len = int(0.2 * data_len)
        validation_len = int(0.2 * data_len)

        index_set = set()
        for r in range(0, data_len):
            index_set.add(r)
        test_set_indexs = random.sample(index_set, test_len)

        index_set.difference_update(test_set_indexs)
        sorted(index_set)
        validation_set_indexs = random.sample(index_set, validation_len)

        index_set.difference_update(validation_set_indexs)
        sorted(index_set)
        train_set_indexs = index_set
        self.train_x = np.array(self.get_data_set(npx, train_set_indexs))

        self.train_y = np.array(self.get_data_set(npy, train_set_indexs))

        self.mini_batch_x, self.mini_batch_y = self.get_mini_batch_set()
        self.train_is_adult = np.array(self.get_data_set(npis_adult, train_set_indexs))

        self.test_x = np.array(self.get_data_set(npx, test_set_indexs))

        self.test_y = np.array(self.get_data_set(npy, test_set_indexs))
        self.test_is_adult = np.array(self.get_data_set(npis_adult, test_set_indexs))

        self.validation_x = np.array(self.get_data_set(npx, validation_set_indexs))

        self.validation_y = np.array(self.get_data_set(npy, validation_set_indexs))
        self.validation_is_adult = np.array(self.get_data_set(npis_adult, validation_set_indexs))

        print(self.validation_x)
        print(self.validation_y)


    def get_data_set(self, orig_set, select_set):
        ret = []
        for i in select_set:
            ret.append(orig_set.tolist()[i])

        return ret

    def get_mini_batch_set(self):
        retx = []
        rety = []
        for i in range(len(self.train_x)):
            if i % 2 == 0:
                retx.append(self.train_x.tolist()[i])
                rety.append(self.train_y.tolist()[i])

        return np.array(retx), np.array(rety)

    def show_raw(self):
        fig, ax = plt.subplots()
        ax.plot(self.load_dict["x"],self.load_dict["y"],'-')
        plt.show()
    
    def show_is_adult(self):
        fig, ax = plt.subplots()
        ax.plot(self.load_dict["x"],self.load_dict["is_adult"],'o')
        plt.show()



if __name__ == '__main__':
    data = Data("../DATA/weight.json")

    print(data.train_x[:100])
    print(data.train_x[:50])
    # data.show_raw()
    #
    # data.show_is_adult()

