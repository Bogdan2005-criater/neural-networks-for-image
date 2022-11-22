import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision.transforms import transforms
from tqdm import tqdm

## Открытие файла##


## Открытие файла##


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


class NoisyCleanDataset(Dataset):
    def __init__(self, noisy_path, images, clean_path=None, transforms=None):
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return (len(self.images))

    def __getitem__(self, i):
        noisy_image = cv2.imread(f"{self.noisy_path}/{self.images[i]}")
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

        if self.transforms:
            noisy_image = self.transforms(noisy_image)

        if self.clean_path is not None:
            clean_image = cv2.imread(f"{self.clean_path}/{self.images[i]}")
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
            clean_image = self.transforms(clean_image)
            return (noisy_image, clean_image, self.images[i])
        else:
            return (noisy_image, self.images[i])


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((402, 402)),
    transforms.ToTensor(),
])


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = nn.functional.interpolate(encoded, scale_factor=2)
        decoded = self.decoder(decoded)
        return decoded


class MyModel():
    def __init__(self, Dataset, Model, transforms):
        self.Dataset = Dataset
        self.model = Model().to(device)
        self.transform = transforms

    def load_weights(self, path):
        if device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def show_info(self):
        print(summary(self.model, (1, 402, 402)))

    def fit(self, n_epochs, noisy_path, clean_path, train_imgs, test_imgs):

        train_data = self.Dataset(noisy_path, train_imgs, clean_path, self.transform)
        val_data = self.Dataset(noisy_path, test_imgs, clean_path, self.transform)

        trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
        valloader = DataLoader(val_data, batch_size=4, shuffle=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            verbose=True
        )

        self.model.train()
        self.train_loss = []
        self.val_loss = []
        running_loss = 0.0

        for epoch in range(n_epochs):
            self.model.train()
            for i, data in enumerate(trainloader):
                noisy_img = data[0]
                clean_img = data[1]
                noisy_img = noisy_img.to(device)
                clean_img = clean_img.to(device)
                optimizer.zero_grad()
                outputs = self.model(noisy_img)
                loss = criterion(outputs, clean_img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    print(f'Epoch {epoch + 1} batch {i}: Loss {loss.item() / 4}')
            self.train_loss.append(running_loss / len(trainloader.dataset))
            print('Validation ...')
            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(enumerate(valloader), total=int(len(val_data) / valloader.batch_size)):
                    noisy_img = data[0]
                    clean_img = data[1]
                    noisy_img = noisy_img.to(device)
                    clean_img = clean_img.to(device)
                    outputs = self.model(noisy_img)
                    loss = criterion(outputs, clean_img)
                    running_loss += loss.item()
                current_val_loss = running_loss / len(valloader.dataset)
                self.val_loss.append(current_val_loss)
                print(f"Val Loss: {current_val_loss:.5f}")

    def predict(self, img):
        os.makedirs('outputs', exist_ok=True)
        self.model.eval()
        if type(img) == str:
            if os.path.isfile(img):
                filename = os.path.basename(img)
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = self.transform(img).to(device)
                img = self.model(img)
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(f'outputs/{filename}', img * 255)
            else:
                images = os.listdir(img)
                predictDataset = self.Dataset(img, images, transforms=self.transform)
                predictDataloader = DataLoader(predictDataset, batch_size=4, shuffle=False)
                with torch.no_grad():
                    for i, data in tqdm(enumerate(predictDataloader),
                                        total=int(len(predictDataset) / predictDataloader.batch_size)):
                        noisy_img = data[0]
                        noisy_img = noisy_img.to(device)
                        outputs = self.model(noisy_img)
                        for im, image_name in zip(outputs, data[1]):
                            im = im.detach().cpu().permute(1, 2, 0).numpy()
                            cv2.imwrite(f'outputs/{image_name}', im * 255)
        if type(img) == np.ndarray:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.transform(img).to(device)
            img = self.model(img)
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            cv2.imwrite('outputs/cleaned_img.jpg', img * 255)


AutoEncoder = MyModel(NoisyCleanDataset, Autoencoder, transform)

AutoEncoder.load_weights('model2.pth')


## Обработка изображения ##

def rebuilding(path_im, num):
    from PIL import Image
    # Loading the image
    image = Image.open(path_im)
    # Specifying the RGB mode to the image
    image = image.convert('RGB')
    # Converting an image from PNG to JPG format
    old_image = image

    h, w = old_image.size
    h_new = h
    w_new = w
    if w_new % 402 != 0:  # переопределение картинки по пикселям
        k_w = math.ceil(w / 402)
        w_new = k_w * 402
    if h_new % 402 != 0:  # переопределение картинки по пикселям
        k_h = math.ceil(h / 402)
        h_new = k_h * 402
    k = k_h * k_w
    # print(k_h, k_w)
    backG = Image.new('RGB', (h_new, w_new), 'white')
    # backG.show()
    area = (0, 0, h_new, w_new)
    backG.save('backG.jpg')
    # print(backG.size)
    # cropped_img = backG.crop(area)
    # cropped_img.show()
    backG.paste(old_image, (0, 0))
    # # print(backG.size)
    # backG.show()
    arr = [None] * k
    i_h = 0
    i_w = 0
    from PIL import Image

    if k > 1:
        h_crop = 402
        w_crop = 402
        y1 = 0
        y2 = w_crop
        i = 0
        for i in range(k):
            time_im = backG
            x1 = h_crop * i_h
            x2 = h_crop * (i_h + 1)
            i_h += 1
            time_im = time_im.crop((x1, y1, x2, y2))
            if i_h == k_h:
                i_w += 1
                y1 = w_crop * i_w
                y2 = w_crop * (i_w + 1)
                i_h = 0

            time_im.save('parts/' + str(i) + '.jpg')
            arr[i] = str(i) + '.jpg'
        AutoEncoder.predict('parts/')

        for q in arr:
            Contr = Image.open('outputs/' + q)
            Contr = Contr.crop((1, 1, 401, 401))
            Contr.save('outputs/' + q)

        backG2 = Image.new('RGB', (h_new, w_new), 'white')
        i_h = 0
        i_w = 0
        x = 0
        y = 0
        h_crop = 400
        w_crop = 400
        a1 = Image.open('outputs/1.jpg')
        i = 0
        for l in range(k):
            a = Image.open('outputs/' + arr[l])
            x2 = h_crop * (i_h + 1)
            x = h_crop * i_h
            i_h += 1
            backG2.paste(a, (x, y))
            if i_h == k_h:
                i_w += 1
                y = w_crop * i_w
                i_h = 0
        backG2 = backG2.crop((0, 0, h, w))
        backG2.save('clear_img/' + str(num) + '.jpg')
        num += 1
    if k == 1:
        backG.save('parts/img.jpg')
        AutoEncoder.predict('parts/')  # ПРОВЕРИТЬ !!!
        new1 = Image.open('parts/img.jpg')
        new1 = new1.crop((0, 0, h, w))
        new1.save('clear_img/' + str(num) + '.jpg')


import tkinter.filedialog as fd
import tkinter as tk
from tkinter import ttk

asdd = ''


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title = "bla-bla-bla"
        self.show = 0
        btn_file = tk.Button(self, text='Выбрать файл', command=self.choose_file)
        btn_dir = tk.Button(self, text='Скачать файл', command=self.download_file)
        self.lable = tk.Label()
        self.lable_num = tk.Label()
        self.succes = tk.Label(fg='#006a35')
        self.p1 = ttk.Progressbar(self, length=200, cursor='spider', mode="determinate", orient=tk.HORIZONTAL)
        btn_file.pack(padx=60, pady=10)
        self.p1.pack(padx=100, pady=10)
        self.lable_num.pack(padx=30, pady=0)
        self.lable.pack(padx=30, pady=0)
        btn_dir.pack(padx=60, pady=10)
        self.succes.pack(padx=60, pady=0)

    def choose_file(self):
        filetypes = (('Изображение', '*.jpg *.gif *.png'), ('Текстовый файл', ' *.txt'), ('Любой', ' * '))
        filename = fd.askopenfilename(title='Открыть файл', initialdir=' / ', filetypes=filetypes, multiple=True)
        import shutil
        shutil.rmtree("outputs", ignore_errors=True, )
        shutil.rmtree("parts", ignore_errors=True)
        os.mkdir('outputs')
        os.mkdir('parts')
        shutil.rmtree("clear_img", ignore_errors=True)
        os.mkdir('clear_img')
        if filename:
            num = 0
            size1 = 0
            for i in filename:
                size1 += 1
            size = (100 / size1)
            for i in filename:
                num += 1
                rebuilding(i, num)
                self.lable["text"] = i
                self.lable_num["text"] = str(num) + '/' + str(size1)
                self.p1["value"] += size
                self.update()
                time.sleep(0.1)
            self.show = 1
            self.succes["text"] = "Succesfull all"

    def download_file(self):
        if self.show == 1:
            directory = fd.askdirectory(title='Выбрать папку', initialdir='/')

            if directory:
                from zipfile import ZipFile
                import hashlib
                size = os.listdir('clear_img')
                with ZipFile(directory + '/' + hashlib.sha1(str(random.randint(0, 999999999)).encode()).hexdigest(),
                             'w') as f:
                    for i in size:
                        f.write('clear_img/' + i)
            self.show = 0
        else:
            print('Ошибка')


if __name__ == '__main__':
    app = App()
    app.mainloop()
    import shutil

    ## Обработка изображения ##
