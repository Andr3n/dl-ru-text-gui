import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
from torchvision.transforms import ToTensor
from super_image import PanModel, ImageLoader
from PIL import Image

recognizer = None
generator = None

NUM_CLS = 33
Z_DIM = 100

MODELS_DICT = {
    'word_r': 'models/word_recognition.pt',
    'letter_r': 'models/letter_recognition.pt',
    'letter_g': 'models/letter_generator.pth'
}

RU_ALPHABET = {
    'А': 0, #
    'Б': 1, #
    'В': 26, #
    'Г': 7, #
    'Д': 3, #
    'Е': 4, #
    'Ж': 32, #
    'З': 31, #
    'И': 8, #
    'Й': 9, #
    'К': 10, #
    'Л': 12, #
    'М': 13, #
    'Н': 15, #
    'О': 16, #
    'П': 17, #
    'Р': 18, #
    'С': 19, #
    'Т': 22, #
    'У': 25, #
    'Ф': 6, #
    'Х': 11, #
    'Ц': 23, #
    'Ч': 2, #
    'Ш': 20, #
    'Щ': 21, #
    'Ъ': 24, #
    'Ы': 27, # 
    'Ь': 14, #
    'Э': 5, #
    'Ю': 30, #
    'Я': 28, #
    'Ё': 29 #
}

RU_ALPHABET_DICT = {
    0: 'а', 
    1: 'б', 
    2: 'ч', 
    3: 'д', 
    4: 'е', 
    5: 'э', 
    6: 'ф', 
    7: 'г', 
    8: 'и', 
    9: 'й', 
    10: 'к', 
    11: 'х', 
    12: 'л', 
    13: 'м', 
    14: 'ь', 
    15: 'н', 
    16: 'о', 
    17: 'п', 
    18: 'р', 
    19: 'с', 
    20: 'ш', 
    21: 'щ', 
    22: 'т', 
    23: 'ц', 
    24: 'ъ', 
    25: 'у', 
    26: 'в', 
    27: 'ы', 
    28: 'я', 
    29: 'ё', 
    30: 'ю', 
    31: 'з', 
    32: 'ж'
}

RU_WORDS_DICT = {
    0: 'a', 
    1: 'bez', 
    2: "bol'shoj", 
    3: 'by', 
    4: "byt'", 
    5: 'chelovek', 
    6: 'chto', 
    7: 'chtoby', 
    8: 'dazhe', 
    9: 'delo', 
    10: "den'", 
    11: 'dlja', 
    12: 'do', 
    13: 'dolzhen', 
    14: 'drugoj', 
    15: 'dva', 
    16: 'ee', 
    17: 'ego', 
    18: 'esche', 
    19: 'esli', 
    20: 'eto', 
    21: 'etot', 
    22: 'gde', 
    23: 'god', 
    24: "govorit'", 
    25: "hotet'", 
    26: 'i', 
    27: 'idti', 
    28: 'ih', 
    29: 'ili', 
    30: "imet'", 
    31: 'iz', 
    32: 'ja', 
    33: 'k', 
    34: 'kak', 
    35: 'kakoj', 
    36: 'kogda', 
    37: 'kotoryj', 
    38: 'kto', 
    39: 'li', 
    40: 'mesto', 
    41: "moch'", 
    42: 'moj', 
    43: 'mozhno', 
    44: 'my', 
    45: 'na', 
    46: 'nado', 
    47: 'nash', 
    48: 'ne', 
    49: 'nichto', 
    50: 'no', 
    51: 'novyj', 
    52: 'nu', 
    53: 'o', 
    54: "ochen'", 
    55: 'odin', 
    56: 'on', 
    57: 'ona', 
    58: 'oni', 
    59: 'ot', 
    60: 'pervyj', 
    61: 'po', 
    62: 'pod', 
    63: 'posle', 
    64: 'potom', 
    65: 'pri', 
    66: 'rabota', 
    67: 'raz', 
    68: 'ruka', 
    69: 's', 
    70: 'sam', 
    71: 'samyj', 
    72: 'sebya', 
    73: 'sejchas', 
    74: "skazat'", 
    75: 'slovo', 
    76: 'so', 
    77: "stat'", 
    78: 'svoj', 
    79: 'tak', 
    80: 'takoj', 
    81: 'tam', 
    82: 'to', 
    83: "tol'ko", 
    84: 'tot', 
    85: 'tut', 
    86: 'ty', 
    87: 'u', 
    88: 'uzhe', 
    89: 'v', 
    90: "ves'", 
    91: 'vo', 
    92: 'vot', 
    93: 'vremja', 
    94: 'vse', 
    95: 'vy', 
    96: 'za', 
    97: 'zhe', 
    98: "zhizn'", 
    99: "znat'"
}


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, num_classes=10, label_embed_size=5, channels=3, conv_dim=64):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embed_size)
        self.tconv1 = conv_block(z_dim + label_embed_size, conv_dim * 4, pad=0, transpose=True)
        self.tconv2 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.tconv3 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.tconv4 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        x = x.reshape([x.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])
        x = torch.cat((x, label_embed), dim=1)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = torch.tanh(self.tconv4(x))
        return x

def initialize_generator(model_variant=None):
    if model_variant == 'letter' or model_variant is None:
        model_path = MODELS_DICT['letter_g']

    generator = Generator(z_dim=100, num_classes=33, label_embed_size=6, channels=3, conv_dim=64)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator

def initialize_recognizer(model_variant=None):
    if model_variant == 'letter' or model_variant is None:
        model_path = MODELS_DICT['letter_r']
        NUM_CLS = 33
    elif model_variant == 'word':
        model_path = MODELS_DICT['word_r']
        NUM_CLS = 100

    recognizer = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    in_features = recognizer.classifier[-1].in_features
    recognizer.classifier[-1] = nn.Linear(in_features, NUM_CLS)
    recognizer.load_state_dict(torch.load(model_path))
    recognizer.eval()
    return recognizer

def generate_letter(generator, label):
    
    z = torch.randn(1, Z_DIM)

    fixed_label = torch.Tensor([RU_ALPHABET[label]]).long()
    fake_imgs = generator(z, fixed_label)
    fake_imgs = (fake_imgs + 1) / 2
    fake_imgs_ = vutils.make_grid(fake_imgs, normalize=False, nrow=1)

    img_path = os.path.join('models', 'gen_image.png')
    vutils.save_image(fake_imgs_, img_path)
    model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=3) 
    inputs = ImageLoader.load_image(Image.open(img_path))
    preds = model(inputs)
    ImageLoader.save_image(preds, img_path)

    return img_path


def recognize_letter(recognizer, img_path):
    
    image = Image.open(img_path)
    to_tensor = ToTensor()
    image_tensor = to_tensor(image)

    with torch.no_grad():
        output = recognizer(image_tensor.unsqueeze(0))

    predicted_index = torch.argmax(output).tolist()
    predicted_class = RU_ALPHABET_DICT[predicted_index].upper()

    return predicted_class

def recognize_word(recognizer, img_path):
    
    image = Image.open(img_path)
    to_tensor = ToTensor()
    image_tensor = to_tensor(image)

    with torch.no_grad():
        output = recognizer(image_tensor.unsqueeze(0))

    predicted_index = torch.argmax(output).tolist()
    predicted_class = RU_WORDS_DICT[predicted_index].upper()

    return predicted_class

generator = initialize_generator()

recognizer = initialize_recognizer()