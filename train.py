import re
import argparse
import pickle
import os
import numpy
import codecs


class Model:
    def __init__(self):
        self.__dictionary = {}
        self.__language_model = {}

    def fit(self, text):
        text = text.replace('\n', ' ')
        regex = re.compile('[^а-яА-ЯёЁ -]')
        clear_text = regex.sub('', text).lower()

        clear_text_list = clear_text.split()
        lenght_text = len(clear_text_list)

        if lenght_text < 2:
            print('Ошибка текста')
            return

        for index in range(lenght_text - 1):
            word = clear_text_list[index]
            next_word = clear_text_list[index+1]

            if word not in self.__dictionary:
                self.__dictionary[word] = [0, {}]  # 0-ой элемент - количество слов во ВСЕХ текстах

            if next_word not in self.__dictionary[word][1]:
                self.__dictionary[word][1][next_word] = 0  # считаю кол-во слов в текстах, для прав. вероятн.

            self.__dictionary[word][0] += 1
            self.__dictionary[word][1][next_word] += 1

    def generate(self, start_word, seed_text, len_text):
        numpy.random.seed(seed_text)
        if not self.__dictionary:
            return 'Ошибка, заполните модель текстами'

        output_text = []
        key_array = numpy.array(list(self.__dictionary.keys()))

        if start_word:
            output_text += start_word.split()
            last_word = output_text[-1]
        else:
            last_word = numpy.random.choice(key_array)
            output_text.append(last_word)

        lenght_output_text = len(output_text)
        for j in range(len_text - lenght_output_text):
            if last_word not in key_array or not self.__dictionary[last_word][1]:
                last_word = numpy.random.choice(key_array)
            else:
                keys_next_words_data = self.__language_model[last_word][0]
                probability_next_word = self.__language_model[last_word][1]
                numpy_keys_nw = numpy.array(keys_next_words_data)
                last_word = numpy.random.choice(numpy_keys_nw, p=probability_next_word)
            output_text.append(last_word)
        return ' '.join(output_text)

    def make_language_model(self):
        dict_keys = list(self.__dictionary.keys())
        for i in dict_keys:
            count_word = self.__dictionary[i][0]
            word_data = self.__dictionary[i][1]
            keys_word_data = list(word_data.keys())
            probability_data = list(map(lambda x: word_data[x] / count_word, keys_word_data))
            self.__language_model[i] = [keys_word_data, probability_data]

    def __getstate__(self):
        return self.__language_model

    def __setstate__(self, state):
        self.__language_model = state


p = argparse.ArgumentParser(description='train')
p.add_argument('--input-dir', type=str,  default='stdin', help='the path to the directory where '
                                                               'the collection of documents is located')
p.add_argument('--model', type=str, help='the path to the file where the model is saved')
args = p.parse_args()

m = Model()
if args.input_dir == 'stdin':
    count_texts = int(input('Введите сколько текстов хотите ввести: '))
    for num_text in range(count_texts):
        print('Введите', num_text + 1, 'текст (в конце текста введите End_text):')
        input_text = ''
        while 'End_text' not in input_text:
            new_line = input()
            input_text = input_text + new_line
        m.fit(input_text)
    print('Спасибо за тексты')
else:
    files = os.listdir(path=args.input_dir)
    for filename in files:
        with codecs.open(os.path.join(args.input_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
            input_text = f.read()
            m.fit(input_text)

m.make_language_model()


if args.model:
    path_to_save = f"{args.model}/model.pkl"
else:
    path_to_save = 'model.pkl'

with open(path_to_save, "wb") as fp:
    pickle.dump(m, fp)
