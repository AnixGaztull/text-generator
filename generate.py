import re
import argparse
import pickle
import numpy


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
        if not self.__language_model:
            return 'Ошибка, заполните модель текстами'

        output_text = []
        key_array = numpy.array(list(self.__language_model.keys()))

        if start_word:
            output_text += start_word.split()
            last_word = output_text[-1]
        else:
            last_word = numpy.random.choice(key_array)
            output_text.append(last_word)

        lenght_output_text = len(output_text)
        for j in range(len_text - lenght_output_text):
            if last_word not in key_array or not self.__language_model[last_word]:
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
            probability_data = list(map(lambda x: word_data[x][0] / count_word, keys_word_data))
            self.__language_model[i] = [keys_word_data, probability_data]

    def __getstate__(self):
        return self.__language_model

    def __setstate__(self, state):
        self.__language_model = state


p = argparse.ArgumentParser(description='generate')
p.add_argument('--model', type=str, default='', help='the path to the file from which the model is loaded')
p.add_argument('--prefix', type=str, help='The beginning of a sentence')
p.add_argument('--length', type=str, default='', help='sentence length')

args = p.parse_args()

try:
    with open(args.model, "rb") as fp:
        m = pickle.load(fp)

        seed = numpy.random.randint(0, 2 ** 32 + 1)

        print(m.generate(args.prefix, seed, int(args.length)))
except FileNotFoundError:
    print('Ошибка. Введите правильный адрес модели')
