import logging
import math
import os
import re
import struct
from urllib.parse import quote, unquote

import requests
from fake_useragent import UserAgent


class SoGou:

    def __init__(self):
        self.ua = UserAgent()

    def download_all_lexicon(self, start=0, end=629):
        for page in range(start, end):
            url = f'https://pinyin.sogou.com/dict/cate/index/{page}'
            data = self.one_classify_lexicon(url)
            yield data

    def one_classify_lexicon(self, url):
        response = requests.get(url=url, headers={'User-Agent': self.ua.random}).text
        max_page = re.search('分类下共有(.+)个词库', response).group(1)
        urls = re.findall('http://download.pinyin.sogou.com/dict/download_cell.php?.+"', response)
        for page in range(2, math.ceil(int(max_page) / 10) + 1):
            response = requests.get(url=url + f'/default/{page}', headers={'User-Agent': self.ua.random}).text
            urls.extend(re.findall('http://download.pinyin.sogou.com/dict/download_cell.php?.+"', response))
        return map(lambda x: unquote(x[:-1]), urls)

    def search_name_lexicon(self, search_name):
        url_word = quote(search_name.encode('GBK'))  # 将中文字转化为URL链接。注意搜狗将中文字进行的GBK编码。而不是UTF-8
        urls = 'https://pinyin.sogou.com/dict/search/search_list/%s/normal/' % url_word  # 搜索链接
        response = requests.get(url=urls, headers={'User-Agent': self.ua.random}).text
        max_page = re.search('共有(.+)个搜索结果', response).group(1)
        m = re.findall('//pinyin.sogou.com/d/dict/download_cell.php?.+"', response)
        for page in range(2, math.ceil(int(max_page) / 10) + 1):
            response = requests.get(url=urls + str(page), headers={'User-Agent': self.ua.random}).text
            m.extend(re.findall('//pinyin.sogou.com/d/dict/download_cell.php?.+"', response))  # 将匹配到的下载链接装到链表中
        return map(lambda x: 'https:' + x[:-1], m)

    @staticmethod
    def _url_to_chinese(url):
        url_word = unquote(url)
        match = re.search('name=(.+)&|name=(.+)', url_word)
        name = match.group(1) if match.group(1) else match.group(2)
        return name, url_word

    def download_to_text(self, url_word):
        name, url_word = self._url_to_chinese(url_word)
        response = requests.get(url=url_word, headers={'User-Agent': self.ua.random})
        return name, self._to_txt(response.content)

    def download_to_scel(self, url_word, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        name, url_word = self._url_to_chinese(url_word)
        path = os.path.abspath(save_dir) + os.sep + name + '.scel'
        response = requests.get(url=url_word, headers={'User-Agent': self.ua.random})
        with open(path, mode='wb')as f:
            f.write(response.content)
        return path

    @staticmethod
    def _to_txt(data):
        ls_word = []  # 转化搜狗为UTF-8格式内容
        w = ''  # 每一个词条
        for i in range(0, len(data), 2):
            x = data[i:i + 2]  # 搜狗的UTF-8编码是两个字节
            t = struct.unpack('H', x)[0]  # 将其转化为无符号的短整形
            if 19968 < t < 40959 or t == 10:  # 判断是否是中文字符。10表示的是换行
                if t != 10:  # 不换行放在单个词条
                    w += chr(t)
                elif t == 10 and len(w):  # 换行且不等于空
                    ls_word.append(w)
                    w = ''
        return ls_word[1:]  # 第一行是注释。不需要，去除第一行

    def download_to_txt(self, url, txt_dir):
        if not os.path.isdir(txt_dir):
            os.makedirs(txt_dir)
        name, text = self.download_to_text(url)
        path = os.path.abspath(txt_dir) + os.sep + name + '.txt'
        with open(path, 'w', encoding='utf-8')as f:
            f.writelines(t + '\n' for t in text)
        return path


sg = SoGou()


def download_category(category_id, save_dir):
    category_url = 'https://pinyin.sogou.com/dict/cate/index/' + str(category_id)
    urls = sg.one_classify_lexicon(category_url)
    _dir = os.path.join(save_dir, str(category_id))
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    for url in urls:
        try:
            logging.info('Download from %s', url)
            name, data = sg.download_to_text(url)
            name = name.replace(' ', '').replace('/', '_').strip().lower() + '.txt'
            f = os.path.join(_dir, name)
            logging.info('Save into %s', f)
            with open(f, mode='wt', encoding='utf8') as fout:
                for d in sorted(set(data)):
                    fout.write(d + '\n')
        except Exception as e:
            print(e)
            continue


def download_all_category(save_dir):
    for i in range(653):  # sogou has 653 categories in total
        download_category(i, save_dir)


def collect(save_dir, output_file, maxlen=6):
    files = []

    for d in os.listdir(save_dir):
        d = os.path.join(save_dir, d)
        if os.path.isdir(d):
            for f in os.listdir(d):
                files.append(os.path.join(d, f))

    files = sorted(set(files))
    print('Collect from {} files in total'.format(len(files)))
    vocabs = set()

    for f in files:
        if not os.path.exists(f):
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\n').strip()
                if not line:
                    continue
                if len(line) > maxlen:
                    continue
                vocabs.add(line)

    output_file = os.path.join(save_dir, output_file)
    with open(output_file, mode='wt', encoding='utf8') as fout:
        for v in sorted(vocabs):
            fout.write(v + '\n')
