import logging
import os

from pyunit_sogou import SoGou

sg = SoGou()


def download_category(category_id, save_dir):
    category_url = 'https://pinyin.sogou.com/dict/cate/index/' + str(category_id)
    urls = sg.one_classify_lexicon(category_url)
    _dir = os.path.join(save_dir, str(category_id))
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    for url in urls:
        try:
            logging.info('Download from {}', url)
            name, data = sg.download_to_text(url)
            name = name.replace(' ', '').replace('/', '_').strip().lower() + '.txt'
            f = os.path.join(_dir, name)
            logging.info('Save into {}', f)
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
