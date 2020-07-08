from pyunit_sogou import SoGou
import os
import logging


sg = SoGou()


def download_category(category_id, save_dir):
    category_url = 'https://pinyin.sogou.com/dict/cate/index/' + str(category_id)
    urls = sg.one_classify_lexicon(category_url)
    _dir = os.path.join(save_dir, str(category_id))
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    for url in urls:
        logging.info('Download from {}', url)
        name, data = sg.download_to_text(url)
        name = name.replace(' ', '').replace('/', '_').strip().lower() + '.txt'
        f = os.path.join(_dir, name)
        logging.info('Save into {}', f)
        with open(f, mode='wt', encoding='utf8') as fout:
            for d in sorted(set(data)):
                fout.write(d + '\n')


if __name__ == "__main__":
    download_category(167, '/tmp/sogou')
