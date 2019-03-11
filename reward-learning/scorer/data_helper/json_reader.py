import json
import os
from collections import OrderedDict
import random
from nltk.tokenize import sent_tokenize
import numpy as np

from resources import RAW_DATA_DIR, DATA_DIR


def readScores():
    score_path = os.path.join(RAW_DATA_DIR, 'lqual.jsonl')
    scores = []
    id_list = []

    for line in open(score_path, 'r'):
        data = json.loads(line)
        entry = OrderedDict()
        entry['id'] = data['input']['contents']['id']
        entry['ref'] = data['input']['contents']['reference']
        entry['sys_name'] = data['input']['contents']['system']
        entry['sys_summ'] = data['input']['contents']['text']
        del data['input']
        del data['output']['_responses']
        entry['scores'] = data['output']
        scores.append(entry)
        if entry['id'] not in id_list:
            id_list.append(entry['id'])

    return scores, id_list


def readArticles():
    article_path = os.path.join(RAW_DATA_DIR, 'articles.jsonl')
    articles = []
    article_id_list = []
    for line in open(article_path, 'r'):
        data = json.loads(line)
        entry = OrderedDict()
        entry['id'] = data['id']
        entry['article'] = data['text']
        articles.append(entry)
        if entry['id'] not in article_id_list:
            article_id_list.append(entry['id'])

    return articles, article_id_list


def readReferences():
    ref_path = os.path.join(RAW_DATA_DIR, 'lqual_all.jsonl')
    refs = []
    id_list = []

    for line in open(ref_path, 'r'):
        data = json.loads(line)
        if data['id'] in id_list:
            continue
        entry = OrderedDict()
        entry['id'] = data['id']
        entry['ref'] = data['reference']
        refs.append(entry)
        id_list.append(entry['id'])

    return refs, id_list


def findIdxByID(list_of_dic, id):
    for ii in range(len(list_of_dic)):
        if list_of_dic[ii]['id'] == id:
            return ii
    return -1


def readArticleRefs(ids=None, as_dict=False):
    refs, ref_ids = readReferences()
    article_path = os.path.join(RAW_DATA_DIR, 'articles.jsonl')
    for line in open(article_path, 'r'):
        data = json.loads(line)
        if data['id'] not in ref_ids:
            continue
        idx = findIdxByID(refs, data['id'])
        assert idx != -1
        refs[idx]['article'] = data['text']

    if ids is not None:
        refs = [r for r in refs if r['id'] in ids]

    if as_dict:
        article_ref_dict = {}

        for r in refs:
            article_ref_dict.update({r['id']: {'article': r['article'], 'ref': r['ref']}})

        return article_ref_dict
    else:
        return refs


def createSortedScores():
    scores, id_list = readScores()
    scores_per_article = {}

    for s in scores:
        overall_score = s["scores"]["overall"]
        id = s["id"]

        if overall_score is None:
            continue

        if id not in scores_per_article:
            scores_per_article.update({id: []})

        scores_per_article[id].append(s)

        # if s["sys_name"] == "reference":
        #     ref_scores.append(overall_score)
        # else:
        #     other_scores.append(overall_score)

    # Sort the list of scores or remove lists with only one entry
    remove_keys = []

    for id, s in scores_per_article.items():
        if len(s) <= 1:
            remove_keys.append(id)
        else:
            scores_per_article[id] = sorted(scores_per_article[id], key=lambda k: k['scores']['overall'])

    for rk in remove_keys:
        scores_per_article.pop(rk)

    for id, s in scores_per_article.items():
        summ_id = 0
        rank = 0

        for i in range(len(s)):
            s[i]['summ_id'] = summ_id
            summ_id += 1

        for i in range(len(s) - 1):
            # Detect equality and check if left hand summary is the reference summary
            if s[i]['scores']['overall'] == s[i + 1]['scores']['overall'] and s[i]['sys_name'].lower() == "reference":
                # In this case, swap the order!
                tmp = s[i]
                s[i] = s[i + 1]
                s[i + 1] = tmp

            s[i]['rank'] = rank

            if s[i]['scores']['overall'] != s[i + 1]['scores']['overall']:
                rank += 1

        s[-1]['rank'] = rank

    json.dump(scores_per_article, open(os.path.join(DATA_DIR, "sorted_scores.json"), "w"))


def readSortedScores():
    sorted_scores_path = os.path.join(DATA_DIR, 'sorted_scores.json')

    if not os.path.exists(sorted_scores_path):
        createSortedScores()

    return json.load(open(sorted_scores_path, "r"))


if __name__ == '__main__':
    scores, id_list = readScores()
    article_ref = readArticleRefs()

    print('\nscore length: {}'.format(len(scores)))
    print('unique id num in scores: {}'.format(len(id_list)))
    # entry = random.choice(scores)
    # for item in entry:
    #    print('{} : {}'.format(item,entry[item]))

    ref_scores = []
    other_scores = []
    none_scores = 0

    for s in scores:
        overall_score = s["scores"]["overall"]
        id = s["id"]

        if overall_score is None:
            none_scores += 1
            continue

        if s["sys_name"] == "reference":
            ref_scores.append(overall_score)
        else:
            other_scores.append(overall_score)

    print("ref mean ", np.mean(ref_scores))
    print("ref std ", np.std(ref_scores))
    print("other mean ", np.mean(other_scores))
    print("other std ", np.std(other_scores))
    print("min ", np.min(ref_scores + other_scores))
    print("max ", np.max(ref_scores + other_scores))
    print("all mean ", np.mean(ref_scores + other_scores))
    print("none scores ", none_scores)

    exit(0)

    print('\nref length : {}'.format(len(article_ref)))
    entry = random.choice(article_ref)
    for item in entry:
        print('{} : {}'.format(item, entry[item]))

    ### get the avg. number of sentences in refs. and in articles.
    ref_sent_nums = []
    ref_token_nums = []
    art_sent_nums = []
    art_token_nums = []
    for entry in article_ref:
        ref = entry['ref']
        article = entry['article']
        ref_sent_nums.append(len(sent_tokenize(ref)))
        ref_token_nums.append(len(ref.split(' ')))
        art_sent_nums.append(len(sent_tokenize(article)))
        art_token_nums.append(len(article.split(' ')))

    print('\n')
    print('ref sent num: max {}, min {}, mean {}, std {}'.format(
        np.max(ref_sent_nums), np.min(ref_sent_nums), np.mean(ref_sent_nums), np.std(ref_sent_nums)
    ))
    print('ref token num: max {}, min {}, mean {}, std {}'.format(
        np.max(ref_token_nums), np.min(ref_token_nums), np.mean(ref_token_nums), np.std(ref_token_nums)
    ))
    print('article sent num: max {}, min {}, mean {}, std {}'.format(
        np.max(art_sent_nums), np.min(art_sent_nums), np.mean(art_sent_nums), np.std(art_sent_nums)
    ))
    print('article token num: max {}, min {}, mean {}, std {}'.format(
        np.max(art_token_nums), np.min(art_token_nums), np.mean(art_token_nums), np.std(art_token_nums)
    ))
