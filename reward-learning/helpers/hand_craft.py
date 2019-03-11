import sys
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../')

import argparse
import logging
import torch
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict
from nltk.corpus import stopwords
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import pickle
from tqdm import tqdm

from models.phis.phi_api import PhiAPI
from resources import LANGUAGE,BASE_DIR,SWAP_FEATURES_DIR,TEST_FEATURES_DIR
from helpers.data_helpers import sent2stokens_wostop,extract_ngrams2
from scorer.data_helper.json_reader import readArticleRefs
from scorer.data_helper.json_reader import readScores, readArticleRefs, readSortedScores


class HAND_CRAFT(PhiAPI):
    def __init__(self,wanted_features=['js']):
        # TODO If any arguments are passed to the constructor
        # self.param1 = param1
        # self.ngrams = ngrams
        self.wanted_features = wanted_features
        self.stopwords = set(stopwords.words(LANGUAGE))
        self.stemmer = PorterStemmer()
        pass

    @staticmethod
    def phi_options(phi_options=None):
        # TODO If any arguments are required they should be added here
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        # phi_options.add_argument("--param1", action="store", default=42, type=int, help="Just a parameter")
        # phi_options.add_argument("--ngrams", action="append", default=None, type=int, help="A list of ngrams")

        return phi_options

    def getWordDistribution(self,text,vocab,nlist):
        if vocab is None:
            vocab_list = []
            build_vocab = True
        else:
            vocab_list = vocab
            build_vocab = False

        dic = OrderedDict((el,0) for el in vocab_list)

        ngrams = []
        for n in nlist:
            if n == 1:
                ngrams.extend(sent2stokens_wostop(text,self.stemmer,self.stopwords,LANGUAGE))
            else:
                ngrams.extend(extract_ngrams2([text],self.stemmer,LANGUAGE,n))

        for ww in ngrams:
            if ww in dic:
                dic[ww] = dic[ww]+1
            elif build_vocab:
                dic[ww] = 1

        return list(dic.keys()), list(dic.values())

    def jsd(self, p, q, base=np.e):
        ## convert to np.array
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        if p.sum() == 0 or q.sum() == 0:
            return -1.

        p, q = p/p.sum(), q/q.sum()
        m = 1./2*(p + q)
        return scipy.stats.entropy(p,m, base=base)/2. +  scipy.stats.entropy(q, m, base=base)/2.


    def getJS(self,inputs,type):
        vocab, doc_word_dist = self.getWordDistribution(inputs['article'],None,[1,2])
        _, sum_word_dist = self.getWordDistribution(inputs[type],vocab,[1,2])
        return [self.jsd(sum_word_dist,doc_word_dist)]

    def getRouge(self,inputs,fname,type):
        nlist = fname.split('_')[1:]
        nlist = [int(ii) for ii in nlist]
        vocab, doc_word_dist = self.getWordDistribution(inputs['article'],None,nlist)
        _, sum_word_dist = self.getWordDistribution(inputs[type],vocab,nlist)
        rscore = np.sum([1 for cc in sum_word_dist if cc is not 0])*1./len(sum_word_dist)
        return [rscore]

    def getTFIDF(self,inputs,fname,type):
        corpus = [inputs['article'],inputs[type]]
        vv = TfidfVectorizer(stop_words=LANGUAGE.lower())
        vectors = vv.fit_transform(corpus)
        rewards = []

        if 'cos' in fname:
            cos = cosine_similarity(vectors[0,:],vectors[1,:])[0][0]
            rewards.append(cos)
        if 'avg' in fname:
            avg_value = np.mean(vectors[1])
            rewards.append(avg_value)

        return rewards

    def getRedundancy(self,inputs,fname,type):
        nlist = fname.split('_')[1:]
        nlist = [int(ii) for ii in nlist]
        rewards = []
        for n in nlist:
            summ_ngram = list(extract_ngrams2([inputs[type]],self.stemmer,LANGUAGE.lower(),n))
            if len(summ_ngram) == 0:
                rewards.append(0.)
            else:
                rewards.append(1.-len(set(summ_ngram))/float(len(summ_ngram)))
        return rewards


    def __call__(self, inputs, type, tensor=False):
        encoding = []

        for feature in self.wanted_features:
            if feature == 'js':
                ff = self.getJS(inputs,type)
            elif 'rouge' in feature :
                ff = self.getRouge(inputs,feature,type)
            elif 'tfidf' in feature:
                ff = self.getTFIDF(inputs,feature,type)
            elif 'redundancy' in feature:
                ff = self.getRedundancy(inputs,feature,type)
            encoding.extend(ff)

        if tensor:
            return torch.FloatTensor(encoding)
        else:
            return np.array(encoding)

        '''
        for k, v in inputs.items():
            if "summary" in k:
                # TODO Here the tf-idf feature vector should be produced and turned into a tensor
                #  (instead of a random tensor)
                batch = torch.cat([torch.randn(1, 300) for _ in v], dim=0)
                encodings.update({k: batch})
            elif "article" == k:
                # TODO Encode the article
                batch = torch.cat([torch.randn(1, 300) for _ in v], dim=0)
                encodings.update({k: batch})

        return encodings
        '''

    @property
    def output_dim(self):
        # TODO This number reflects the length of each feature vector
        return 300

    @property
    def trainable(self):
        return False

    @classmethod
    def init(cls, input_dim, phi_args_list=None, *args, **kwargs):
        # The parameter input_dim is the vector length of the embeddings, in case of tf*idf no embeddings are used
        phi_args, unparsed_args = HAND_CRAFT.phi_options().parse_known_args(phi_args_list)

        logging.info("Phi arguments: {}".format(phi_args))
        return HAND_CRAFT(), phi_args, unparsed_args

        # TODO In case any arguments have been parsed, they should be given to the class constructor
        # return TF_IDF(phi_args.param1, phi_args.ngrams), phi_args, unparsed_args


init = HAND_CRAFT.init


def simpleTest():
    inputs = {
        'ref_summary': 'marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports. journalists at bild and paris match are "very confident" the video clip is real, an editor says. andreas lubitz had informed his lufthansa training school of an episode of severe depression, airline says.',
        'sys_summary': 'he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . earlier , a spokesman for the prosecutor \'s office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot \'s license . marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports.',
        'article': 'marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . '' robin \'s comments follow claims by two magazines , german daily bild and french paris match , of a cell phone video showing the harrowing final seconds from on board germanwings flight 9525 as it crashed into the french alps . all 150 on board were killed . paris match and bild reported that the video was recovered from a phone at the wreckage site . the two publications described the supposed video , but did not post it on their websites . the publications said that they watched the video , which was found by a source close to the investigation . `` one can hear cries of \' my god \' in several languages , '' paris match reported . `` metallic banging can also be heard more than three times , perhaps of the pilot trying to open the cockpit door with a heavy object . towards the end , after a heavy shake , stronger than the others , the screaming intensifies . then nothing . '' `` it is a very disturbing scene , '' said julian reichelt , editor-in-chief of bild online . an official with france \'s accident investigation agency , the bea , said the agency is not aware of any such video . lt. col. jean-marc menichini , a french gendarmerie spokesman in charge of communications on rescue efforts around the germanwings crash site , told cnn that the reports were `` completely wrong '' and `` unwarranted . '' cell phones have been collected at the site , he said , but that they `` had n\'t been exploited yet . '' menichini said he believed the cell phones would need to be sent to the criminal research institute in rosny sous-bois , near paris , in order to be analyzed by specialized technicians working hand-in-hand with investigators . but none of the cell phones found so far have been sent to the institute , menichini said . asked whether staff involved in the search could have leaked a memory card to the media , menichini answered with a categorical `` no . '' reichelt told `` erin burnett : outfront '' that he had watched the video and stood by the report , saying bild and paris match are `` very confident '' that the clip is real . he noted that investigators only revealed they \'d recovered cell phones from the crash site after bild and paris match published their reports . `` that is something we did not know before . ... overall we can say many things of the investigation were n\'t revealed by the investigation at the beginning , '' he said . what was mental state of germanwings co-pilot ? german airline lufthansa confirmed tuesday that co-pilot andreas lubitz had battled depression years before he took the controls of germanwings flight 9525 , which he \'s accused of deliberately crashing last week in the french alps . lubitz told his lufthansa flight training school in 2009 that he had a `` previous episode of severe depression , '' the airline said tuesday . email correspondence between lubitz and the school discovered in an internal investigation , lufthansa said , included medical documents he submitted in connection with resuming his flight training . the announcement indicates that lufthansa , the parent company of germanwings , knew of lubitz \'s battle with depression , allowed him to continue training and ultimately put him in the cockpit . lufthansa , whose ceo carsten spohr previously said lubitz was 100 % fit to fly , described its statement tuesday as a `` swift and seamless clarification '' and said it was sharing the information and documents -- including training and medical records -- with public prosecutors . spohr traveled to the crash site wednesday , where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside . he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . menichini told cnn late tuesday that no visible human remains were left at the site but recovery teams would keep searching . french president francois hollande , speaking tuesday , said that it should be possible to identify all the victims using dna analysis by the end of the week , sooner than authorities had previously suggested . in the meantime , the recovery of the victims \' personal belongings will start wednesday , menichini said . among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board . check out the latest from our correspondents . the details about lubitz \'s correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and lubitz \'s possible motive for downing the jet . a lufthansa spokesperson told cnn on tuesday that lubitz had a valid medical certificate , had passed all his examinations and `` held all the licenses required . '' earlier , a spokesman for the prosecutor \'s office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot \'s license . kumpa emphasized there \'s no evidence suggesting lubitz was suicidal or acting aggressively before the crash . investigators are looking into whether lubitz feared his medical condition would cause him to lose his pilot \'s license , a european government official briefed on the investigation told cnn on tuesday . while flying was `` a big part of his life , '' the source said , it \'s only one theory being considered . another source , a law enforcement official briefed on the investigation , also told cnn that authorities believe the primary motive for lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems . lubitz \'s girlfriend told investigators he had seen an eye doctor and a neuropsychologist , both of whom deemed him unfit to work recently and concluded he had psychological issues , the european government official said . but no matter what details emerge about his previous mental health struggles , there \'s more to the story , said brian russell , a forensic psychologist . `` psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they were n\'t going to keep doing their job and they \'re upset about that and so they \'re suicidal , '' he said . `` but there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person \'s problems . '' germanwings crash compensation : what we know . who was the captain of germanwings flight 9525 ? cnn \'s margot haddad reported from marseille and pamela brown from dusseldorf , while laura smith-spark wrote from london . cnn \'s frederik pleitgen , pamela boykoff , antonia mortensen , sandrine amiel and anna-maja rappard contributed to this report .'
    }


    tf_idf = HAND_CRAFT(['js', 'rouge_1', 'rouge_2', 'tfidf_cos_avg', 'redundancy_1_2'])
    encoding = tf_idf(inputs)
    print(encoding)

def writeSwapFeatures(wanted_feature,samples_file=None,debug=True):
    vectoriser = HAND_CRAFT([wanted_feature])

    articles_dict = {}
    refs_dict = {}
    article_refs = readArticleRefs()

    for entry in article_refs:
        articles_dict.update({entry["id"]: entry["article"]})
        refs_dict.update({entry["id"]: entry["ref"]})

    if samples_file is None:
        sfiles = glob.glob(os.path.join(BASE_DIR,"data", "samples", "*.p"))
    else:
        sfiles = [os.path.join(BASE_DIR,"data", "samples", samples_file)]

    for sample_file in sfiles:
    #for sample_file in [os.path.join(BASE_DIR,"data", "samples", "1_100.p")]:
        feature_dic = {}
        out_dir = os.path.join(SWAP_FEATURES_DIR,wanted_feature)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # nb_swaps, nb_samples = os.path.splitext(os.path.basename(sample_file))[0].split("_")
        samples = pickle.load(open(sample_file, "rb"))
        logging.info("Loaded swap samples from {}".format(sample_file))
        desc = 'generating {} feature for summaries in {}'.format(wanted_feature,sample_file.split('/')[-1])
        print(desc)
        cnt = 0

        for article_id, swap_dict in samples.items(): #tqdm(samples.items(),desc=desc):
            cnt += 1
            if debug and cnt >= 10: break
            feature_swap_dict = {}
            feature_dic[article_id] = feature_swap_dict

            article = articles_dict[article_id]
            ref = refs_dict[article_id]

            feature_swap_dict['ref'] = vectoriser({'article':article,'ref':ref},'ref')
            for nb_swaps_key, swap_samples in swap_dict.items():
                if len(swap_samples) == 0:
                    continue
                feature_list = []
                feature_swap_dict[nb_swaps_key] = feature_list
                random_id = np.random.randint(0,len(swap_samples))
                feature_swap_dict['check_{}_{}'.format(nb_swaps_key,random_id)] = swap_samples[random_id]
                for ss in swap_samples:
                    ff = vectoriser({'article':article,'ss':ss['summary']},'ss')
                    feature_list.append(ff)
        pickle.dump(feature_dic,open(os.path.join(out_dir,'{}_{}'.format(wanted_feature,sample_file.split('/')[-1])),'wb'))


def writeTestFeatures(wanted_feature):
    print('Writing test features ',wanted_feature)
    vectoriser = HAND_CRAFT([wanted_feature])
    feature_dic = {}

    ### read articles first
    articles_dict = {}
    article_refs = readArticleRefs()
    for entry in article_refs:
        articles_dict.update({entry["id"]: entry["article"]})

    ### read summaries with human scores
    sorted_scores = readSortedScores()
    for article_id, score_list in sorted_scores.items():
        article = articles_dict[article_id]
        if len(score_list) >= 1:
            feature_dic[article_id] = {}
        else:
            continue

        for summary in score_list:
            if summary['summ_id'] not in feature_dic[article_id]:
                feature_dic[article_id][summary['summ_id']] = vectoriser({'article':article,'ss':summary['sys_summ']},'ss')

    pickle.dump(feature_dic,open(os.path.join(TEST_FEATURES_DIR,'{}.p'.format(wanted_feature)),'wb'))

def readSwapFeatures(wanted_feature):
    for feature_file in os.listdir(os.path.join(SWAP_FEATURES_DIR,wanted_feature)):
        if feature_file[-2:] != '.p':
            continue
        feature_dic = pickle.load(open(os.path.join(SWAP_FEATURES_DIR,wanted_feature,feature_file),'rb'))
        print(feature_file, len(feature_dic))
        cnt = 0
        for key in feature_dic:
            print(key, feature_dic[key])
            cnt += 1
            if cnt >= 10:
                break

def readTestFeatures():
    for feature_file in os.listdir(TEST_FEATURES_DIR):
        if feature_file[-2:] != '.p':
            continue
        feature_dic = pickle.load(open(os.path.join(TEST_FEATURES_DIR,feature_file),'rb'))
        print(feature_file, len(feature_dic))
        cnt = 0
        for key in feature_dic:
            print(key, feature_dic[key])
            cnt += 1
            if cnt >= 10:
                break


if __name__ == '__main__':
    #assert len(sys.argv) == 4
    #writeSwapFeatures(sys.argv[1],sys.argv[2],bool(sys.argv[3]))
    #writeSwapFeatures('js','1_100.p',True)

    #writeTestFeatures('js')

    #simpleTest()
    readSwapFeatures('tfidf_cos_avg')

    #readTestFeatures()
