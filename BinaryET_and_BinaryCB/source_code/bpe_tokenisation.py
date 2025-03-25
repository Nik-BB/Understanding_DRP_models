'''Processing specific to the DeepTTC model'''
import sys
import codecs
import pandas as pd
import numpy as np 
import os
print(os.getcwd())
sys.path.insert(0, 'source_code') 
sys.path.insert(0, '../data') 
#sys.path.insert(0, '../') 
#mport bpe #bpe is just the needed funciton from subword-nmt github 
#but can pip install subword-nmt instead
#https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py


def smile_encoder(smile):
    '''Encodes smiles as ESPF's https://github.com/kexinhuang12345/ESPF'''
    epsf_dir = 'data/epsf'
    vocab_path = f'{epsf_dir}/drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv(f'{epsf_dir}/subword_units_map_chembl_freq_1500.csv')

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    #splits on smile substructures from EPSF github
    smile_substructures = dbpe.process_line(smile).split()
    encoded_substructures = np.asarray(
        [words2idx_d[sub] for sub in smile_substructures]) 
    # except:
    #     encoded_substructures = np.array([0])
    #     print('except')

    #made encoded length max_d by padding or just taking max_d charaters 
    l = len(encoded_substructures)
    if l < max_d:
        encoded_substructures = np.pad(
            encoded_substructures, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        encoded_substructures = encoded_substructures[:max_d]
        input_mask = [1] * max_d

    return encoded_substructures, np.asarray(input_mask)


#code from subword-nmt
#https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py
import sys
import os
import re
import random
import tempfile

class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):

        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes.read().rstrip('\n').split('\n')) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

        self.cache = {}

    def process_lines(self, filename, outfile, dropout=0, num_workers=1):

        if sys.version_info < (3, 0):
            print("Parallel mode is only supported in Python3.")
            sys.exit(1)

        if num_workers == 1:
            _process_lines(self, filename, outfile, dropout, 0, 0)
        elif num_workers > 1:
            with open(filename, encoding="utf-8") as f:
                size = os.fstat(f.fileno()).st_size
                chunk_size = int(size / num_workers)
                offsets = [0 for _ in range(num_workers + 1)]
                for i in range(1, num_workers):
                    f.seek(chunk_size * i)
                    pos = f.tell()
                    while True:
                        try:
                            line = f.readline()
                            break
                        except UnicodeDecodeError:
                            pos -= 1
                            f.seek(pos)
                    offsets[i] = f.tell()
                    assert 0 <= offsets[i] < 1e20, "Bad new line separator, e.g. '\\r'"
            res_files = []
            pool = Pool(processes=num_workers)
            for i in range(num_workers):
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.close()
                res_files.append(tmp)
                pool.apply_async(_process_lines, (self, filename, tmp.name, dropout, offsets[i], offsets[i + 1]))
            pool.close()
            pool.join()
            for i in range(num_workers):
                with open(res_files[i].name, encoding="utf-8") as fi:
                    for line in fi:
                        outfile.write(line)
                os.remove(res_files[i].name)
        else:
            raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line, dropout)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence, dropout=0):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments
    
    
def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if not dropout and orig in cache:
        return cache[orig]

    if glossaries_regex and glossaries_regex.match(orig):
        cache[orig] = (orig,)
        return (orig,)

    if len(orig) == 1:
        return orig

    if version == (0, 1):
        word = list(orig) + ['</w>']
    elif version == (0, 2): # more consistent handling of word-final segments
        word = list(orig[:-1]) + [orig[-1] + '</w>']
    else:
        raise NotImplementedError

    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram = ''.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    word = tuple(word)
    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word