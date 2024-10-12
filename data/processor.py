import os
import sys
import csv
import json 
import torch
import pickle
import logging
import inspect
import contextlib
from tqdm import tqdm
from functools import partial
from collections import Counter
from multiprocessing import Pool
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)
 

def lmap(a, b):
    print("Returning in lmap function")
    return list(map(a,b))


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    def wrapper_(func):
        print("Inside wrapper_ function of cache_results function")
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            print("Inside nested wrapper of cache_results function")
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True
            
            model_name = my_args.model_name_or_path.split("/")[-1]
            is_pretrain = my_args.pretrain
            cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}_pretrain{is_pretrain}.pkl")
            refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))
            print("Exciting nested wrapper of cache_results function")
            return results
        print("Exciting wrapper_ function of cache_results function")
        return wrapper
    print("cache_results decorater called")
    return wrapper_


def solve(line,  set_type="train", pretrain=1):
    print("inside solve function of processor.py")
    examples = []
        
    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]
    print("head ent text", head_ent_text)
    print("tail ent text", tail_ent_text)
    print("relation text", relation_text)
    i=0
    
    a = tail_filter_entities["\t".join([line[0],line[1]])]
    b = head_filter_entities["\t".join([line[2],line[1]])]
    print("a", a)
    print("b", b)
    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text 
    
    if pretrain:
        examples.append(
            InputExample(guid=guid, text_a="[MASK]", text_b=text_a, text_c = "", label=ent2id[line[0]], real_label=ent2id[line[0]], en=0, rel=0, entity=line[0]))
    else:
        examples.append(
            InputExample(guid=guid, text_a="[MASK]", text_b=text_b + "[PAD]", text_c = "[UNK]" + " " + text_c, label=lmap(lambda x: ent2id[x], b), real_label=ent2id[line[0]], en=ent2id[line[2]], rel=rel2id[line[1]], entity=line[2]))
        examples.append(
            InputExample(guid=guid, text_a="[UNK] ", text_b=text_b + "[PAD]", text_c = "[MASK]" + text_a, label=lmap(lambda x: ent2id[x], a), real_label=ent2id[line[2]], en=ent2id[line[0]], rel=rel2id[line[1]], entity=line[0]))     
    print("Examples", examples)
    print("Exiting solve function of processor.py")  
    return examples


def filter_init(head, tail, t1,t2, ent2id_, ent2token_, rel2id_):
    print("Assigning variables in filter_init of processor.py")
    global head_filter_entities
    global tail_filter_entities
    global ent2text
    global rel2text
    global ent2id
    global ent2token
    global rel2id

    head_filter_entities = head
    tail_filter_entities = tail
    ent2text =t1
    rel2text =t2
    ent2id = ent2id_
    ent2token = ent2token_
    rel2id = rel2id_


def delete_init(ent2text_):
    print("Assigning variables in delete_init of processor.py")
    global ent2text
    ent2text = ent2text_


def convert_examples_to_features_init(tokenizer_for_convert):
    print("Assigning variables in convert_examples_to_features_init of processor.py")
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(example, max_seq_length, mode, pretrain=1):
    """Loads a data file into a list of `InputBatch`s."""
    print("inside convert_examples_to_features of processor.py")
    text_a = " ".join(example.text_a.split()[:128])
    text_b = " ".join(example.text_b.split()[:128])
    text_c = " ".join(example.text_c.split()[:128])
    
    if pretrain:
        input_text_a = text_a
        input_text_b = text_b
    else:
        input_text_a = tokenizer.sep_token.join([text_a, text_b])
        input_text_b = text_c
    

    inputs = tokenizer(
        input_text_a,
        input_text_b,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="longest",
        add_special_tokens=True,
    )
    assert tokenizer.mask_token_id in inputs.input_ids, "mask token must in input"

    features = asdict(InputFeatures(input_ids=inputs["input_ids"],
                            attention_mask=inputs['attention_mask'],
                            labels=torch.tensor(example.label),
                            label=torch.tensor(example.real_label)
        )
    )
    print("Exiting convert_examples_to_features of processor.py")
    return features


@cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, label_list, tokenizer, mode):
    print("Inside get_dataset function of processor.py which is using the decorator cache_results")
    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    # use training data to construct the entity embedding
    if args.faiss_init and mode == "test" and not args.pretrain:
        mode = "train"
    else:
        pass

    if mode == "train":
        train_examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        train_examples = processor.get_dev_examples(args.data_dir)
    else:
        train_examples = processor.get_test_examples(args.data_dir)

    with open(os.path.join(args.data_dir, f"examples_{mode}.txt"), 'w') as file:
        for line in train_examples:
            d = {}
            d.update(line.__dict__)
            file.write(json.dumps(d) + '\n')
    
    features = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    file_inputs = [os.path.join(args.data_dir, f"examples_{mode}.txt")]
    file_outputs = [os.path.join(args.data_dir, f"features_{mode}.txt")]

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in file_inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in file_outputs
        ]

        encoder = MultiprocessingEncoder(tokenizer, args)
        pool = Pool(16, initializer=encoder.initializer)
        encoder.initializer()
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)
        # encoded_lines = map(encoder.encode_lines, zip(*inputs)) 

        stats = Counter()
        for i, (filt, enc_lines) in tqdm(enumerate(encoded_lines, start=1), total=len(train_examples)):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    features.append(eval(enc_line))
            else:
                stats["num_filtered_" + filt] += 1

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

    num_entities = len(processor.get_entities(args.data_dir))
    for f_id, f in enumerate(features):
        en = features[f_id].pop("en")
        rel = features[f_id].pop("rel")
        for i,t in enumerate(f['input_ids']):
            if t == tokenizer.unk_token_id:
                features[f_id]['input_ids'][i] = en + len(tokenizer)
                break
        
        for i,t in enumerate(f['input_ids']):
            if t == tokenizer.pad_token_id:
                features[f_id]['input_ids'][i] = rel + len(tokenizer) + num_entities
                break

    features = KGCDataset(features)
    print("Exciting get_dataset function of processor.py which is using the decorator cache_results")
    return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    print("Initializing InputExample class")
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, real_label=None, en=None, rel=None, entity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. list of entities
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.real_label = real_label
        self.en = en
        self.rel = rel # rel id
        self.entity = entity
        print("text_a", text_a)
        print("text_b", text_b)
        print("text_c", text_c)


@dataclass
class InputFeatures:
    """A single set of features of data."""
    print("inside InputFeatures of processor.py")
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None
    label: torch.Tensor = None
    en: torch.Tensor = 0
    rel: torch.Tensor = 0
    entity: torch.Tensor = None


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        print("Inside _read_tsv funtion of DataProcess class processor.py")
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            print("Exiting _read_tsv funtion of DataProcess class processor.py")
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self, tokenizer, args):
        print("initialising KGProcessor class processor.py")
        self.labels = set()
        self.tokenizer = tokenizer
        self.args = args
        self.entity_path = os.path.join(args.data_dir, "entity2textlong.txt") if os.path.exists(os.path.join(args.data_dir, 'entity2textlong.txt')) \
        else os.path.join(args.data_dir, "entity2text.txt")
    
    def get_train_examples(self, data_dir):
        """See base class."""
        print("Returing in get_train_examples KGProcessor class processor.py")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir, self.args)

    def get_dev_examples(self, data_dir):
        """See base class."""
        print("Returing in get_dev_examples KGProcessor class processor.py")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
      """See base class."""
      print("Returing in get_test_examples KGProcessor class processor.py")
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv")), "test", data_dir, self.args)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        print("Inside get_relations KGProcessor class processor.py")
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        print("Exciting get_relations KGProcessor class processor.py")
        return list(rel2token.values())

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        print("Inside get_labels KGProcessor class processor.py")
        relation = []
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                relation.append(line.strip().split("\t")[-1])
        print("Exiting get_labels KGProcessor class processor.py")
        return relation

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        print("Inside get_entities KGProcessor class processor.py")
        with open(self.entity_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        print("Exciting get_entities KGProcessor class processor.py")
        return list(ent2token.values())

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        print("Returning in get_train_triples KGProcessor class processor.py")
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        print("Returning in get_dev_triples KGProcessor class processor.py")
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir, chunk=""):
        """Gets test triples."""
        print("Returning in get_test_triples KGProcessor class processor.py")
        return self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv"))

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        print("Inside _create_examples KGProcessor class processor.py")
        # entity to text
        ent2text = {}
        ent2text_with_type = {}
        with open(self.entity_path, 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                end = temp[1]#.find(',')
                if "wiki" in data_dir:
                    assert "Q" in temp[0]
                ent2text[temp[0]] = temp[1].replace("\\n", " ").replace("\\", "") #[:end]
  
        entities = list(ent2text.keys())
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        ent2id = {ent : i for i, ent in enumerate(entities)}
        
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      
        relation_names = {}
        with open(os.path.join(data_dir, "relations.txt"), "r") as file:
            for line in file.readlines():
                t = line.strip()
                relation_names[t] = rel2text[t]

        tmp_lines = []
        not_in_text = 0
        for line in tqdm(lines, desc="delete entities without text name."):
            if (line[0] not in ent2text) or (line[2] not in ent2text) or (line[1] not in rel2text):
                not_in_text += 1
                continue
            tmp_lines.append(line)
        lines = tmp_lines
        print(f"total entity not in text : {not_in_text} ")

        # rel id -> relation token id
        rel2id = {w:i for i,w in enumerate(relation_names.keys())}

        examples = []
        # head filter head entity
        head_filter_entities = defaultdict(list)
        tail_filter_entities = defaultdict(list)

        dataset_list = ["train.tsv", "dev.tsv", "test.tsv"]
        # in training, only use the train triples
        if set_type == "train" and not args.pretrain: dataset_list = dataset_list[0:1]
        for m in dataset_list:
            with open(os.path.join(data_dir, m), 'r') as file:
                train_lines = file.readlines()
                for idx in range(len(train_lines)):
                    train_lines[idx] = train_lines[idx].strip().split("\t")

            for line in train_lines:
                tail_filter_entities["\t".join([line[0], line[1]])].append(line[2])
                head_filter_entities["\t".join([line[2], line[1]])].append(line[0])

        
        
        max_head_entities = max(len(_) for _ in head_filter_entities.values())
        max_tail_entities = max(len(_) for _ in tail_filter_entities.values())


        # use bce loss, ignore the mlm
        if set_type == "train" and args.bce:
            lines = []
            for k, v in tail_filter_entities.items():
                h, r = k.split('\t')
                t = v[0]
                lines.append([h, r, t])
            for k, v in head_filter_entities.items():
                t, r = k.split('\t')
                h = v[0]
                lines.append([h, r, t])
        

        # for training , select each entity as for get mask embedding.
        if args.pretrain:
            rel = list(rel2text.keys())[0]
            lines = []
            for k in ent2text.keys():
                lines.append([k, rel, k])
        
        print(f"max number of filter entities : {max_head_entities} {max_tail_entities}")

        from os import cpu_count
        threads = min(1, cpu_count())
        filter_init(head_filter_entities, tail_filter_entities,ent2text, rel2text, ent2id, ent2token, rel2id
            )
        
        annotate_ = partial(
                solve,
                pretrain=self.args.pretrain
            )
        examples = list(
            tqdm(
                map(annotate_, lines),
                total=len(lines),
                desc="convert text to examples"
            )
        )

        tmp_examples = []
        for e in examples:
            for ee in e:
                tmp_examples.append(ee)
        examples = tmp_examples
        # delete vars
        del head_filter_entities, tail_filter_entities, ent2text, rel2text, ent2id, ent2token, rel2id
        print("Exciting _create_examples KGProcessor class processor.py")
        return examples


class Verbalizer(object):
    def __init__(self, args):
        print("Initialising Verbalizer class processor.py")
        if "WN18RR" in args.data_dir:
            self.mode = "WN18RR"
        elif "FB15k" in args.data_dir:
            self.mode = "FB15k"
        elif "umls" in args.data_dir:
            self.mode = "umls"
          
    def _convert(self, head, relation, tail):
        print("Inside _convert function Verbalizer class processor.py")
        if self.mode == "umls":
            return f"The {relation} {head} is "
        print("Exiting _convert function Verbalizer class processor.py")
        return f"{head} {relation}"


class KGCDataset(Dataset):
    def __init__(self, features):
        print("Initializing KGCDataset classs processor.py")
        self.features = features

    def __getitem__(self, index):
        print("returning __getitem__ in KGCDataset processor.py")
        return self.features[index]
    
    def __len__(self):
        print("returning __len__ in KGCDataset processor.py")
        return len(self.features)


class MultiprocessingEncoder(object):
    def __init__(self, tokenizer, args):
        print("Initializing MultiprocessingEncoder class processor.py")
        self.tokenizer = tokenizer
        self.pretrain = args.pretrain
        self.max_seq_length = args.max_seq_length

    def initializer(self):
        print("Initializng initializer in MultiprocessingEncoder class processor.py")
        global bpe
        bpe = self.tokenizer

    def encode(self, line):
        print("Returning in encode function in MultiprocessingEncoder class processor.py")
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        print("Returning in decode function in MultiprocessingEncoder class processor.py")
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        print("Inside encode_lines function in MultiprocessingEncoder class processor.py")
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                return ["EMPTY", None]
            enc_lines.append(json.dumps(self.convert_examples_to_features(example=eval(line))))
        print("Exiting encode_lines function in MultiprocessingEncoder class processor.py")
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        print("Inside decode_lines function in MultiprocessingEncoder class processor.py")
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        print("Exiting decode_lines function in MultiprocessingEncoder class processor.py")
        return ["PASS", dec_lines]

    def convert_examples_to_features(self, example):
        print("Inside convert_examples_to_features function in MultiprocessingEncoder class processor.py")
        pretrain = self.pretrain
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""
        
        text_a = example['text_a']
        text_b = example['text_b']
        text_c = example['text_c']

        if pretrain:
            # the des of xxx is [MASK] .
            # xxx is the description of [MASK].
            input_text = f"The description of {text_a} is that {text_b} ."
            input_text = input_text.encode('utf-8', 'ignore').decode('utf-8')
            print("input text", input_text)
            inputs = bpe(
                input_text,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
        else:
            if text_a == "[MASK]":
                input_text_a = bpe.sep_token.join([text_a, text_b])
                input_text_b = text_c
            else:
                input_text_a = text_a
                input_text_b = bpe.sep_token.join([text_b, text_c])

            input_text_a = input_text_a.encode('utf-8', 'ignore').decode('utf-8')
            print("Input text a", input_text_a)
            input_text_b = input_text_b.encode('utf-8', 'ignore').decode('utf-8')
            print("Input text b", input_text_b)
            inputs = bpe(
                input_text_a,
                input_text_b,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
        assert bpe.mask_token_id in inputs.input_ids, "mask token must in input"

        features = asdict(InputFeatures(input_ids=inputs["input_ids"],
                                attention_mask=inputs['attention_mask'],
                                labels=example['label'],
                                label=example['real_label'],
                                en=example['en'],
                                rel=example['rel'],
                                entity=example['entity']
            )
        )
        print("Exiting convert_examples_to_features function in MultiprocessingEncoder class processor.py")
        return features
