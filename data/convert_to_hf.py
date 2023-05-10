from collections import Counter
import ujson
from datasets import Dataset, DatasetDict


if __name__ == '__main__':
    in_fn = '/nlp/projects/summarization/msr_acl_2023/dendrite_data_tmp/chemistry/processed_docs.json'
    print(f'Loading from {in_fn}')
    with open(in_fn, 'r') as fd:
        docs = ujson.load(fd)

    cts = []
    outputs = {'train': [], 'validation': [], 'test': []}
    for doc in docs:
        fp = doc.pop('fp')
        fn = doc.pop('fn')
        split = doc.pop('split')
        doc.pop('Article File', None)
        doc.pop('AccessionID', None)
        doc.pop('PMID', None)
        doc.pop('MID', None)
        doc.pop('Article Citation', None)
        doc.pop('License', None)
        doc.pop('journal', None)
        doc.pop('Retracted', None)
        doc.pop('LastUpdated (YYYY-MM-DD HH:MM:SS)', None)
        sections = doc.pop('sections')

        body_str = []
        header_str = []

        DELIM = '<!>'

        for section in sections:
            header_str.append('' if section['header'] is None else section['header'])
            assert DELIM not in section['body']
            body_str.append(section['body'])

        header_str = DELIM.join(header_str)
        body_str = DELIM.join(body_str)

        doc['headers'] = header_str
        doc['sections'] = body_str

        pmc_source = doc.pop('pmc_source', None)
        if pmc_source is not None:
            doc['article_source'] = 'PubMed Author Manuscript' if pmc_source == 'author_manuscript' else 'PubMed Open Access'
        elif 'chemrxiv' in fp:
            doc['article_source'] = 'ChemRxiv'
        elif 'rsc' in fp:
            doc['article_source'] = 'Royal Society of Chemistry (RSC)'
        elif 'beilstein' in fp:
            doc['article_source'] = 'Beilstein'
        elif 'nature_coms' in fp:
            doc['article_source'] = 'Nature Communications Chemistry'
        elif 'scientific_reports' in fp:
            doc['article_source'] = 'Scientific Reports - Nature'
        elif 'chemistry_open' in fp:
            doc['article_source'] = 'Chemistry Open'
        elif 'cell' in fp:
            doc['article_source'] = 'Chem Cell'
        else:
            raise Exception('Unrecognized.')

        cts.append(doc['article_source'])

        outputs[split].append(doc)

    print(Counter(cts).most_common())

    cts = Counter(cts)
    keys = list(sorted(list(cts.keys())))
    for key in keys:
        print(key, cts[key])

    print(len(outputs['train']))
    outputs = DatasetDict({
        'train': Dataset.from_list(outputs['train']),
        'validation': Dataset.from_list(outputs['validation']),
        'test': Dataset.from_list(outputs['test']),
    })

    out_dir = '/nlp/projects/summarization/msr_acl_2023/dendrite_data_tmp/chemistry_hf'
    print('Saving to Hub')
    outputs.push_to_hub('griffin/ChemSum')
    print('Saving to disk')
    outputs.save_to_disk(out_dir)
