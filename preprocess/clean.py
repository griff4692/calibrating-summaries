import ftfy


def clean(example):
    example['sections'] = ftfy.fix_text(example['sections'].encode().decode('unicode_escape', 'ignore'))
    return example
