import os
import json
import glob
import hashlib
import pandas as pd

import translators as ts

# ====================
# Aesthetic Printing
# ====================

def aesthetic_print_equals(title):
    print('='*len(title))
    print(title)
    print('='*len(title),'\n')

def aesthetic_print_stars(title):
    print('*'*len(title))
    print(title)
    print('*'*len(title),'\n')

# =================
# File Management
# =================

class DirectoryDict:

    def __init__(self, directory_path, file_type='.json'):
        # Extract only .json file types from directory
        self.directory = get_dict_data(directory_path, file_type)

    def stats(self):
        for name, instance in self.directory.items():
            aesthetic_print_equals(f'File Name: {name}')
            if 'entry' in instance.keys():
                print(f'Number of Entries: {len(instance["entry"])}\n')

    def filenames(self):
        return list(self.directory.keys())

def attempt_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"The file {file_path} does not exist")

# =================
# File Conversion
# =================

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data
    
def dataframe_to_json_file(df, save_file_path):

    records = df.to_dict('records')

    with open(save_file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, cls=NpEncoder) + "\n")

def list_of_dict_to_json_file(list_of_dict, save_file_path):

    with open(save_file_path, "w", encoding="utf-8") as f:
        for record in list_of_dict:
            f.write(json.dumps(record, cls=NpEncoder) + "\n")

def json_file_to_list_of_dict(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data

def get_dict_file_directory(folder_path, file_type='.json'):
    folder_path = folder_path.strip('/')
    file_paths = glob.glob(f"{folder_path}/*{file_type}")
    file_names = [f.split('/')[-1] for f in file_paths]
    file_directory = {name: {'path': path} for path, name in zip(file_paths, file_names)}
    return file_directory

def get_dict_data(folder_path, file_type='.json'):
    file_directory = get_dict_file_directory(folder_path, file_type)
    for name, instance in file_directory.items():
        entry = json_file_to_list_of_dict(instance['path'])
        file_directory[name]['entry'] = entry
    return file_directory

def load_json_as_dictionary(json_file_path):
    '''
    Given the file path to a .json file, return the contents of the .json file as a dictionary.

    Parameters
    ----------
    json_file_path : json_file_path
        The file path of the .json file whose contents are to be loaded as a dictionary. Must end with '.json'.

    Returns
    -------
    dict
        A dictionary of the contents of the .json file located at the input file path.
    '''
    # Check filename
    if not isinstance(json_file_path, str) or len(json_file_path) <= len('.json'):
        raise Exception('File name must have a name in string and must end with ".json"')
    if not json_file_path.endswith('.json'):
        raise Exception('File name must end with ".json"')

    # Load JSON as dictionary
    with open(json_file_path, 'r') as f:
        return json.load(f)

def save_dictionary_as_json(dictionary, json_file_path):
    '''
    Given a dictionary, save it as a .json file.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be saved as a .json file.

    json_file_path : json_file_path
        The .json file path to save the dictionary. Must end with '.json'.
    '''
    # Check filename
    if not isinstance(json_file_path, str) or len(json_file_path) <= len('.json'):
        raise Exception('File name must have a name in string and must end with ".json"')
    if not json_file_path.endswith('.json'):
        raise Exception('File name must end with ".json"')

    # Save dictionary as JSON
    print(f'Saving dictionary as "{json_file_path}"')
    with open(json_file_path, 'w') as output_file:
        json.dump(dictionary, output_file)
    print('File saved.')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# ========
# Others
# ========

def apply_ast_literal_eval(df, columns=['event_clusters', 'sentence_no', 'sentences', 'laser_embeddings']):
    df = pd.read_csv('subtask3_en_with_laser_embeddings.csv')
    for col in columns:
        try:
            df[col] = df[col].apply(lambda x: ast.literal_eval(str(x).replace('[array(','').replace(')]', '')))
        except:
            print(f'Error with column {col}')
            continue
    return df

# =============
# Translation
# =============

def translate_single(text, target='es'):

    lang_dict = {'es': 'es',
                 'pt': 'pt',
                 'spa': 'es'}

    target_normalised = lang_dict[target]

    cache_dir = 'translation_cache_' + target_normalised
    data_md5 = hashlib.md5(text.encode('utf-8')).hexdigest()
    data_md5_file = os.path.join(cache_dir, data_md5 + '.json')

    # Check if cached translation available
    if os.path.exists(data_md5_file):
        with open(data_md5_file, 'r') as f:
            translation = json.load(f)
            return translation
    else:

        # Otherwise, make translation
        translation = ts.alibaba(text, 'auto', target) # ts.google, ts.bing, ts.baidu, ts.alibaba

        # Cache translation
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(data_md5_file, 'w') as f:
            json.dump(translation, f)

        return translation

def translate_json_file(json_file_path, translate_to='es', key='sentences'):

    # Read file
    docs = read(json_file_path)

    # Translate
    print(f'Translating to {translate_to}')
    docs_translated = []
    for idx, doc in enumerate(docs):
        docs_translated.append(translate_dict(doc, translate_to=translate_to, key=key))
        print(f'... Done with entry {idx+1} out of {len(docs)}')

    # Save as new file
    save_file_path = json_file_path.replace('.json', f'_{translate_to}.json')
    list_of_dict_to_json_file(docs_translated, save_file_path)
    print(f'Translation complete. Translated file save to: {save_file_path}\n')

def translate_dict(doc, translate_to='es', key='sentences'):
    doc[key] = [translate_single(sent, translate_to) for sent in doc[key]]
    return doc

def translate_csv_file(csv_file_path, translate_to='es', columns=['']):

    # Read file
    df = pd.read_csv(csv_file_path)

    # Translate
    print(f'Translating to {translate_to}')
    columns = [col for col in columns if col in df.columns]
    for col in columns:
        translated_col = []
        for idx, text in enumerate(df[col]):
            translated_text = translate_single(str(text), translate_to)
            translated_col.append(translated_text)
            print(f'... [{col}] Done with entry {idx+1} out of {len(df)}')
        df[col] = translated_col

    # Save as new file
    save_file_path = csv_file_path.replace('.csv', f'_{translate_to}.csv')
    df.to_csv(save_file_path, index=False)
    print(f'Translation complete. Translated file save to: {save_file_path}\n')

    return df
