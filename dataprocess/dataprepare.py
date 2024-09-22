import argparse
import pandas as pd
import json
import gzip
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct SMILES list, mask dictionary, and pre-training dataset')
    parser.add_argument('--df_path', type=str, default='processed/building_block_processed_mw150.csv', help='Path to the input CSV file, building_block with func_group')
    parser.add_argument('--df_r_path', type=str, default='raw/reaction_template.csv', help='Path to the reaction_template CSV file')

    args = parser.parse_args()

    # match the molecular weight value from the file name
    #match = re.search(r'_mw(\d+_\dk)', args.df_path)
    match = re.search(r'_mw(\d+)', args.df_path)
    if match:
        mw_value = match.group(1)
    else:
        raise ValueError("The molecular weight value could not be found in the file name.")

    output_filename_s = f'final/smiles_list_mw{mw_value}.json.gz'

    # Generate smiles_list.json.gz
    df = pd.read_csv(args.df_path)
    smiles_list = df['SMILES'].tolist()
    with gzip.open(output_filename_s, 'wt', encoding='UTF-8') as f_out:
        json.dump(smiles_list, f_out)
    print(f'SMILES list saved at {output_filename_s}')

    # Generate mask_dict.json.gz
    df_r = pd.read_csv(args.df_r_path)
    bi_list = df_r.loc[df_r["uni/bi"] == "bi", "name"].tolist()
    output_list = [name + '_reactant_1' for name in bi_list] + [name + '_reactant_2' for name in bi_list]

    print(f'The bi-component reaction dictionary contains {len(output_list)} keys')

    mask_dict = {}
    # Traverse each element in output_list, and traverse the df['func_group'] column, adding the indices of the rows that contain the key value to the list
    for key in output_list:
        dict_value = [] 
        for i, val in enumerate(df['func_group']):
            if key in val:
                dict_value.append(i)  # mask_dict collects the molecule indices corresponding to each reaction tag
        mask_dict[key] = dict_value

    output_filename_r = f'final/mask_dict_mw{mw_value}.json.gz'
    with gzip.open(output_filename_r, 'wt', encoding='UTF-8') as f_out:
        json.dump(mask_dict, f_out)
    print(f'Mask dictionary saved at {output_filename_r}')

    max_list_length = max(len(lst) for lst in mask_dict.values())
    print(f'The maximum number of selectable molecules for a single tag in mask_dict is {max_list_length}')
    
    # Create pre-training dataset
    all_tags = list(mask_dict.keys()) + ["ANY", "EOS", "UNI"]
    rows = [{'tag': tag, 'smiles': smiles} for tag in all_tags for smiles in smiles_list]
    df_p = pd.DataFrame(rows)
    output_filename_p = f'final/pretrain_mw{mw_value}.csv'
    df_p.to_csv(output_filename_p, index=False)
