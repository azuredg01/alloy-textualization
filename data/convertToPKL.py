import pandas as pd
from sklearn.model_selection import train_test_split


target_dict = {
    # 'all_data_YS': 'PROPERTY: YS (MPa)',
    # 'all_data_YM': 'PROPERTY: Calculated Young modulus (GPa)',
    # 'all_data_HV': 'PROPERTY: HV',
    # 'all_data_El': 'PROPERTY: Elongation (%)',
    # 'all_data_UTS': 'PROPERTY: UTS (MPa)',
    # 'all_data_ExpDensity': 'PROPERTY: Exp. Density (g/cm$^3$)',
    'all_data_calcDensity': 'PROPERTY: Calculated Density (g/cm$^3$)',
}


def convert_to_pkl(dataset='all_data', key=None):
    target_file = key
    dir = f'{dataset}/{target_file}'

    # normalize YS, split into train and val sets
    # dataset = 'MPEA'
    # data = pd.read_csv(f'./data/{dataset}/{dataset}.csv')
    data = pd.read_csv(f'{dir}.csv')


    # if dataset == 'ys_clean':
    #     targetCol = 'YS'
    # else:
    #     targetCol = 'PROPERTY: Calculated Young modulus (GPa)'

    targetCol = target_dict[target_file]

    # data[targetCol] /= max(data[targetCol])
    # tr, vl = train_test_split(data, test_size=0.15)

    # 1) 先切分（固定種子）
    tr, vl = train_test_split(data, 
                            test_size=0.15, 
                            random_state=42, # new add
                            shuffle=True # new add
                            )

    # 2) 只用 train 的統計量正規化
    tmax = tr[targetCol].max()
    print(f'Max {targetCol} in train set: {tmax}')
    

    tr["target_nor"] = tr[targetCol] / tmax
    vl["target_nor"] = vl[targetCol] / tmax

    tr.to_csv(f'{dir}/tr_{target_file}_convert.csv')
    vl.to_csv(f'{dir}/vl_{target_file}_convert.csv')


    # # tr = pd.read_csv(f'./data/{dataset}/tr.csv')
    # ytr = tr["target_nor"]
    # Xtr = tr.drop("target_nor", axis=1)

    # # vl = pd.read_csv(f'./data/{dataset}/vl.csv')
    # yvl = vl["target_nor"]
    # Xvl = vl.drop("target_nor", axis=1)
    # texttr = []
    # for i, row in Xtr.iterrows():
    #     string = ''
    #     for k, v in row.items():
    #         string += f'{k}: {v}. '
    #     texttr.append(string)

    # textvl = []
    # for i, row in Xvl.iterrows():
    #     string = ''
    #     for k, v in row.items():
    #         string += f'{k}: {v}. '
    #     textvl.append(string)

    # df_train = pd.DataFrame()
    # df_train['text'] = texttr
    # df_train['target'] = ytr.tolist()

    # df_val = pd.DataFrame()
    # df_val['text'] = textvl
    # df_val['target'] = yvl.tolist()

    # df_train.to_pickle(f'{dir}/tr_{target_file}_convert.pkl')
    # df_val.to_pickle(f'{dir}/vl_{target_file}_convert.pkl')

def main():
    for key in target_dict.keys():
        print(f'{key}: {target_dict[key]}')

        convert_to_pkl('all_data', key)

if __name__ == "__main__":
    main()