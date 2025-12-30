import pandas as pd
from sklearn.model_selection import train_test_split

# 目前只處理 Microstructure 分類
target_dict = {
    # 'all_data_Micro': 'PROPERTY: Microstructure',
    'all_data_Micro_other': 'PROPERTY: Microstructure',
}


def convert_to_pkl(dataset='all_data', key=None):
    """
    將原始 CSV 切成 train / val，並把 Microstructure 轉成整數類別 target。
    不做正規化，也不產生 text，text 交給 generate_text_test.py 處理。
    """
    target_file = key
    dir = f'{dataset}/{target_file}'

    csv_path = f'{dir}.csv'
    print(f'[INFO] Loading CSV: {csv_path}')
    data = pd.read_csv(csv_path)

    targetCol = target_dict[target_file]
    if targetCol not in data.columns:
        raise ValueError(f"找不到欄位 '{targetCol}'，請確認 CSV 內容。")

    # 1) 固定 random_state 切 train / val
    tr, vl = train_test_split(
        data,
        test_size=0.15,
        random_state=42,
        shuffle=True,
    )

    # 2) 將 Microstructure 轉成整數類別 label
    col_train = tr[targetCol]

    if col_train.dtype == object:
        # 文字類別，例如 'FCC', 'BCC', 'Other'...
        labels, uniques = pd.factorize(col_train, sort=True)
        label_map = {int(i): str(u) for i, u in enumerate(uniques)}
        print('[INFO] Detected categorical targets (object). Label mapping:')
        for i, u in label_map.items():
            print(f'  {i} -> {u}')

        # 用同一組 uniques 對 val encode，確保一致
        tr_labels = labels
        vl_labels = pd.Categorical(vl[targetCol], categories=uniques).codes
        if (vl_labels < 0).any():
            raise ValueError("val set 有在 train set 沒出現過的類別，請檢查資料。")
    else:
        # 數值型：直接轉 int
        print(f'[INFO] Detected numeric targets: dtype={col_train.dtype}')
        tr_labels = tr[targetCol].astype(int).values
        vl_labels = vl[targetCol].astype(int).values
        label_map = None

    # 存 label mapping（只有文字類別時才有）
    if label_map is not None:
        map_df = pd.DataFrame(
            {"label_id": list(label_map.keys()), "label_name": list(label_map.values())}
        )
        map_df.to_csv(f'{dir}/label_mapping_{target_file}.csv', index=False)
        print(f'[INFO] Label mapping saved to {dir}/label_mapping_{target_file}.csv')

    # 3) 在 train / val 加上 target（整數類別）
    tr["target"] = tr_labels
    vl["target"] = vl_labels

    # 4) 儲存切好的 CSV（給 generate_text_test 用）
    out_tr_csv = f'{dir}/tr_{target_file}_convert.csv'
    out_vl_csv = f'{dir}/vl_{target_file}_convert.csv'

    tr.to_csv(out_tr_csv, index=False)
    vl.to_csv(out_vl_csv, index=False)

    print(f'[INFO] Saved train csv: {out_tr_csv} (rows={len(tr)})')
    print(f'[INFO] Saved val   csv: {out_vl_csv} (rows={len(vl)})')
    print('')


def main():
    for key in target_dict.keys():
        print(f'{key}: {target_dict[key]}')
        convert_to_pkl('all_data', key)


if __name__ == "__main__":
    main()
