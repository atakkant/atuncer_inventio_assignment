import argparse
import pandas as pd
import os

class SalesFeaturePipeline:
    def __init__(self, brand_df, product_df, store_df, sales_df):
        self.brand_df = brand_df
        self.product_df = product_df
        self.store_df = store_df
        self.sales_df = sales_df
        self.merged = None
        self.full_df = None

    @classmethod
    def load_data_from_csv(cls,brand_csv, product_csv, strore_csv, sales_csv):
        brand_df = pd.read_csv(brand_csv)
        product_df = pd.read_csv(product_csv)
        store_df = pd.read_csv(strore_csv)
        sales_df = pd.read_csv(sales_csv)
        return cls(brand_df, product_df, store_df, sales_df)

    def preprocess(self):
        product = self.product_df.merge(self.brand_df, left_on='brand', right_on='name', suffixes=('_product', '_brand'))
        product = product.rename(columns={'id_product': 'product_id', 'id_brand': 'brand_id'})

        merged = self.sales_df.merge(product[['product_id', 'brand_id']], left_on='product', right_on='product_id')
        merged = merged.merge(self.store_df.rename(columns={'id': 'store_id'}), left_on='store', right_on='store_id')

        merged['date'] = pd.to_datetime(merged['date'])
        self.merged = merged

    def compute_product_features(self):
        df = self.merged.groupby(['product_id', 'store_id', 'date'])['quantity'].sum().reset_index()
        df = df.rename(columns={'quantity': 'sales_product'})
        df = df.sort_values(['product_id', 'store_id', 'date'])
        df['MA7_P'] = df.groupby(['product_id', 'store_id'])['sales_product'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['LAG7_P'] = df.groupby(['product_id', 'store_id'])['sales_product'].shift(7)
        return df

    def compute_brand_features(self):
        df = self.merged.groupby(['brand_id', 'store_id', 'date'])['quantity'].sum().reset_index()
        df = df.rename(columns={'quantity': 'sales_brand'})
        df = df.sort_values(['brand_id', 'store_id', 'date'])
        df['MA7_B'] = df.groupby(['brand_id', 'store_id'])['sales_brand'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['LAG7_B'] = df.groupby(['brand_id', 'store_id'])['sales_brand'].shift(7)
        return df

    def compute_store_features(self):
        df = self.merged.groupby(['store_id', 'date'])['quantity'].sum().reset_index()
        df = df.rename(columns={'quantity': 'sales_store'})
        df = df.sort_values(['store_id', 'date'])
        df['MA7_S'] = df.groupby('store_id')['sales_store'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['LAG7_S'] = df.groupby('store_id')['sales_store'].shift(7)
        return df

    def merge_features(self, product_feat, brand_feat, store_feat):
        df = product_feat.merge(self.merged[['product_id', 'store_id', 'brand_id', 'date']].drop_duplicates(),
                                on=['product_id', 'store_id', 'date'], how='left')
        df = df.merge(brand_feat, on=['brand_id', 'store_id', 'date'], how='left')
        df = df.merge(store_feat, on=['store_id', 'date'], how='left')
        df = df.sort_values(by=['product_id', 'brand_id', 'store_id', 'date']).reset_index(drop=True)
        self.full_df = df

    def filter_date_range(self, min_date, max_date):
        df = self.full_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] >= min_date) & (df['date'] <= max_date)]

    def compute_top_wmape(self, top):
        df = self.full_df.dropna(subset=['sales_product', 'MA7_P'])
        wmape_df = (
            df.groupby(['product_id', 'store_id', 'brand_id'])
              .apply(lambda g: (abs(g['sales_product'] - g['MA7_P']).sum()) / g['sales_product'].sum()
                     if g['sales_product'].sum() != 0 else None)
              .reset_index(name='WMAPE')
        )
        return wmape_df.sort_values('WMAPE', ascending=False).head(top)

def main():
    parser = argparse.ArgumentParser(description="Generate sales features and WMAPE report.")
    parser.add_argument("--min-date", type=str, default="2021-01-08", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--max-date", type=str, default="2021-05-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=5, help="Number of top WMAPE rows to output")

    args = parser.parse_args()
    min_date = args.min_date
    max_date = args.max_date
    top = args.top
    print(f"Filtering data between {min_date} and {max_date} and computing top {top} WMAPE")
    
    data_path = './q5-dataeng-forecasting-features/input_data/data'
    brand_csv = os.path.join(data_path, 'brand.csv')
    product_csv = os.path.join(data_path, 'product.csv')
    sales_csv = os.path.join(data_path, 'sales.csv')
    store_csv = os.path.join(data_path, 'store.csv')

    print("Loading data from CSV files...")
    pipeline = SalesFeaturePipeline.load_data_from_csv(brand_csv, product_csv, store_csv, sales_csv)
    pipeline.preprocess()

    print("Computing features...")
    product_feat = pipeline.compute_product_features()
    brand_feat = pipeline.compute_brand_features()
    store_feat = pipeline.compute_store_features()

    print("Merging features...")
    pipeline.merge_features(product_feat, brand_feat, store_feat)
    filtered = pipeline.filter_date_range(min_date, max_date)
    top_wmape_df = pipeline.compute_top_wmape(top)

    print("Saving filtered features and WMAPE to CSV files...")
    filtered.to_csv("features.csv", index=False)
    top_wmape_df.to_csv("mapes.csv", index=False)
    print("Filtered features and WMAPE saved to features.csv and mapes.csv.")
    

if __name__ == "__main__":
    print("Starting the sales feature pipeline...")
    main()
    print("Pipeline completed successfully.")
