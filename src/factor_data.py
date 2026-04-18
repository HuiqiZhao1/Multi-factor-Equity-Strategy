"""
factor_data.py
直接从 Kenneth French 网站下载因子数据 CSV，与股票收益率合并。
"""

import pandas as pd
import zipfile
import io
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Kenneth French 数据库的直接下载链接
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def download_and_read_csv(url, skip_footer=0):
    """
    从 Kenneth French 网站下载 ZIP，解压并读取里面的 CSV。
    """
    print(f"正在下载: {url.split('/')[-1]}")

    response = urllib.request.urlopen(url)
    zip_data = zipfile.ZipFile(io.BytesIO(response.read()))

    # ZIP 里只有一个 CSV 文件
    csv_name = zip_data.namelist()[0]
    raw_text = zip_data.open(csv_name).read().decode("utf-8")

    return raw_text


def parse_ff3(raw_text):
    """
    解析 Fama-French 三因子 CSV。
    文件格式比较特殊：前几行是说明文字，数据从某行开始，
    而且月度数据和年度数据混在一起。
    """
    lines = raw_text.strip().split("\n")

    # 找到月度数据的起始行（第一行纯数字开头的）
    data_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 月度数据的格式是 YYYYMM, 如 201501
        first_field = line.split(",")[0].strip()
        if first_field.isdigit() and len(first_field) == 6:
            data_rows.append(line)

    # 转成 DataFrame
    from io import StringIO
    csv_text = "Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(data_rows)
    df = pd.read_csv(StringIO(csv_text))

    # 把 YYYYMM 转成日期
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    # 从百分比转成小数
    df = df / 100

    return df


def parse_momentum(raw_text):
    """
    解析动量因子 CSV，逻辑和三因子类似。
    """
    lines = raw_text.strip().split("\n")

    data_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        first_field = line.split(",")[0].strip()
        if first_field.isdigit() and len(first_field) == 6:
            data_rows.append(line)

    from io import StringIO
    csv_text = "Date,WML\n" + "\n".join(data_rows)
    df = pd.read_csv(StringIO(csv_text))

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    df = df / 100

    return df


def merge_all():
    """
    合并：股票收益率 + 三因子 + 动量因子。
    """
    # 读取之前生成的收益率数据
    returns = pd.read_csv(
        PROCESSED_DIR / "monthly_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    # 下载并解析因子数据
    ff3_raw = download_and_read_csv(FF3_URL)
    ff3 = parse_ff3(ff3_raw)

    mom_raw = download_and_read_csv(MOM_URL)
    mom = parse_momentum(mom_raw)

    # 合并三因子和动量
    factors = ff3.join(mom, how="inner")

    # 筛选时间范围
    factors = factors[(factors.index >= "2015-01-01") & (factors.index <= "2025-01-01")]

    # 找到两边都有的日期
    common_dates = returns.index.intersection(factors.index)
    print(f"\n重叠日期: {len(common_dates)} 个月")
    print(f"起始: {common_dates[0].strftime('%Y-%m')}")
    print(f"结束: {common_dates[-1].strftime('%Y-%m')}")

    # 对齐
    returns = returns.loc[common_dates]
    factors = factors.loc[common_dates]

    # 保存
    factors.to_csv(PROCESSED_DIR / "ff_factors.csv")
    print(f"\n因子数据已保存: {PROCESSED_DIR / 'ff_factors.csv'}")
    print(f"因子列: {list(factors.columns)}")

    return returns, factors


if __name__ == "__main__":
    try:
        returns, factors = merge_all()
        print(f"\n收益率矩阵: {returns.shape}")
        print(f"因子矩阵: {factors.shape}")
        print("\n因子数据前5行:")
        print(factors.head())
        print("\n完成！")
    except Exception as e:
        print(f"出错了: {e}")
        import traceback
        traceback.print_exc()