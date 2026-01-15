import pandas as pd
import sys
from abc import abstractmethod

sys.path.append(r"\Users\marika.ivanova\PycharmProjects")
sys.path.append(r"C:\Users\marika.ivanova\PycharmProjects\CoefficientCalc")

sys.path.append(r"C:\Users\Public\Projekty\univerzalni")
sys.path.append(r"C:\Users\Public\Projekty\univerzalni\CoefficientCalc")

import cisteni_dat_imputace_2 as imputace
from CoefficientCalc import CoefficientCalculator


class EXP_sklad:

    def __init__(self, brand, whs):

        path = fr"\\10.0.132.133\data\99_AI\Inputy"
        self.brand = brand
        self.whs = whs

        self.df_exp_sklady_centrala_today = imputace.get_df_sklad_centr()
        self.df_exp_sklady_centrala_yesterday = pd.read_csv(rf"{path}\Misc\ExportSkladyCentralaVcera.csv",
                                                            sep="\t", encoding="cp1250")

        self.df_exp_karty_zbozi = imputace.get_df_karty()

        self.df_exp_zapujcky = imputace.get_df_zapujcky()

        df_all_exports, _ = imputace.get_df_all_boutiques()

        # z exportu vsech butiku chceme jen ty, ktere jsou na buticich (neprodane, nevracene a maji zapujceno z EXP
        df_exp_all_btqs_zapujcky_today = EXP_sklad.filter_zapujcky(df_all_exports)
        df_exp_all_btqs_zapujcky_yesterday = pd.read_csv(rf"{path}\Misc\ZapujckyButikyVcera.csv",
                                                            sep="\t", encoding="cp1250")
        self.df_exp_all_btqs_zapujcky_today = EXP_sklad.filter_sklady(df_exp_all_btqs_zapujcky_today, self.whs)
        self.df_exp_all_btqs_zapujcky_yesterday = EXP_sklad.filter_sklady(df_exp_all_btqs_zapujcky_yesterday, self.whs)

        self.df_id_styles = imputace.get_df_style_set(brand)

        # add style_set column to both export_sklady_centrala a export karty zbozi
        self.df_exp_sklady_centrala_today["style_set"] = self.df_exp_sklady_centrala_today["zkr"].map(
            self.df_id_styles["style_set"]).apply(lambda x: x if isinstance(x, set) else set())
        self.df_exp_sklady_centrala_yesterday["style_set"] = self.df_exp_sklady_centrala_yesterday["zkr"].map(
            self.df_id_styles["style_set"]).apply(lambda x: x if isinstance(x, set) else set())

        # add style_set column to both zapujcky dfs
        self.df_exp_all_btqs_zapujcky_today["style_set"] = self.df_exp_all_btqs_zapujcky_today["zkr"].map(
            self.df_id_styles["style_set"]).apply(lambda x: x if isinstance(x, set) else set())
        self.df_exp_all_btqs_zapujcky_yesterday["style_set"] = self.df_exp_all_btqs_zapujcky_yesterday["zkr"].map(
            self.df_id_styles["style_set"]).apply(lambda x: x if isinstance(x, set) else set())

        self.df_exp_karty_zbozi["style_set"] = self.df_exp_karty_zbozi["zkr"].map(self.df_id_styles["style_set"]).apply(
            lambda x: x if isinstance(x, set) else set())
        self.df_exp_zapujcky["style_set"] = self.df_exp_zapujcky["zbozi_zkratka"].map(
            self.df_id_styles["style_set"]).apply(lambda x: x if isinstance(x, set) else set())

        # stav exp+showroom dnes
        self.df_exp_showroom_drahe_today = self.df_exp_sklady_centrala_today[
            self.df_exp_sklady_centrala_today["sklad"].isin(self.whs)]

        # zapujcky, ktere budou vracene do 3 mes
        cutoff_date = pd.Timestamp.now() + pd.DateOffset(months=3)
        self.df_zapujcky_recent = self.df_exp_zapujcky[(self.df_exp_zapujcky["datum"] <= cutoff_date) &
                                                       (self.df_exp_zapujcky["butik"].isin(self.whs))]


    @abstractmethod
    def filter_sklady(df, sklady):
        try:
            df_filt = df[df["sklad"].isin(sklady)]
        except KeyError:
            return df
        return df_filt

    @abstractmethod
    def filter_zapujcky(df, include_prodane=False):
        """
        returns only rows with nonempty ZapujcenoZ
        """
        try:
            df_zapujcky = df[df["ZapujcenoZ"].notna() & (df["ZapujcenoZ"].str.strip() != "")]
            if not include_prodane:
                df_zapujcky = df_zapujcky[df_zapujcky["datum_vraceni"].isna() & df_zapujcky["datum_prodej"].isna()]
        except KeyError:
            return df
        return df_zapujcky

    @abstractmethod
    def save_yesterday_data():
        """
        ulozeni aktualniho stavu ExportSkladyCentrala a zapujcek na buticich
        """
        # ExportSkladyCentrala
        exp_sklady_centrala_today = imputace.get_df_sklad_centr()
        exp_sklady_centrala_today.to_csv(
            rf"{imputace.data_server_connection}\Inputy\Misc\ExportSkladyCentralaVcera.csv",
            sep="\t", encoding="cp1250")

        # Zapujcky na buticich
        df_all_btqs, _ = imputace.get_df_all_boutiques()
        df_export_boutiques_zapujcky = EXP_sklad.filter_zapujcky(df_all_btqs)
        df_export_boutiques_zapujcky.to_csv(rf"{imputace.data_server_connection}\Inputy\Misc\ZapujckyButikyVcera.csv",
                                            sep="\t", encoding="cp1250")

    def apply_conditions(
            self,
            df,
            sklad: str | list[str] | None = None,
            cena_min=None,
            cena_max=None,
            druh=None,
    ):
        """
        Aplikujeme podminky tak, aby v df zustaly jen polozky, ktere nas zajimaji
        :param df: vstupni dataframe ktery budeme filtrovat
        :param sklad: podminka na konkretni sklad
        :param cena_min:
        :param cena_max:
        :return:
        """
        mask = pd.Series(True, index=df.index)

        if sklad is not None:
            if isinstance(sklad, str):
                mask &= df["sklad"] == sklad
            else:  # list[str]
                mask &= df["sklad"].isin(sklad)

        if cena_min is not None:
            mask &= df["CenaSDph"] >= cena_min

        if cena_max is not None:
            mask &= df["CenaSDph"] <= cena_max

        return df[mask]

    def yesterday_not_today(
            self,
            dfs_yesterday_list,
            dfs_today_list,
            key="zaruky",
            **conditions
    ):
        """
        Applies yesterday_not_today logic pairwise on lists of DataFrames
        and returns the combined result.
        """

        if len(dfs_yesterday_list) != len(dfs_today_list):
            raise ValueError("dfs_yesterday_list and dfs_today_list must have the same length")

        results = []

        for df_yesterday, df_today in zip(dfs_yesterday_list, dfs_today_list):
            y_filt = self.apply_conditions(df_yesterday, **conditions)
            t_filt = self.apply_conditions(df_today, **conditions)

            df_ynt = y_filt[~y_filt[key].isin(t_filt[key])]
            results.append(df_ynt)

        return pd.concat(results, ignore_index=True)


    def determine_missing_styles(self, df_missing, df_today_list):
        """
        Urci styly (triplets), ktere se vyskytuji u zmizelych sperku,
        ale uz se nevyskytuji v zadnem z dnesnich sledovanych dataframes.

        :param df_missing: DataFrame se sloupcem style_set (set of triplets) s polozkami, ktere zmizely
        :param df_today_list: iterable DataFrame, kazdy se sloupcem style_set
        :return: set triplets
        """

        missing_styles = set().union(*df_missing["style_set"])
        # return missing_styles  # TODO: pouze debug, tento radek vymazat

        today_styles = set().union(
            *(
                style_set
                for df in df_today_list
                for style_set in df["style_set"]
            )
        )

        return missing_styles - today_styles

    def find_best_replacement_ids(self, missing_styles):
        """
        for each style, determine suitable items in export karty zbozi. select the one with best coefficient
        :param missing_styles: styles that disappeared from toay
        :return: items to produce
        """
        if len(missing_styles) == 0:
            return pd.DataFrame()
        forbidden_states = {"UKONCENO", "KUSOVKA", "PERSONALIZAK", "DOPRODEJ"}  # forbidden Ex_DesignStav values
        calc = CoefficientCalculator()
        dfs_to_produce = []
        for style in missing_styles:
            mask = (
                    ~self.df_exp_karty_zbozi["Ex_DesignStav"].isin(forbidden_states)
                    & self.df_exp_karty_zbozi["style_set"].apply(lambda s: style in s)
            )
            df_filtered = self.df_exp_karty_zbozi[mask]
            df_filtered = calc.calculate_coef(df_filtered, False)
            df_filtered.sort_values(by=["coefficient"], ascending=[False], inplace=True)
            dfs_to_produce.append(df_filtered.head(1))
        df_to_produce = pd.concat(dfs_to_produce, ignore_index=True)
        return df_to_produce

    def order_disappeared_styles(self):
        """
        hlavni funkce, vola vse potrebne
        :return:
        """
        df_missing_eans = self.yesterday_not_today(
            [self.df_exp_sklady_centrala_yesterday, self.df_exp_all_btqs_zapujcky_yesterday],
            [self.df_exp_sklady_centrala_today, self.df_exp_all_btqs_zapujcky_today],
            sklad=self.whs,
            cena_min=100000,
            cena_max=500000,
        )
        # TODO: misto dopredu vyfiltrovaneho self.dfexp_showroom_drahe_today by mohly byt na vstupu podminky filtrovani
        # jako v self.yesterday_not_today
        missing_styles_set = self.determine_missing_styles(df_missing_eans, df_today_list=[
            self.df_exp_showroom_drahe_today, self.df_zapujcky_recent, self.df_exp_all_btqs_zapujcky_today])
        df_best_replacement_ids = self.find_best_replacement_ids(missing_styles_set)

        return df_best_replacement_ids


if __name__ == '__main__':

    try:
        arg = sys.argv[1]
    except IndexError:
        arg = 0
    if arg == "snapshot":
        print("Copying data snapshots...")
        EXP_sklad.save_yesterday_data()
        sys.exit()
    EXP_sklad_cls = EXP_sklad(brand="A", whs=["EXP", "SHOWROOM"])
    df_order_disappeared_styles = EXP_sklad_cls.order_disappeared_styles()

