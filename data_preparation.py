"""
Legacy data preparation script - maintained for compatibility
Use data/data_loader.py for new implementations
"""

import pandas as pd
import warnings

warnings.warn(
    "This module is deprecated. Use data.data_loader instead.", DeprecationWarning
)

# Import the new data loader
try:
    from data.data_loader import DataLoader

    def prepare_data():
        """Prepare data using new data loader"""
        loader = DataLoader()
        return loader.create_integrated_dataset()

except ImportError:
    # Fallback to original implementation
    def prepare_data():
        """Original data preparation implementation"""

        mar = pd.read_csv(
            "/Volumes/MARAL/CSV/F01/V2AR1001.csv",
            sep=";",
            encoding="cp850",
            parse_dates=["LANUMMER", "WARENGR"],
        )
        mar["LANUMMER"] = mar["LANUMMER"].str.strip()
        mar = mar.drop_duplicates(subset="LANUMMER")

        nf_list = []
        for land in ["F01", "F02", "F03", "F04"]:
            nf_land = pd.read_csv(
                f"/Volumes/MARAL/CSV/{land}/V2SC1010.csv",
                sep=";",
                encoding="cp850",
                parse_dates=["DATUM"],
            )
            nf_list.append(nf_land)

        nf = pd.concat(nf_list, ignore_index=True)

        nf["ART_NR"] = nf["ART_NR"].str.strip()
        nf["GROESSE"] = nf["GROESSE"].str.strip().fillna("")
        nf["FARBE"] = nf["FARBE"].str.strip().fillna("")
        nf["LANUMMER"] = (
            nf["ART_NR"].str.ljust(8)
            + nf["GROESSE"].str.ljust(4)
            + nf["FARBE"].str.ljust(2)
        )
        nf["LANUMMER"] = nf["LANUMMER"].str.strip()
        nf["UNITPREIS"] = (nf["PREIS"] + nf["MWST"]) / nf["MENGE"]
        nf = nf[nf["UNITPREIS"] > 0]
        nf["MONAT"] = pd.to_datetime(nf["DATUM"], errors="coerce")
        nf["MONAT"] = nf["MONAT"].dt.to_period("M").dt.to_timestamp()

        nf = nf[
            nf["ART_NR"].str.contains(
                r"VK|PUVIS|UN|PUV", na=False, case=False, regex=True
            )
            == False
        ]

        nf_monatlich = (
            nf.groupby(["LANUMMER", "MONAT"])
            .agg(anz_produkt=("MENGE", "sum"), unit_preis=("UNITPREIS", "mean"))
            .reset_index()
        )
        nf_monatlich["anz_produkt"] = nf_monatlich["anz_produkt"].apply(int)
        nf_monatlich["unit_preis"] = nf_monatlich["unit_preis"].apply(
            lambda x: round(x, 2)
        )

        nfmar = nf_monatlich.merge(
            mar[["LANUMMER", "WARENGR"]], on="LANUMMER", how="left"
        )
        nfmar = nfmar[nfmar["WARENGR"].notna()]
        nfmar = nfmar.rename(
            columns={"WARENGR": "product_category_id", "LANUMMER": "product_id"}
        )

        return nfmar


# For backward compatibility
if __name__ == "__main__":
    data = prepare_data()
    data.to_excel("present/data_since_2020-01.xlsx", index=False)
    print(f"Prepared {len(data)} records")
