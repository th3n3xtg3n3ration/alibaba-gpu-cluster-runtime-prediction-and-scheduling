#!/usr/bin/env python3
"""
export_thesis_results.py
========================
Automated extraction of all publication-grade figures (PNG) and benchmark tables (HTML)
directly from executed Jupyter Research Notebooks (01 to 05).

Ensures 100% mathematical and visual consistency between the notebooks and thesis artifacts.

Usage:
    python scripts/export_thesis_results.py                 # Fast extraction from existing outputs (<1s)
    python scripts/export_thesis_results.py --execute       # Auto-executes empty notebooks if needed
    python scripts/export_thesis_results.py --force-execute # Re-runs all notebooks from scratch
    python scripts/export_thesis_results.py --lang tr       # Extracts from Turkish notebooks
"""

import argparse
import base64
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "results" / "figures" / "thesis_export"

PNG_DIR = EXPORT_DIR / "png"
HTML_DIR = EXPORT_DIR / "html"

HTML_STYLE_HEADER = (
    "<html><head><style>"
    "table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px} "
    "td,th{border:1px solid #ccc;padding:6px 10px} th{background:#f0f0f0}"
    "</style></head><body><div>\n"
)
HTML_STYLE_FOOTER = "\n</div></body></html>"

NOTEBOOKS_EN = [
    ("01_data_overview.ipynb", "NB01", True, False),
    ("02_workload_analysis.ipynb", "NB02", True, False),
    ("03_feature_engineering.ipynb", "NB03", True, False),
    ("04_runtime_prediction_models.ipynb", "NB04", True, True),
    ("05_scheduler_evaluation_32_gpu.ipynb", "NB05_32GPU", True, True),
    ("05_scheduler_evaluation_256_gpu.ipynb", "NB05_256GPU", True, True),
]

NOTEBOOKS_TR = [
    ("01_veri_ozeti.ipynb", "NB01", True, False),
    ("02_is_yuku_analizi.ipynb", "NB02", True, False),
    ("03_ozellik_muhendisligi.ipynb", "NB03", True, False),
    ("04_calisma_zamani_tahmin_modelleri.ipynb", "NB04", True, True),
    ("05_gorev_zamanlayici_degerlendirme_32_gpu.ipynb", "NB05_32GPU", True, True),
    ("05_gorev_zamanlayici_degerlendirme_256_gpu.ipynb", "NB05_256GPU", True, True),
]

def extract_from_nb_dict(nb: dict, prefix: str, extract_png: bool, extract_html: bool) -> tuple[int, int]:
    """Extract figures and tables from a notebook JSON dictionary."""
    fig_idx = 1
    table_idx = 1
    extracted_pngs = 0
    extracted_htmls = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        for out in cell.get("outputs", []):
            data = out.get("data", {})

            # Extract PNG
            if extract_png and "image/png" in data:
                png_b64 = data["image/png"]
                if isinstance(png_b64, list):
                    png_b64 = "".join(png_b64)
                png_bytes = base64.b64decode(png_b64)

                out_name = f"{prefix}-Figure{fig_idx:02d}.png"
                out_path = PNG_DIR / out_name
                with open(out_path, "wb") as f_img:
                    f_img.write(png_bytes)
                print(f"  [PNG]  {out_name}")
                fig_idx += 1
                extracted_pngs += 1

            # Extract HTML Table
            if extract_html and "text/html" in data:
                html_content = data["text/html"]
                if isinstance(html_content, list):
                    html_content = "".join(html_content)

                if "<table" in html_content:
                    full_html = HTML_STYLE_HEADER + html_content + HTML_STYLE_FOOTER
                    out_name = f"{prefix}_Table{table_idx:02d}.html"
                    out_path = HTML_DIR / out_name
                    with open(out_path, "w", encoding="utf-8") as f_tbl:
                        f_tbl.write(full_html)
                    print(f"  [HTML] {out_name}")
                    table_idx += 1
                    extracted_htmls += 1

    return extracted_pngs, extracted_htmls

def execute_notebook(nb_path: Path):
    """Execute a notebook and save the executed notebook with outputs."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    print(f"🔄 [ÇALIŞTIRILIYOR] '{nb_path.name}' arka planda çalıştırılıyor...")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_node = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    try:
        ep.preprocess(nb_node, {"metadata": {"path": str(nb_path.parent)}})
    except Exception as err:
        raise RuntimeError(f"❌ [HATA] '{nb_path.name}' çalıştırılırken hata oluştu:\n{err}") from err

    # Save the executed notebook back to disk
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_node, f)
    print(f"💾 [KAYDEDİLDİ] '{nb_path.name}' başarıyla tamamlandı ve çıktıları kaydedildi.")

def run_pipeline(lang: str = "en", auto_execute: bool = False, force_execute: bool = False):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    nb_dir = BASE_DIR / "notebooks" / lang
    notebooks_list = NOTEBOOKS_TR if lang == "tr" else NOTEBOOKS_EN

    total_pngs = 0
    total_htmls = 0

    print("=" * 65)
    print(f"Tez Çıktı Aktarımı ({lang.upper()}) | Mod: {'Force-Execute' if force_execute else ('Auto-Execute' if auto_execute else 'Hızlı Çıkarım')}")
    print("=" * 65)

    for nb_file, prefix, extract_png, extract_html in notebooks_list:
        nb_path = nb_dir / nb_file
        if not nb_path.exists():
            raise FileNotFoundError(f"❌ [HATA] Notebook dosyası bulunamadı: {nb_path}")

        # If force-execute is requested, run notebook directly
        if force_execute:
            execute_notebook(nb_path)

        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ [HATA] {nb_file} dosyası JSON olarak okunamadı: {e}") from e

        # Try extracting from existing outputs
        pngs, htmls = extract_from_nb_dict(nb, prefix, extract_png, extract_html)

        # If 0 outputs found
        if pngs == 0 and htmls == 0:
            if auto_execute:
                # Auto-execute empty notebook
                execute_notebook(nb_path)
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb_executed = json.load(f)
                pngs, htmls = extract_from_nb_dict(nb_executed, prefix, extract_png, extract_html)

                if pngs == 0 and htmls == 0:
                    raise RuntimeError(
                        f"❌ [HATA] '{nb_file}' çalıştırılmasına rağmen hiçbir görsel veya tablo çıktısı üretmedi!"
                    )
            else:
                print(
                    f"⚠️  [UYARI] '{nb_file}' çıktısı boş! Otomatik çalıştırmak için '--execute' bayrağını kullanabilirsiniz:\n"
                    f"     python scripts/export_thesis_results.py --execute\n"
                )

        total_pngs += pngs
        total_htmls += htmls

    print("=" * 65)
    print(f"İşlem Tamamlandı: Toplam {total_pngs} PNG Grafik ve {total_htmls} HTML Tablo aktarıldı.")
    print(f"Çıktı Dizini: {EXPORT_DIR}")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(
        description="Jupyter Notebook'lardan tüm görsel ve tabloları dışa aktarma aracı."
    )
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="Çıktısı boş olan notebook'ları otomatik olarak çalıştır ve kaydet."
    )
    parser.add_argument(
        "--force-execute",
        action="store_true",
        help="Tüm notebook'ları dolu olsa bile sıfırdan baştan çalıştırıp çıktıları yenile."
    )
    parser.add_argument(
        "--lang", "-l",
        choices=["en", "tr"],
        default="en",
        help="Hangi dil klasöründeki notebook'ların işleneceği (varsayılan: en)."
    )

    args = parser.parse_args()
    try:
        run_pipeline(lang=args.lang, auto_execute=args.execute, force_execute=args.force_execute)
    except Exception as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
