# pokemkt_app.py
# --------------------------------------------------------------------------------------
# Objectif
# - Scraper CardMarket (FR) pour des cartes Pokémon et produits scellés en FR
# - Conserver le prix en Near Mint (et mieux) + image, set/extension
# - Ajouter une 2e colonne "Dernières ventes eBay" (ventes réussies)
# - Générer pour chaque carte un graphique d'évolution de prix (basé sur ventes eBay)
# - Rechercher/filtrer par extensions
#
# ⚠️ Notes importantes (lire !)
# - Respecte les CGU/robots.txt de CardMarket/eBay. Ce code est un PROTOTYPE éducatif.
# - CardMarket et eBay peuvent changer leurs sélecteurs / anti-bot => adapte si besoin.
# - eBay ne fournit publiquement que ~90 jours de ventes réalisées. POUR 2 ANS :
#     * Si tu obtiens l'API CardMarket (ou un endpoint du graphique de la carte),
#       tu peux fusionner ces historiques pour couvrir 24 mois.
#     * En attendant, on trace un historique basé sur les ventes eBay (jusqu'à 90 jours)
#       pour donner le visuel et l'architecture ; ça s'étendra si tu ajoutes une source 24 mois.
# - Pour l'image : on sauvegarde l'image principale de la fiche CardMarket.
#
# Dépendances :
#   pip install playwright pysimplegui requests beautifulsoup4 lxml pandas matplotlib python-dateutil
#   python -m playwright install
# --------------------------------------------------------------------------------------

import os
import re
import csv
import time
import json
import string
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import PySimpleGUI as sg
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from playwright.sync_api import sync_playwright

# ------------------------------ Config -----------------------------------
APP_NAME = "PokéMarket Scraper"
BASE_OUT = Path("output")
IMG_DIR = BASE_OUT / "images"
CHART_DIR = BASE_OUT / "charts"
CSV_PATH = BASE_OUT / "cards.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

REQUESTS_TIMEOUT = 25

@dataclass
class CardRow:
    title: str
    set_name: str
    language: str
    condition_basis: str
    cm_min_nm_price_eur: float | None
    cm_url: str
    image_file: str | None
    ebay_last_sold_summary: str
    ebay_last_sold_count_90d: int
    ebay_avg_90d_eur: float | None
    ebay_last_sale_date: str | None

# --------------------------- Structures de données -----------------------
@dataclass
class CardRow:
    title: str
    set_name: str
    language: str
    condition_basis: str  # "NM+" si Near Mint et mieux
    cm_min_nm_price_eur: float | None
    cm_url: str
    image_file: str | None
    ebay_last_sold_summary: str  # e.g. "7 ventes / 30j, moy=82.3€, médiane=81€"
    ebay_last_sold_count_90d: int
    ebay_avg_90d_eur: float | None
    ebay_last_sale_date: str | None

# --------------------------- Utilitaires ---------------------------------
SAFE_CHARS = f"-_.() {string.ascii_letters}{string.digits}"

def slugify(text: str) -> str:
    s = "".join(c for c in text if c in SAFE_CHARS).strip().replace(" ", "_")
    return re.sub(r"_+", "_", s)[:100]


def ensure_dirs():
    BASE_OUT.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True, parents=True)
    CHART_DIR.mkdir(exist_ok=True, parents=True)


def save_csv(rows: list[CardRow]):
    ensure_dirs()
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df


def save_image(url: str, dest: Path) -> bool:
    try:
        h = {"User-Agent": USER_AGENT, "Referer": "https://www.cardmarket.com/"}
        r = requests.get(url, headers=h, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False

def safe_theme(theme_name="LightBlue2"):
    if hasattr(sg, "theme"):
        sg.theme(theme_name)
    else:
        sg.change_look_and_feel(theme_name)

# --------------------------- CardMarket scraping --------------------------
CM_HEADERS = {"User-Agent": USER_AGENT}

SET_ANCHOR_RE = re.compile(r"/fr/Pokemon/Products/.*/(.*)")


def parse_cm_product_page(html: str, product_url: str) -> dict:
    """Extrait infos principales + prix NM min si possible.
    Retourne dict avec: title, set_name, image_url, nm_prices (list[float]), language_guess
    """
    soup = BeautifulSoup(html, "lxml")

    # Titre
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Set / extension (après libellé "Édité dans")
    set_name = ""
    for label in soup.select("div, dt"):
        if label.get_text(strip=True).lower().startswith("édité dans"):
            # Le lien suivant est le set
            nxt = label.find_next("a")
            if nxt:
                set_name = nxt.get_text(strip=True)
            break
    if not set_name:
        # fallback via fil d'ariane
        bc = soup.select_one("ol.breadcrumb a[href*='/Pokemon/Products/'] + a + a")
        if bc:
            set_name = bc.get_text(strip=True)

    # Image principale
    image_url = None
    img = soup.select_one("img[alt][src*='cardmarket']") or soup.select_one("picture img")
    if img and img.get("src"):
        image_url = img["src"]
        if image_url.startswith("//"):
            image_url = "https:" + image_url

    # Tableau des offres -> chercher lignes où la condition contient NM ou Near Mint
    nm_prices = []
    for tr in soup.select("table tr"):
        txt = tr.get_text(" ", strip=True).lower()
        if not txt:
            continue
        # repère condition
        if (" near mint" in txt) or (" nm" in txt) or ("mint" in txt and "near" in txt):
            # extraire un prix en €
            m = re.search(r"([0-9]+[.,][0-9]{2})\s*€", txt)
            if m:
                nm_prices.append(float(m.group(1).replace(",", ".")))

    language_guess = "Français" if "/fr/" in product_url else ""

    return {
        "title": title,
        "set_name": set_name,
        "image_url": image_url,
        "nm_prices": nm_prices,
        "language": language_guess or "Français",
    }


def fetch_cardmarket_product(product_url: str) -> dict:
    with requests.Session() as s:
        s.headers.update(CM_HEADERS)
        # Accepter les cookies via param (si popup) : on tente une première requête
        r = s.get(product_url, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        html = r.text
        return parse_cm_product_page(html, product_url)

# --------------------------- eBay "sold" scraping -------------------------
# On construit une requête eBay FR pour ventes réussies & complétées, en euros.
# Exemple de pattern de requête :
#   https://www.ebay.fr/sch/i.html?_nkw=Tortank+ex+MEW200+151+francais&_sacat=183454&LH_Sold=1&LH_Complete=1&LH_PrefLoc=1

EBAY_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"}


def make_ebay_query(card_title: str, set_name: str) -> str:
    base = "https://www.ebay.fr/sch/i.html"
    # heuristique : on enlève ce qui est entre parenthèses, garde nom FR si dispo
    t = re.sub(r"\(.*?\)", "", card_title).strip()
    # ajouter set et filtre français
    q = f"{t} {set_name} français"
    q = re.sub(r"\s+", "+", q)
    params = "_sacat=183454&LH_Sold=1&LH_Complete=1&LH_PrefLoc=1&_ipg=200"
    return f"{base}?_nkw={q}&{params}"


def parse_ebay_sold(html: str) -> list[dict]:
    """Retourne une liste {title, price_eur, date} à partir de la page résultats eBay.
    eBay change souvent ses classes; on reste robuste via texte.
    """
    soup = BeautifulSoup(html, "lxml")
    items = []
    # chaque résultat : li.s-item ou div.s-item__wrapper
    for li in soup.select("li.s-item, div.s-item__wrapper"):
        title_el = li.select_one(".s-item__title")
        price_el = li.select_one(".s-item__price, .s-item__detail--primary span")
        date_el = li.find(text=re.compile(r"Vendu le|Sold"))
        if not title_el or not price_el:
            continue
        title = title_el.get_text(strip=True)
        # prix en EUR
        m = re.search(r"([0-9]+[.,][0-9]{2})\s*€", price_el.get_text())
        if not m:
            continue
        price = float(m.group(1).replace(",", "."))
        # date
        date = None
        if date_el:
            dm = re.search(r"(\d{1,2}\s\w+\s\d{4})", date_el)
            if dm:
                try:
                    date = dateparser.parse(dm.group(1), dayfirst=True).date()
                except Exception:
                    date = None
        items.append({"title": title, "price_eur": price, "date": date})
    return items


def fetch_ebay_sold(card_title: str, set_name: str) -> list[dict]:
    url = make_ebay_query(card_title, set_name)
    with requests.Session() as s:
        s.headers.update(EBAY_HEADERS)
        r = s.get(url, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return parse_ebay_sold(r.text)

# --------------------------- Graphiques -----------------------------------

def plot_price_history(card_title: str, set_name: str, sold_items: list[dict]) -> str | None:
    """Crée un graphique PNG basé sur les ventes eBay (prix vs date). Retourne le chemin du fichier."""
    if not sold_items:
        return None
    ensure_dirs()
    # garder uniquement ceux avec date
    pts = [(it["date"], it["price_eur"]) for it in sold_items if it.get("date")]
    if not pts:
        return None
    pts.sort(key=lambda x: x[0])

    dates = [pd.to_datetime(d) for d, _ in pts]
    prices = [p for _, p in pts]

    plt.figure()
    plt.plot(dates, prices)
    plt.title(f"{card_title} — eBay ventes réussies")
    plt.xlabel("Date")
    plt.ylabel("Prix (EUR)")

    fn = CHART_DIR / f"{slugify(card_title)}.png"
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
    return str(fn)

# --------------------------- Pipeline carte --------------------------------

def process_card(product_url: str) -> CardRow | None:
    try:
        cm = fetch_cardmarket_product(product_url)
    except Exception as e:
        print("CardMarket échec:", e)
        return None

    title = cm.get("title") or ""
    set_name = cm.get("set_name") or ""
    image_url = cm.get("image_url")
    nm_prices = cm.get("nm_prices") or []

    cm_min = min(nm_prices) if nm_prices else None

    # image
    image_file = None
    if image_url:
        image_file = IMG_DIR / f"{slugify(title)}.jpg"
        ok = save_image(image_url, image_file)
        image_file = str(image_file) if ok else None

    # eBay sold
    try:
        sold = fetch_ebay_sold(title, set_name)
    except Exception as e:
        print("eBay échec:", e)
        sold = []

    # stats eBay (90j environ)
    sold_prices = [x["price_eur"] for x in sold]
    count = len(sold_prices)
    avg = float(sum(sold_prices) / count) if count else None
    last_date = None
    if sold:
        dated = [x for x in sold if x.get("date")]
        if dated:
            last_date = max(x["date"] for x in dated).isoformat()

    # graphique
    chart_path = plot_price_history(title, set_name, sold)
    if chart_path:
        print(f"Graphique créé: {chart_path}")

    summary = (
        f"{count} ventes ~90j, "
        f"moy={avg:.2f}€" if avg is not None else "aucune"
    )

    return CardRow(
        title=title,
        set_name=set_name,
        language=cm.get("language") or "",
        condition_basis="NM+",
        cm_min_nm_price_eur=cm_min,
        cm_url=product_url,
        image_file=image_file,
        ebay_last_sold_summary=summary,
        ebay_last_sold_count_90d=count,
        ebay_avg_90d_eur=avg,
        ebay_last_sale_date=last_date,
    )

# --------------------------- UI (PySimpleGUI) ------------------------------
HELP_TEXT = (
    "Colle des URLs produit CardMarket (FR), une par ligne.\n"
    "Clique 'Scraper' pour récupérer NM (FR) + Image + Ventes eBay.\n"
    "Utilise le champ 'Filtre extension' pour filtrer l'affichage par set (ex: 151).\n"
    "Les résultats (CSV + images + charts) sont dans le dossier 'output/'."
)


def run_ui():
    ensure_dirs()
    sg.theme("LightBlue2")

    layout = [
        [sg.Text(APP_NAME, font=("Segoe UI", 14, "bold"))],
        [sg.Multiline(size=(100, 8), key="-URLS-", default_text="https://www.cardmarket.com/fr/Pokemon/Products/Singles/151/Blastoise-ex-V3-MEW200")],
        [sg.Button("Scraper", key="-RUN-"), sg.Button("Ouvrir dossier"), sg.Text("Filtre extension:"), sg.Input(key="-SETFILTER-", size=(20,1)), sg.Button("Filtrer")],
        [sg.Table(values=[], headings=[
            "Titre", "Extension", "Langue", "CM min (NM)", "eBay (90j)", "Dernière vente", "URL"
        ], key="-TABLE-", auto_size_columns=False, col_widths=[35,16,8,10,22,12,40], enable_events=True, justification="left")],
        [sg.Text("Aide:"), sg.Text(HELP_TEXT)],
        [sg.Output(size=(120, 12))],
    ]

    win = sg.Window(APP_NAME, layout, finalize=True, resizable=True)

    rows: list[CardRow] = []

    def refresh_table(filter_set: str = ""):
        filtered = [r for r in rows if (not filter_set or filter_set.lower() in (r.set_name or "").lower())]
        data = [
            [r.title, r.set_name, r.language, f"{r.cm_min_nm_price_eur:.2f}€" if r.cm_min_nm_price_eur else "-",
             r.ebay_last_sold_summary, r.ebay_last_sale_date or "-", r.cm_url]
            for r in filtered
        ]
        win["-TABLE-"].update(values=data)

    while True:
        ev, vals = win.read(timeout=100)
        if ev in (sg.WINDOW_CLOSED, "Quitter"):
            break
        if ev == "Ouvrir dossier":
            os.startfile(str(BASE_OUT.resolve()))
        if ev == "Filtrer":
            refresh_table(vals["-SETFILTER-"])
        if ev == "-RUN-":
            url_text = vals["-URLS-"] or ""
            urls = [u.strip() for u in url_text.splitlines() if u.strip()]
            out: list[CardRow] = []
            for u in urls:
                print(f"→ Traitement: {u}")
                row = process_card(u)
                if row:
                    out.append(row)
                time.sleep(1.0)
            if out:
                rows.extend(out)
                save_csv(rows)
                print(f"Sauvegardé: {CSV_PATH}")
                refresh_table(vals["-SETFILTER-"])
            else:
                print("Aucune donnée récupérée.")

    win.close()


if __name__ == "__main__":
    run_ui()
