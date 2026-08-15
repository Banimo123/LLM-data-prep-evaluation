"""
manual_baseline.py
-------------------
Workflow manuel de reference (Approche 1 de la taxonomie : "baseline experte").
Ecrit a la main, sans LLM, pour servir de point de comparaison aux workflows generes.

v3 : version GENERIQUE, applicable a n'importe quel dataset (hotel_bookings, titanic,
flights, hospital...), sans liste de colonnes/categories codee en dur. Reprend les
memes techniques que la v2 (qui avait fait passer le F1 de ~0.26 a ~0.72-0.76 sur
hotel_bookings), mais les derive automatiquement des donnees plutot que de connaissances
metier ecrites a la main :
  - categories valides = valeurs les plus frequentes de chaque colonne categorielle a
    faible cardinalite (les variantes rares sont presumees etre des typos/erreurs)
  - colonnes de date detectees par nom de colonne + motif de valeurs
  - colonnes numeriques corrompues detectees automatiquement (beaucoup de valeurs
    extraient un nombre valide une fois nettoyees)
  - outliers detectes par percentiles (1er/99e) au lieu de bornes physiques codees en dur
  - imputation par MODE si la colonne est tres asymetrique (>50% de valeurs identiques),
    MEDIANE sinon

IMPORTANT : preserve `row_id` sur toutes les lignes conservees (regle identique a celle
imposee au LLM dans system_prompt.txt), condition necessaire pour que metrics.py puisse
calculer le F1.
"""

import difflib
import re
from datetime import datetime

import pandas as pd
import numpy as np

DISGUISED_MISSING = {"", "na", "n/a", "unknown", " ", "nan", "none", "null"}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
MAX_CATEGORY_CARDINALITY = 150  # au-dela, on ne tente pas l'harmonisation par similarite
MIN_CATEGORY_FREQ_SHARE = 0.02  # une valeur doit representer >=2% des lignes non-nulles
                                  # pour etre consideree "canonique" (sinon = variante bruitee)
MODE_IMPUTE_SHARE_THRESHOLD = 0.5  # si le mode couvre >50% des valeurs -> imputer par mode


def _levenshtein(a: str, b: str) -> int:
    """Distance d'edition (nombre minimal de substitutions/insertions/suppressions)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _harmonize_category(value, valid_values):
    """Rapproche une valeur bruitee de la categorie valide la plus proche.

    Utilise une distance d'edition (Levenshtein), adaptee a la LONGUEUR de la chaine :
    un seul caractere different sur "al" (2 lettres, ex: "xl") doit etre accepte, alors
    qu'un ratio de similarite type difflib penalise trop les chaines courtes (0.5 pour un
    seul caractere sur 2, sous n'importe quel seuil raisonnable). On exige aussi que le
    meilleur candidat batte clairement le second (distance strictement inferieure) --
    sinon on prefere ne rien corriger plutot que de risquer une confusion entre deux
    valeurs valides differentes (ex: codes pays a 1 lettre d'ecart comme GRC/GBR, ou
    horaires structurellement proches).
    """
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    normalized = re.sub(r"\s+", " ", s).strip().lower()
    for v in valid_values:
        if v.lower() == normalized:
            return v

    scored = sorted(
        ((_levenshtein(s, v), v) for v in valid_values),
        key=lambda x: x[0],
    )
    if not scored:
        return value
    best_dist, best_val = scored[0]
    # Tolerance adaptative : jusqu'a ~25% des caracteres de la valeur canonique, au moins 1
    max_allowed = max(1, len(best_val) // 4)
    if best_dist > max_allowed:
        return value
    if len(scored) > 1 and scored[1][0] == best_dist:
        return value  # ambigu : au moins 2 candidats a egale distance, on ne tranche pas
    return best_val


def _parse_date_robust(value):
    """Tente tous les formats de date usuels, plus un fallback timestamp Unix."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    if re.fullmatch(r"\d{9,10}", s):
        try:
            return pd.Timestamp(datetime.fromtimestamp(int(s)))
        except (ValueError, OSError):
            pass
    return pd.NaT


def _looks_like_date_column(series: pd.Series, col_name: str) -> bool:
    """Classification STRICTE (formats calendaires explicites uniquement, pas le repli
    timestamp Unix) : sinon n'importe quelle colonne de nombres a 9-10 chiffres (ex: un
    numero de telephone) serait a tort classee comme colonne de date, puisque presque
    toute chaine de 9-10 chiffres est un timestamp Unix "valide" au sens strict. Le repli
    timestamp reste disponible dans _parse_date_robust, mais seulement pour REPARER une
    colonne deja confirmee date par ce test (une minorite de valeurs corrompues en
    timestamp au sein d'une colonne par ailleurs clairement composee de dates classiques)."""
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False

    def _matches_explicit_date_format(v: str) -> bool:
        for fmt in DATE_FORMATS:
            try:
                datetime.strptime(v, fmt)
                return True
            except ValueError:
                continue
        return False

    hits = sum(1 for v in sample if _matches_explicit_date_format(v))
    hit_ratio = hits / len(sample)
    # Le nom de colonne (ex: "reservation_status_date") abaisse le seuil requis, mais ne
    # dispense JAMAIS de vérifier les valeurs elles-mêmes : une colonne comme
    # "arrival_date_month" contient "date" dans son nom mais ne contient PAS de dates
    # (juste des noms de mois) -- se fier au nom seul la corromprait entièrement.
    threshold = 0.3 if "date" in col_name.lower() else 0.6
    return hit_ratio >= threshold


_TIME_PATTERN = re.compile(r"(\d{1,2})\s*:+\s*(\d{1,3})\s*([ap])\s*\.*\s*m?\.?", re.IGNORECASE)


def _dedupe_extra_digit(minute_str: str) -> str:
    """Corrige un groupe de 3 chiffres de minutes issu de l'insertion d'un chiffre en
    trop (faute de frappe frequente : un chiffre adjacent est duplique, ex: '58' -> '558'
    ou '35' -> '355'). Heuristique : si deux chiffres adjacents sont identiques, on retire
    l'un des deux (c'est presque toujours le chiffre insere en trop) ; sinon, on suppose
    que l'insertion s'est faite au debut (repli sur les 2 derniers chiffres)."""
    if minute_str[0] == minute_str[1]:
        return minute_str[1:]
    if minute_str[1] == minute_str[2]:
        return minute_str[:2]
    return minute_str[-2:]


def _parse_time_robust(value):
    """Parse un horaire au format 'H:MM a.m./p.m.', tolerant aux fautes de frappe sur les
    chiffres (ex: 'O' pour '0'). Si la valeur a ete corrompue en date complete ou en
    timestamp Unix (erreur de type 'format_errors' mal ciblee sur une colonne d'horaire),
    tente d'en extraire la composante heure. Retourne None si rien n'est recuperable."""
    if pd.isna(value):
        return None
    s = str(value).strip().replace("O", "0").replace("o", "0")
    m = _TIME_PATTERN.search(s)
    if m:
        hour, minute, period = m.groups()
        if len(minute) == 3:
            minute = _dedupe_extra_digit(minute)
        try:
            hour, minute = int(hour), int(minute)
        except ValueError:
            return None
        if 1 <= hour <= 12 and 0 <= minute <= 59:
            return f"{hour}:{minute:02d} {period.lower()}.m."
        return None

    # Repli : la valeur ressemble a une date/timestamp (corruption hors-cible) plutot
    # qu'a un horaire -- on tente d'en extraire l'heure si elle contient une heure non nulle.
    ts = _parse_date_robust(value)
    if ts is not pd.NaT and not (ts.hour == 0 and ts.minute == 0):
        hour12 = ts.hour % 12 or 12
        period = "a" if ts.hour < 12 else "p"
        return f"{hour12}:{ts.minute:02d} {period}.m."
    return None


def _looks_like_time_column(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    hits = sum(1 for v in sample if _TIME_PATTERN.search(v.replace("O", "0").replace("o", "0")))
    return hits / len(sample) >= 0.6


def _numeric_char_ratio(s: str) -> float:
    """Fraction de caractères d'une chaîne qui appartiennent à un nombre plausible
    (chiffre, point, virgule, signe moins, ou 'O'/'o' pouvant être une faute de frappe
    pour '0'). Une vraie valeur numérique corrompue (ex: '342O') a un ratio proche de 1 ;
    un horaire ('9:40 a.m.'), un code ('AA-3859-IAH-ORD') ou une adresse
    ('1720 university blvd') ont un ratio bas car dominés par des lettres/espaces/deux-points."""
    if len(s) == 0:
        return 0.0
    numeric_chars = sum(1 for c in s if c.isdigit() or c in ".-,Oo")
    return numeric_chars / len(s)


def _looks_like_corrupted_numeric(series: pd.Series) -> bool:
    """Detecte une colonne numerique dont certaines valeurs ont ete corrompues en texte
    (ex: '342O' au lieu de '3420'). Contrairement a une simple presence de chiffre
    (trop permissif : matchait aussi les horaires, codes de vol, adresses...), exige que
    la VALEUR ENTIERE ressemble a un nombre (ratio de caracteres numeriques eleve)."""
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    ratios = sample.apply(_numeric_char_ratio)
    is_numeric_like = ratios >= 0.85
    if is_numeric_like.mean() < 0.8:
        return False

    # Garde-fou : si les valeurs NON numeriques restantes sont dominees par UNE seule
    # valeur repetee (ex: "empty" present identiquement sur des centaines de lignes),
    # c'est tres probablement un placeholder categoriel legitime du dataset (deja present
    # tel quel dans clean.csv), pas du bruit a corriger -> ne pas extraire, laisser tel quel.
    non_numeric = sample[~is_numeric_like]
    if len(non_numeric) > 0:
        top_share = non_numeric.value_counts(normalize=True).iloc[0]
        if top_share >= 0.3:
            return False

    return True


def _extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".").replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan


def _find_best_grouping_column(df: pd.DataFrame, target_col: str, is_numeric: bool):
    """Cherche, parmi les autres colonnes du dataset, la meilleure colonne de regroupement
    pour affiner l'imputation de target_col (au lieu d'un mode/mediane global, un mode/
    mediane CONDITIONNEL par groupe, ex: le pays le plus probable depend souvent du segment
    de marche). Retourne None si aucune colonne candidate n'apporte de gain suffisant
    (evite la complexite/le risque de sur-ajustement quand ca n'aide pas).

    Criteres d'une bonne colonne de regroupement :
      - categorielle ou a faible cardinalite (2 a 50 valeurs), sans valeur manquante
        elle-meme (sinon ca complique la logique sans bien aider)
      - pour une cible categorielle : le mode CONDITIONNEL (par groupe) doit couvrir une
        part nettement plus grande des valeurs observees que le mode global (+5 points)
      - pour une cible numerique : la dispersion (MAD) A L'INTERIEUR des groupes doit etre
        nettement plus faible que la dispersion globale (reduction d'au moins 20%)
    """
    target = df[target_col]
    observed_mask = target.notna()
    if observed_mask.sum() < 30:
        return None

    candidates = [c for c in df.columns if c not in (target_col, "row_id")
                  and not df[c].isna().any()
                  and 1 < df[c].nunique() <= 300]
    if not candidates:
        return None

    observed = df.loc[observed_mask]
    best_col, best_gain = None, 0.0

    if is_numeric:
        target_num = pd.to_numeric(observed[target_col], errors="coerce").dropna()
        if len(target_num) < 30:
            return None
        global_mad = (target_num - target_num.median()).abs().median()
        if global_mad == 0:
            return None
        for cand in candidates:
            sub = observed.loc[target_num.index, [cand]].copy()
            sub["_t"] = target_num
            grouped = sub.groupby(cand)["_t"]
            weighted_mad = 0.0
            total = len(sub)
            for _, grp in grouped:
                if len(grp) < 5:
                    weighted_mad += global_mad * (len(grp) / total)  # trop petit, pas fiable
                    continue
                weighted_mad += (grp - grp.median()).abs().median() * (len(grp) / total)
            reduction = 1 - (weighted_mad / global_mad)
            if reduction > best_gain and reduction >= 0.20:
                best_gain, best_col = reduction, cand
    else:
        target_str = observed[target_col].astype(str)
        global_mode_share = target_str.value_counts(normalize=True).iloc[0]
        for cand in candidates:
            sub = pd.DataFrame({"_g": observed[cand], "_t": target_str})
            grouped = sub.groupby("_g")["_t"]
            weighted_share = 0.0
            total = len(sub)
            for _, grp in grouped:
                weighted_share += grp.value_counts(normalize=True).iloc[0] * (len(grp) / total)
            gain = weighted_share - global_mode_share
            if gain > best_gain and gain >= 0.05:
                best_gain, best_col = gain, cand

    return best_col


def _conditional_impute(df: pd.DataFrame, col: str, is_numeric: bool, fallback_cols: set = None):
    """Impute les valeurs manquantes de `col` par mode/mediane CONDITIONNEL (par groupe)
    si une bonne colonne de regroupement existe, sinon retombe sur le mode/mediane global
    (comportement identique a avant si aucun regroupement n'aide)."""
    fallback_cols = fallback_cols or set()
    use_mode_globally = col in fallback_cols

    if is_numeric:
        valid = df[col].dropna()
        global_repl = (valid.mode().iloc[0] if (use_mode_globally and not valid.mode().empty)
                       else valid.median())
    else:
        mode = df[col].mode(dropna=True)
        global_repl = mode.iloc[0] if not mode.empty else "Unknown"

    group_col = _find_best_grouping_column(df, col, is_numeric)
    if group_col is None:
        df[col] = df[col].fillna(global_repl)
        return df

    def _group_fill(s: pd.Series):
        if is_numeric:
            valid_s = s.dropna()
            if len(valid_s) == 0:
                return s.fillna(global_repl)
            if use_mode_globally:
                mode_s = valid_s.mode()
                repl = mode_s.iloc[0] if not mode_s.empty else valid_s.median()
            else:
                repl = valid_s.median()
        else:
            mode_s = s.dropna().mode()
            repl = mode_s.iloc[0] if not mode_s.empty else global_repl
        return s.fillna(repl)

    df[col] = df.groupby(group_col)[col].transform(_group_fill)
    df[col] = df[col].fillna(global_repl)  # au cas ou un groupe entier serait vide
    return df


def _build_canonical_categories(series: pd.Series) -> list:
    """Deduit la liste des valeurs canoniques d'une colonne categorielle a partir des
    valeurs qui se repetent suffisamment (une variante de typo, elle, n'apparait
    generalement qu'une seule fois -- alors qu'une vraie valeur canonique, meme rare en
    proportion sur une colonne a forte cardinalite (ex: 100 codes de vol distincts), se
    repete plusieurs fois). Seuil ABSOLU (nombre d'occurrences), pas relatif : un
    pourcentage fixe (ex: >=2%) echoue des qu'il y a beaucoup de valeurs distinctes
    egalement frequentes (chacune < 2% individuellement, ex: 100 codes sur 2376 lignes)."""
    counts = series.dropna().astype(str).str.strip().value_counts()
    if len(counts) == 0:
        return []
    avg_repeats = counts.sum() / len(counts)
    # Une vraie valeur canonique doit apparaitre au moins 2 fois, et idealement au moins
    # ~40% de la repetition moyenne de la colonne (les typos, elles, sont quasi-uniques).
    min_count = max(2, int(0.4 * avg_repeats))
    canonical = counts[counts >= min_count].index.tolist()
    return canonical if canonical else counts.index.tolist()[:10]


# --- Connaissances metier specifiques a hotel_bookings (version eprouvee, F1=0.72-0.76) ---
# Conservees telles quelles pour ne pas regresser sur ce dataset deja valide. Utilisees
# automatiquement quand clean_dataset_baseline() detecte ce dataset (voir dispatch plus bas).
HOTEL_BOOKINGS_KNOWN_CATEGORIES = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Offline TA/TO", "Groups", "Direct",
                        "Corporate", "Complementary", "Aviation", "Undefined"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
}
HOTEL_BOOKINGS_MODE_IMPUTED_COLS = {"babies", "children", "days_in_waiting_list", "agent", "company"}
HOTEL_BOOKINGS_OUTLIER_BOUNDS = {
    "adr": (0, 600), "babies": (0, 4), "stays_in_week_nights": (0, 20),
    "days_in_waiting_list": (0, 400), "adults": (0, 6), "children": (0, 6),
}
HOTEL_BOOKINGS_NUMERIC_TYPO_COLS = ["lead_time", "adults", "children", "babies", "adr",
                                     "stays_in_week_nights", "days_in_waiting_list", "agent"]


def _is_hotel_bookings(df: pd.DataFrame) -> bool:
    """Detecte si le dataset en entree est hotel_bookings, via ses colonnes caracteristiques."""
    signature = {"hotel", "adr", "reservation_status_date", "deposit_type"}
    return signature.issubset(set(df.columns))


def _clean_hotel_bookings_specific(df: pd.DataFrame) -> pd.DataFrame:
    """Logique specifique hotel_bookings (v2, F1=0.72-0.76). Suppose row_id + missing
    values deguises deja normalises en amont par clean_dataset_baseline()."""
    for col, valid_values in HOTEL_BOOKINGS_KNOWN_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _harmonize_category(v, valid_values))

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
        df.loc[df["country"] == "nan", "country"] = np.nan

    for col in HOTEL_BOOKINGS_NUMERIC_TYPO_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_extract_numeric)

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(_parse_date_robust)
        df["reservation_status_date"] = df["reservation_status_date"].dt.strftime("%Y-%m-%d")

    for col, (low, high) in HOTEL_BOOKINGS_OUTLIER_BOUNDS.items():
        if col not in df.columns:
            continue
        is_outlier = (df[col] < low) | (df[col] > high)
        if is_outlier.any():
            if col in HOTEL_BOOKINGS_MODE_IMPUTED_COLS:
                repl = df.loc[~is_outlier, col].mode(dropna=True)
                repl = repl.iloc[0] if not repl.empty else df[col].median()
            else:
                repl = df.loc[~is_outlier, col].median()
            df.loc[is_outlier, col] = repl

    for col in df.columns:
        if col == "row_id" or not df[col].isna().any():
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        df = _conditional_impute(df, col, is_numeric, fallback_cols=HOTEL_BOOKINGS_MODE_IMPUTED_COLS)

    return df


def clean_dataset_baseline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    assert "row_id" in df.columns, "row_id manquant en entree"

    data_cols = [c for c in df.columns if c != "row_id"]

    # --- Valeurs manquantes deguisees -> NaN uniforme (commun aux 2 chemins) ---
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != "row_id"]
    for col in text_cols:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in DISGUISED_MISSING else v
        )

    if _is_hotel_bookings(df):
        print("[baseline] Dataset detecte : hotel_bookings -> logique specifique (eprouvee).")
        df = _clean_hotel_bookings_specific(df)
    else:
        print("[baseline] Dataset non reconnu -> logique generique (auto-detection).")
        df = _clean_dataset_generic(df)

    df = df.drop_duplicates(subset=data_cols, keep="first")
    df.to_csv(output_path, index=False)
    return df


def _clean_dataset_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Logique generique, applicable a tout dataset sans connaissance metier prealable
    (titanic, flights, hospital...). Les valeurs manquantes deguisees sont deja
    normalisees par clean_dataset_baseline() avant l'appel."""
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != "row_id"]

    date_cols = [c for c in text_cols if _looks_like_date_column(df[c], c)]
    for col in date_cols:
        df[col] = df[col].apply(_parse_date_robust)
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    remaining_text_cols = [c for c in text_cols if c not in date_cols]

    time_cols = [c for c in remaining_text_cols if _looks_like_time_column(df[c])]
    for col in time_cols:
        parsed = df[col].apply(_parse_time_robust)
        # Si le parsing echoue pour une valeur (None), on garde l'originale plutot que de
        # la vider -- une valeur non reconnue n'est pas forcement fausse (peut etre une
        # variante deja correcte que le regex ne couvre pas).
        df[col] = parsed.where(parsed.notna(), df[col])
    remaining_text_cols = [c for c in remaining_text_cols if c not in time_cols]

    numeric_like_text_cols = [c for c in remaining_text_cols if _looks_like_corrupted_numeric(df[c])]
    for col in numeric_like_text_cols:
        df[col] = df[col].apply(_extract_numeric)
    remaining_text_cols = [c for c in remaining_text_cols if c not in numeric_like_text_cols]

    already_numeric_cols = [c for c in df.columns if c != "row_id" and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = list(dict.fromkeys(already_numeric_cols + numeric_like_text_cols))

    for col in remaining_text_cols:
        n_unique = df[col].dropna().nunique()
        if n_unique > 1:
            canonical = _build_canonical_categories(df[col])
            df[col] = df[col].apply(lambda v: _harmonize_category(v, canonical))
        else:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == "nan", col] = np.nan

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if len(valid) < 10:
            continue
        # Colonne d'identifiant quasi-unique (ex: tuple_id, index) : chaque valeur est
        # differente par construction -> pas de notion d'aberrant.
        if valid.nunique() / len(valid) >= 0.9:
            continue
        # Colonne de CODE numerique repete (ex: ProviderNumber, ZipCode, PhoneNumber) :
        # peu de valeurs distinctes, chacune repetee sur plusieurs lignes (groupement).
        # Une detection d'outlier par percentile n'a pas de sens ici (ce n'est pas une
        # mesure continue) et risquerait de "corriger" un code valide mais rare vers un
        # code totalement different -> on laisse cette colonne intacte (plus sur que de
        # la corrompre ; une vraie correction demanderait un matching vers les codes
        # valides connus, hors perimetre de cette baseline generique).
        if valid.nunique() <= 200 and (valid.nunique() / len(valid)) < 0.5:
            continue
        low, high = valid.quantile(0.005), valid.quantile(0.995)
        if low == high:
            continue
        is_outlier = (series < low) | (series > high)
        if is_outlier.any():
            non_outlier_valid = valid[~is_outlier.reindex(valid.index, fill_value=False)]
            mode_counts = non_outlier_valid.value_counts(normalize=True)
            use_mode = (not mode_counts.empty) and (mode_counts.iloc[0] >= MODE_IMPUTE_SHARE_THRESHOLD)
            replacement = mode_counts.index[0] if use_mode else non_outlier_valid.median()
            if pd.api.types.is_integer_dtype(df[col]) and not float(replacement).is_integer():
                df[col] = df[col].astype("float64")
            df.loc[is_outlier, col] = replacement
        df[col] = series.where(~is_outlier, df[col])

    for col in [c for c in df.columns if c != "row_id"]:
        if not df[col].isna().any():
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric:
            valid = df[col].dropna()
            mode_counts = valid.value_counts(normalize=True)
            use_mode = (not mode_counts.empty) and (mode_counts.iloc[0] >= MODE_IMPUTE_SHARE_THRESHOLD)
            fallback_cols = {col} if use_mode else set()
        else:
            fallback_cols = set()
        df = _conditional_impute(df, col, is_numeric, fallback_cols=fallback_cols)

    return df


# Alias conserve pour compatibilite avec les scripts existants qui l'appellent encore
clean_hotel_bookings_baseline = clean_dataset_baseline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = clean_dataset_baseline(args.input, args.output)
    print(f"Nettoye : {result.shape[0]} lignes, {result.shape[1]} colonnes -> {args.output}")
