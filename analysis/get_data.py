"""Download the Cookie Cats A/B test data (about 2.7 MB) into data/."""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/robguilarr/ab_testing_cookie_cats/main/datasets/cookie_cats.csv"
out = Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv"
out.parent.mkdir(exist_ok=True)
urlretrieve(URL, out)
print(f"saved {out}")
