"""eda_toolkit CLI.

A :class:`click.Group` with a default command so backward-compatible usage
``eda <input> --report html,excel,pdf`` keeps working, while new subcommands
``eda ts ...`` and ``eda model ...`` (Faz-3) are also available.

Examples:
    eda musteri.csv --report html,excel,pdf,docx,pptx --cleaning moderate --pii mask
    eda a.csv --join b.csv --on musteri_id --report excel
    eda --url https://host/rapor.csv --report html
    eda ts satislar.csv --col tutar --horizon 6 --method sarimax --period 12
    eda model musteri.xlsx --target churn --features yas,gelir --kind logistic
"""
from __future__ import annotations

import tempfile

import click
import pandas as pd

from eda_toolkit.types import (
    PipelineConfig,
    CleaningLevel,
    PiiMode,
    MaskMode,
)
from eda_toolkit.pipeline import run
from eda_toolkit.loader import load_many, load_from_url
from eda_toolkit.analytics.timeseries import decompose, forecast
from eda_toolkit.analytics.modeling import fit_regression


class _DefaultGroup(click.Group):
    """A click Group that routes unknown/leading tokens to a default command.

    Lets ``eda <input> ...`` and ``eda --url ...`` (no subcommand) fall through
    to the ``report`` command, while ``eda ts ...`` / ``eda model ...`` resolve
    normally.
    """

    def __init__(self, *args, default: str | None = None, **kwargs):
        self.default_cmd_name = default
        super().__init__(*args, **kwargs)

    def resolve_command(self, ctx, args):
        if args and args[0] in ("--help", "-h", "--version"):
            return super().resolve_command(ctx, args)
        if args and args[0] not in self.commands:
            cmd = self.commands[self.default_cmd_name]
            return cmd.name, cmd, args
        return super().resolve_command(ctx, args)


@click.group(cls=_DefaultGroup, default="report", invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """eda_toolkit: rapor uretimi + zaman serisi/regresyon analizi."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── report (default) ───────────────────────────────────────────────────────


@cli.command("report")
@click.argument("input", type=click.Path(exists=True), required=False)
@click.option("--report", default="html", help="Format(s): html,excel,pdf,docx,pptx")
@click.option("--cleaning", default="conservative",
              type=click.Choice(["conservative", "moderate", "aggressive"]))
@click.option("--pii", default="warn", type=click.Choice(["warn", "mask"]))
@click.option("--mask-mode", default="partial",
              type=click.Choice(["partial", "full", "hash", "tokenize", "pseudonymize"]))
@click.option("--show-last", type=int, default=4)
@click.option("--lang", default="tr")
@click.option("--url", default=None, help="CSV/JSON/XLSX indirilecek URL (INPUT yerine)")
@click.option("--join", multiple=True, help="Birlestirilecek ek dosya (tekrarlanabilir)")
@click.option("--on", default=None, help="Join anahtari (--join ile)")
def report_cmd(input: str | None, report: str, cleaning: str, pii: str,
               mask_mode: str, show_last: int, lang: str, url: str | None,
               join: tuple[str, ...], on: str | None) -> None:
    """Veri uzerinde temizleme + EDA + PII + rapor uretir."""
    input_path = _resolve_input(input, url, join, on)
    config = PipelineConfig(
        report_formats=[f.strip() for f in report.split(",") if f.strip()],
        cleaning=CleaningLevel(cleaning),
        pii=PiiMode(pii),
        mask_mode=MaskMode(mask_mode),
        show_last=show_last,
        lang=lang,
    )
    result = run(input_path, config)
    for fmt, path in result.reports.items():
        click.echo(f"[{fmt}] {path}")
    _print_scores(result)


# ── ts ─────────────────────────────────────────────────────────────────────


@cli.command("ts")
@click.argument("input", type=click.Path(exists=True))
@click.option("--col", required=True, help="Analiz edilecek sayisal sutun")
@click.option("--horizon", type=int, default=6, show_default=True)
@click.option("--method", default="arima", type=click.Choice(["arima", "sarimax"]))
@click.option("--period", type=int, default=None, help="Mevsim periyodu (sarimax icin)")
def ts_cmd(input: str, col: str, horizon: int, method: str, period: int | None) -> None:
    """Zaman serisi: ayrastirma + tahmin (statsmodels ARIMA/SARIMAX)."""
    df = _read_input(input)
    click.echo("--- Ayrastirma ---")
    d = decompose(df, col, period=period)
    if "error" in d:
        click.echo(f"Hata: {d['error']}")
    else:
        click.echo(f"periyot={d.get('period')} n={d.get('n')}")
    click.echo("--- Tahmin ---")
    f = forecast(df, col, horizon=horizon, method=method, period=period)
    if "error" in f:
        click.echo(f"Hata: {f['error']}")
        return
    if f.get("warning"):
        click.echo(f"UYARI: {f['warning']}")
    click.echo(f"yontem={f.get('method')} horizon={f.get('horizon')}")
    for i, point in enumerate(f["point"], start=1):
        lo, hi = f["ci_lower"][i - 1], f["ci_upper"][i - 1]
        click.echo(f"  +{i}: {point:.2f}  (%95 GA: {lo:.2f} - {hi:.2f})")


# ── model ──────────────────────────────────────────────────────────────────


@cli.command("model")
@click.argument("input", type=click.Path(exists=True))
@click.option("--target", required=True, help="Bagimli degisken")
@click.option("--features", required=True, help="Vergulle ayrilmis ozellikler")
@click.option("--kind", default="linear", type=click.Choice(["linear", "logistic"]))
def model_cmd(input: str, target: str, features: str, kind: str) -> None:
    """Regresyon/modelleme (lineer/lojistik)."""
    df = _read_input(input)
    feats = [x.strip() for x in features.split(",") if x.strip()]
    res = fit_regression(df, target, feats, kind=kind)
    if "error" in res:
        click.echo(f"Hata: {res['error']}")
        return
    click.echo(f"kind={res.get('kind')} n={res.get('n')}")
    metric = res.get("r2", res.get("pseudo_r2", res.get("accuracy")))
    click.echo(f"performans={metric}")
    click.echo("katsayilar:")
    for k, v in (res.get("coefficients") or {}).items():
        pv = (res.get("p_values") or {}).get(k)
        extra = f"  (p={pv:.4f})" if pv is not None else ""
        click.echo(f"  {k}: {v:.4f}{extra}")
    if res.get("vif_warning"):
        click.echo("VIF uyari (coklu-dogrusal-bagimlilik): "
                   + ", ".join(res["vif_warning"]))


# ── helpers ────────────────────────────────────────────────────────────────


def _resolve_input(input: str | None, url: str | None,
                   join: tuple[str, ...], on: str | None) -> str:
    """Return a path to the dataset for ``pipeline.run`` (handles url/join)."""
    if url:
        fmt = url.rsplit(".", 1)[-1].lower() if "." in url else "csv"
        if fmt not in ("csv", "json", "xlsx"):
            fmt = "csv"
        return _dump_temp(load_from_url(url, fmt=fmt))
    if join:
        paths = [input] + list(join) if input else list(join)
        mode = "join" if on else "concat"
        return _dump_temp(load_many(paths, mode=mode, on=on))
    if input is None:
        raise click.UsageError("INPUT veya --url gerekli.")
    return input


def _dump_temp(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _read_input(input: str) -> pd.DataFrame:
    if input.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(input)
    return pd.read_csv(input)


def _print_scores(result) -> None:
    q = result.eda.quality_score
    if q:
        click.echo(f"Veri Kalite Skoru: {q.get('score')}")
    r = result.eda.risk_score
    if r:
        click.echo(f"PII Yeniden-Tanimlama Risk Skoru: {r.get('score')}")


# Entry-point alias (pyproject: eda = "eda_toolkit.cli:main").
main = cli

if __name__ == "__main__":
    cli()
