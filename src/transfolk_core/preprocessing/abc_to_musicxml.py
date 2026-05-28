from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from music21 import converter


def _sanitize_filename(value: str, fallback: str = "untitled") -> str:
    """
    Limpia un texto para usarlo como nombre de archivo o carpeta.

    Solo conserva letras ASCII, números y guion bajo.
    Los espacios, guiones, signos y caracteres raros se convierten en guion bajo.
    Los acentos se normalizan: á -> a, ñ -> n, ü -> u.
    """
    if value is None:
        value = ""

    value = str(value).strip()

    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")

    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    if not value:
        value = fallback

    return value


def _split_abc_tunes(abc_text: str) -> List[str]:
    """
    Divide el contenido completo de un archivo ABC en tunes individuales.
    Cada tune suele empezar por una línea 'X:'.
    """
    lines = abc_text.splitlines()
    tunes: List[List[str]] = []
    current_tune: List[str] = []

    for line in lines:
        if line.startswith("X:"):
            if current_tune:
                tunes.append(current_tune)
            current_tune = [line]
        else:
            if current_tune:
                current_tune.append(line)

    if current_tune:
        tunes.append(current_tune)

    return [
        "\n".join(tune).strip() + "\n"
        for tune in tunes
        if any(line.strip() for line in tune)
    ]


def _extract_title(abc_tune: str) -> str:
    """
    Extrae el título de la tune a partir de la primera línea 'T:'.
    """
    for line in abc_tune.splitlines():
        if line.startswith("T:"):
            return line[2:].strip()

    return "untitled"


def _convert_abc_tune_to_musicxml(abc_tune: str, output_path: Path) -> Tuple[bool, str]:
    """
    Convierte una tune ABC individual a MusicXML.
    """
    try:
        score = converter.parseData(abc_tune, format="abc")
        score.write("musicxml", fp=str(output_path))
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def convert_abc_folder_to_musicxml(
    input_dir: str | Path,
    output_dir: str | Path,
    respect_subfolder: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Convierte todos los archivos .abc de una carpeta a MusicXML.

    Cada archivo .abc puede contener múltiples tunes. La función separa
    cada tune interna y la exporta como un fichero .musicxml independiente.

    Parámetros
    ----------
    input_dir : str | Path
        Ruta a la carpeta que contiene archivos .abc.

    output_dir : str | Path
        Ruta a la carpeta donde se guardarán los .musicxml.

    respect_subfolder : bool, default=False
        Si es True, guarda las obras convertidas en una subcarpeta por cada
        archivo ABC de origen.

        Si es False, guarda todas las obras convertidas directamente en la
        carpeta raíz de salida.

    show_progress : bool, default=True
        Si es True, muestra progreso en consola.

    Retorna
    -------
    dict
        Resumen del proceso con contadores y errores.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"La ruta de entrada no es una carpeta: {input_dir}")

    abc_files = sorted(input_dir.glob("*.abc"))

    if not abc_files:
        raise FileNotFoundError(f"No se encontraron archivos .abc en: {input_dir}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "respect_subfolder": respect_subfolder,
        "abc_files_processed": 0,
        "tunes_detected": 0,
        "converted_ok": 0,
        "failed": 0,
        "errors": [],
    }

    if show_progress:
        print("\n" + "=" * 70)
        print("CONVERSIÓN ABC -> MUSICXML")
        print("=" * 70)
        print(f"Entrada             : {input_dir}")
        print(f"Salida              : {output_dir}")
        print(f"Respetar subcarpetas: {respect_subfolder}")
        print(f"Archivos ABC        : {len(abc_files)}")
        print("=" * 70 + "\n")

    used_names_global = {}

    for file_idx, abc_file in enumerate(abc_files, start=1):
        summary["abc_files_processed"] += 1

        if show_progress:
            tqdm.write("")
            tqdm.write("=" * 70)
            tqdm.write(f"Archivo ABC {file_idx}/{len(abc_files)}: {abc_file.name}")
            tqdm.write("=" * 70)

        try:
            abc_text = abc_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append({
                "file": abc_file.name,
                "tune_index": None,
                "title": None,
                "error": f"No se pudo leer el archivo ABC: {exc}",
            })

            if show_progress:
                tqdm.write(f"[ERROR] No se pudo leer {abc_file.name}: {exc}")

            continue

        tunes = _split_abc_tunes(abc_text)

        if not tunes:
            summary["errors"].append({
                "file": abc_file.name,
                "tune_index": None,
                "title": None,
                "error": "No se detectaron tunes en este archivo ABC",
            })

            if show_progress:
                tqdm.write(f"[WARN] No se detectaron tunes en {abc_file.name}")

            continue

        safe_abc_stem = _sanitize_filename(
            abc_file.stem,
            fallback=f"abc_file_{file_idx:04d}",
        )

        if respect_subfolder:
            target_dir = output_dir / safe_abc_stem
            target_dir.mkdir(parents=True, exist_ok=True)
            used_names = {}
        else:
            target_dir = output_dir
            used_names = used_names_global

        if show_progress:
            tqdm.write(f"Tunes detectadas: {len(tunes)}")

        tune_iterator = tqdm(
            tunes,
            desc="Desempaquetando",
            unit="tune",
            total=len(tunes),
            dynamic_ncols=False,
            ncols=100,
            leave=False,
            disable=not show_progress,
            mininterval=0.2,
            position=0,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

        for i, tune in enumerate(tune_iterator, start=1):
            summary["tunes_detected"] += 1

            title = _extract_title(tune)

            if show_progress:
                visible_title = title.replace("\n", " ").replace("\r", " ")
                tune_iterator.set_postfix_str(
                    f"{i:04d} | {visible_title[:40]}",
                    refresh=True,
                )

            safe_title = _sanitize_filename(title, fallback=f"tune_{i:04d}")

            if respect_subfolder:
                base_filename = f"{i:04d}_{safe_title}"
            else:
                base_filename = f"{safe_abc_stem}_{i:04d}_{safe_title}"

            count = used_names.get(base_filename, 0) + 1
            used_names[base_filename] = count

            if count > 1:
                filename = f"{base_filename}_{count}.musicxml"
            else:
                filename = f"{base_filename}.musicxml"

            output_path = target_dir / filename

            ok, msg = _convert_abc_tune_to_musicxml(tune, output_path)

            if ok:
                summary["converted_ok"] += 1
            else:
                summary["failed"] += 1

                error_data = {
                    "file": abc_file.name,
                    "tune_index": i,
                    "title": title,
                    "output_path": str(output_path),
                    "error": msg,
                }

                summary["errors"].append(error_data)

                if show_progress:
                    tqdm.write(
                        f"[ERROR] {abc_file.name} | "
                        f"Tune {i} | "
                        f"{title} | "
                        f"{msg}"
                    )

        if show_progress:
            tqdm.write(
                f"Finalizado {abc_file.name} | "
                f"Tunes: {len(tunes)} | "
                f"OK acumuladas: {summary['converted_ok']} | "
                f"Fallidas acumuladas: {summary['failed']}"
            )

    if show_progress:
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print(f"Carpeta entrada        : {summary['input_dir']}")
        print(f"Carpeta salida         : {summary['output_dir']}")
        print(f"Respetar subcarpetas   : {summary['respect_subfolder']}")
        print(f"Archivos ABC procesados: {summary['abc_files_processed']}")
        print(f"Tunes detectadas       : {summary['tunes_detected']}")
        print(f"Convertidas OK         : {summary['converted_ok']}")
        print(f"Fallidas               : {summary['failed']}")

        if summary["errors"]:
            print("\nSe han producido errores. Primeros 10 errores:")

            for err in summary["errors"][:10]:
                print(
                    f"  - Archivo: {err['file']} | "
                    f"Tune: {err['tune_index']} | "
                    f"Título: {err['title']} | "
                    f"Error: {err['error']}"
                )
        else:
            print("\n[OK] No se han producido errores")

        print("=" * 70 + "\n")

    return summary