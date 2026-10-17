"""Command-line interface for the local corpus pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .service import CorpusPipeline


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="Локальный импорт CSV без API")
    importer.add_argument("path", type=Path)
    importer.add_argument("--title", default="")
    importer.add_argument("--config", type=Path)
    commands.add_parser("list")
    for name in ("show", "export", "retry-errors", "recover-interrupted"):
        commands.add_parser(name).add_argument("id")
    configure = commands.add_parser("configure")
    configure.add_argument("id")
    configure.add_argument("config", type=Path)
    prepare = commands.add_parser("prepare", help="Подготовить задания и оценку стоимости без API")
    prepare.add_argument("id")
    prepare.add_argument("--stage", choices=["relevance", "extract", "describe", "score"], default="relevance")
    prepare.add_argument("--limit", type=int, default=100)
    run = commands.add_parser("run", help="Явный платный пробный запуск")
    run.add_argument("id")
    run.add_argument("--execute", action="store_true", help="Разрешить обращения к API")
    run.add_argument("--budget-usd", type=float, required=True)
    run.add_argument("--max-jobs", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        # Load credentials without copying or displaying them in any report.
        try:
            from dotenv import load_dotenv
            load_dotenv(args.root / ".env", override=False)
        except ImportError:
            pass
        pipeline = CorpusPipeline(args.root)
        read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
        if args.command == "import":
            result = pipeline.import_csv(args.path, args.title, config=read(args.config) if args.config else None)
        elif args.command == "list":
            result = pipeline.list_corpora()
        elif args.command == "show":
            result = pipeline.get_corpus(args.id)
        elif args.command == "configure":
            result = pipeline.configure(args.id, read(args.config))
        elif args.command == "prepare":
            result = pipeline.prepare(args.id, args.stage, args.limit)
        elif args.command == "run":
            if not args.execute:
                parser.error("Для платного запуска требуется --execute; prepare не вызывает API")
            result = pipeline.run(args.id, max_jobs=args.max_jobs, budget_usd=args.budget_usd)
        elif args.command == "retry-errors":
            result = pipeline.retry_errors(args.id)
        elif args.command == "recover-interrupted":
            result = pipeline.recover_interrupted(args.id)
        else:
            result = pipeline.export(args.id)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
