"""CLI:  python -m riskmart etl | serve | ask "question" """
import argparse
import json
import sqlite3

from . import config


def main():
    ap = argparse.ArgumentParser(prog="riskmart")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("etl", help="fetch public data, load the mart, run data-quality checks")
    e.add_argument("--db", default=str(config.DEFAULT_DB))
    e.add_argument("--seed", type=int, default=42, help="seed for the synthetic portfolio")
    s = sub.add_parser("serve", help="start the API + dashboard")
    s.add_argument("--db", default=str(config.DEFAULT_DB))
    s.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("ask", help="ask a question from the command line")
    a.add_argument("question")
    a.add_argument("--db", default=str(config.DEFAULT_DB))
    args = ap.parse_args()

    if args.cmd == "etl":
        from .etl import run_etl
        print(json.dumps(run_etl(args.db, seed=args.seed), indent=2))
    elif args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port)
    else:
        from .qa import answer_question
        print(json.dumps(answer_question(sqlite3.connect(args.db), args.question).to_dict(), indent=2))


if __name__ == "__main__":
    main()
