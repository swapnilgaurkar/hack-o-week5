"""SQLite persistence for expenses and the monthly budget."""
import sqlite3
from contextlib import closing
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "moneta.db"

CATEGORIES = ["Food and dining", "Transport", "Shopping", "Bills and utilities",
              "Education", "Health", "Entertainment", "Groceries", "Other"]


class Database:
    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL,
                amount REAL NOT NULL, category TEXT NOT NULL, description TEXT NOT NULL DEFAULT '')""")
            c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def add_expense(self, e):
        with closing(self._conn()) as c, c:
            cur = c.execute("INSERT INTO expenses(date,amount,category,description) VALUES(?,?,?,?)",
                            (e["date"], e["amount"], e["category"], e["description"]))
            return self._get(c, cur.lastrowid)

    @staticmethod
    def _get(c, eid):
        r = c.execute("SELECT * FROM expenses WHERE id=?", (eid,)).fetchone()
        return dict(r) if r else None

    def get_expense(self, eid):
        with closing(self._conn()) as c:
            return self._get(c, eid)

    def update_expense(self, eid, e):
        with closing(self._conn()) as c, c:
            if not self._get(c, eid):
                return None
            c.execute("UPDATE expenses SET date=?,amount=?,category=?,description=? WHERE id=?",
                      (e["date"], e["amount"], e["category"], e["description"], eid))
            return self._get(c, eid)

    def delete_expense(self, eid):
        with closing(self._conn()) as c, c:
            return c.execute("DELETE FROM expenses WHERE id=?", (eid,)).rowcount > 0

    def list_expenses(self, q="", category="", sort="date", order="desc"):
        sort_col = "amount" if sort == "amount" else "date"
        direction = "ASC" if order == "asc" else "DESC"
        sql, args = "SELECT * FROM expenses WHERE 1=1", []
        if q:
            esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND (description LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\')"
            args += [f"%{esc}%"] * 2
        if category:
            sql += " AND category=?"
            args.append(category)
        sql += f" ORDER BY {sort_col} {direction}, id DESC"   # column/direction are whitelisted
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(sql, args)]

    def get_budget(self):
        with closing(self._conn()) as c:
            r = c.execute("SELECT value FROM settings WHERE key='monthly_budget'").fetchone()
            return float(r["value"]) if r else 0.0

    def set_budget(self, value):
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO settings(key,value) VALUES('monthly_budget',?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(value),))
