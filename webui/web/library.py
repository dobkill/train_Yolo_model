"""SQLite catalog for local models, datasets, files, selections and test history."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid

IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def local_path(value):
    value = str(value).strip().strip('"')
    if not value or '://' in value:
        raise ValueError('请选择本机上的文件或文件夹。')
    return Path(value).expanduser().resolve()


def canonical(path):
    return os.path.normcase(str(path.resolve()))


def yaml_config(path):
    """Resolve only local sources; a moved YAML can use its adjacent data tree."""
    import yaml
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    except yaml.YAMLError as exc:
        raise ValueError('数据集 YAML 格式无效，请检查缩进和字段。') from exc
    if not isinstance(config, dict) or not config.get('names'):
        raise ValueError('数据集 YAML 必须包含 names 和 val 或 test。')
    raw_root = Path(str(config.get('path', '.')))
    roots = [raw_root if raw_root.is_absolute() else path.parent / raw_root,
             path.parent, path.parent / 'yolo/detection', path.parent / 'yolo/segmentation']
    sources = {}
    selected_root = None
    for split in ('test', 'val', 'train'):
        raw = config.get(split)
        if not raw:
            continue
        values = raw if isinstance(raw, list) else [raw]
        for root in roots:
            resolved = [local_path(root / str(value)) for value in values]
            if all(p.exists() for p in resolved):
                sources[split] = resolved
                selected_root = selected_root or root.resolve()
                break
    if not sources:
        raise ValueError('YAML 中的图片路径均已失效，请修正路径或登记图片文件夹。')
    normalized = {k: v for k, v in config.items() if k != 'download'}
    normalized['path'] = str(selected_root)
    for split in ('train', 'val', 'test'):
        normalized[split] = [str(p) for p in sources.get(split, [])] or None
    return normalized, sources


class Library:
    def __init__(self, db):
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as con:
            con.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                    task TEXT NOT NULL, format TEXT NOT NULL, valid INTEGER DEFAULT 1,
                    issue TEXT DEFAULT '', created_at TEXT NOT NULL, checked_at TEXT);
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                    valid INTEGER DEFAULT 1, issue TEXT DEFAULT '', image_count INTEGER DEFAULT 0,
                    video_count INTEGER DEFAULT 0, annotated INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL, checked_at TEXT);
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, path TEXT NOT NULL,
                    name TEXT NOT NULL, kind TEXT NOT NULL, size INTEGER, mtime REAL);
                CREATE INDEX IF NOT EXISTS file_dataset ON files(dataset_id, kind, name);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, model_id TEXT, dataset_id TEXT, status TEXT NOT NULL,
                    mode TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    report TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outcomes (
                    job_id TEXT, file_id TEXT, model_id TEXT, result_id TEXT, status TEXT,
                    created_at TEXT NOT NULL, PRIMARY KEY(job_id, file_id));
                CREATE INDEX IF NOT EXISTS outcome_file ON outcomes(file_id, model_id, created_at);
            ''')
        # Only the first run imports existing local weights. No dataset is required.
        if not self.get_setting('initialized'):
            train = self.db.parents[3] / 'train'
            candidates = list(train.glob('runs/*/weights/*.pt')) + list(train.glob('pretrained/*.pt'))
            for path in sorted(candidates, key=lambda p: (p.name != 'best.pt', str(p))):
                self.register_model({'path': str(path), 'name': f'{path.parent.parent.name} / {path.name}'
                                     if path.parent.name == 'weights' else path.stem,
                                     'task': 'segment' if '-seg' in path.stem else 'detect'})
            self.set_settings({'initialized': True})
        # A killed process must not leave phantom running tests.
        with self.connect() as con:
            for row in con.execute("SELECT * FROM jobs WHERE status IN ('running','queued','stopping')").fetchall():
                report = json.loads(row['report'])
                report.update(status='interrupted', error='上次服务已退出，测试中断。')
                con.execute('UPDATE jobs SET status=?,report=?,updated_at=? WHERE id=?',
                            ('interrupted', json.dumps(report, ensure_ascii=False), now(), row['id']))
        self.validate()

    @contextmanager
    def connect(self):
        with self.lock:
            con = sqlite3.connect(self.db, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                with con:
                    yield con
            finally:
                con.close()

    def get_setting(self, key):
        with self.connect() as con:
            row = con.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row['value']) if row else None

    def set_settings(self, values):
        with self.connect() as con:
            for key, value in values.items():
                con.execute('INSERT OR REPLACE INTO settings VALUES (?,?)',
                            (key, json.dumps(value, ensure_ascii=False)))

    def settings(self):
        with self.connect() as con:
            return {r['key']: json.loads(r['value']) for r in con.execute('SELECT * FROM settings')}

    def rows(self, table):
        if table not in {'models', 'datasets'}:
            raise ValueError('未知记录类型。')
        with self.connect() as con:
            return [dict(r) for r in con.execute(f'SELECT * FROM {table} ORDER BY created_at,id')]

    def get(self, table, identifier):
        if table not in {'models', 'datasets', 'files'}:
            raise ValueError('未知记录类型。')
        with self.connect() as con:
            row = con.execute(f'SELECT * FROM {table} WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('记录不存在，请重新选择。')
        return dict(row)

    def register_model(self, body):
        path = local_path(body.get('path', ''))
        task = body.get('task', 'detect')
        if task not in {'detect', 'segment'}:
            raise ValueError('请选择物体检测或图像分割。')
        if not path.is_file() or path.suffix.lower() not in {'.pt', '.onnx'}:
            raise ValueError('请选择存在的 YOLO .pt 或 .onnx 模型文件。')
        with self.connect() as con:
            previous = con.execute('SELECT id FROM models WHERE path=?', (canonical(path),)).fetchone()
            identifier = previous['id'] if previous else uuid.uuid4().hex
            con.execute('''INSERT INTO models VALUES (?,?,?,?,?,1,'',?,?)
                ON CONFLICT(path) DO UPDATE SET name=excluded.name,task=excluded.task,
                valid=1,issue='',checked_at=excluded.checked_at''',
                (identifier, canonical(path), str(body.get('name') or path.stem)[:150],
                 task, path.suffix[1:].upper(), now(), now()))
        return self.get('models', identifier)

    def scan(self, path):
        annotated = False
        if path.suffix.lower() in {'.yaml', '.yml'} and path.is_file():
            _, sources = yaml_config(path)
            # Preview the test split, or val if the dataset has no test split.
            roots = sources.get('test') or sources.get('val') or sources['train']
            annotated = True
        else:
            roots = [path]
            annotated = path.is_dir() and (path / 'labels').is_dir()
        found = {}
        for root in roots:
            if root.is_file() and root.suffix.lower() == '.txt':
                candidates = [local_path(root.parent / line.strip())
                              for line in root.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            elif root.is_file():
                candidates = [root]
            else:
                candidates = (Path(base) / name for base, _, names in os.walk(root) for name in names)
            for file in candidates:
                suffix = file.suffix.lower()
                if suffix not in IMAGES | VIDEOS or not file.is_file():
                    continue
                stat = file.stat()
                key = canonical(file)
                found[key] = (key, file.name, 'image' if suffix in IMAGES else 'video', stat.st_size, stat.st_mtime)
        if not found:
            raise ValueError('没有找到支持的图片或视频，或数据路径已失效。')
        return sorted(found.values(), key=lambda r: r[0]), annotated

    def register_dataset(self, body):
        path = local_path(body.get('path', ''))
        if not path.exists():
            raise ValueError('数据集路径不存在。')
        files, annotated = self.scan(path)
        with self.connect() as con:
            previous = con.execute('SELECT id FROM datasets WHERE path=?', (canonical(path),)).fetchone()
            identifier = previous['id'] if previous else uuid.uuid4().hex
            images = sum(f[2] == 'image' for f in files)
            con.execute('''INSERT INTO datasets VALUES (?,?,?,1,'',?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET name=excluded.name,valid=1,issue='',
                image_count=excluded.image_count,video_count=excluded.video_count,
                annotated=excluded.annotated,checked_at=excluded.checked_at''',
                (identifier, canonical(path), str(body.get('name') or path.stem)[:150], images,
                 len(files) - images, int(annotated), now(), now()))
            con.execute('DELETE FROM files WHERE dataset_id=?', (identifier,))
            con.executemany('INSERT INTO files VALUES (?,?,?,?,?,?,?)',
                            [(hashlib.sha256((identifier + f[0]).encode()).hexdigest()[:32], identifier, *f)
                             for f in files])
        return self.get('datasets', identifier)

    def validate(self):
        invalid = []
        for table in ('models', 'datasets'):
            for row in self.rows(table):
                issue = ''
                try:
                    path = Path(row['path'])
                    if table == 'models':
                        if not path.is_file():
                            raise ValueError('模型文件已移动或删除。')
                    else:
                        self.register_dataset(row)
                except (ValueError, OSError) as exc:
                    issue = str(exc)
                with self.connect() as con:
                    con.execute(f'UPDATE {table} SET valid=?,issue=?,checked_at=? WHERE id=?',
                                (int(not issue), issue, now(), row['id']))
                if issue:
                    invalid.append(dict(row, issue=issue, category=table))
        self.invalid = invalid
        return invalid

    def delete(self, table, identifier):
        if table not in {'models', 'datasets'}:
            raise ValueError('未知记录类型。')
        with self.connect() as con:
            con.execute(f'DELETE FROM {table} WHERE id=?', (identifier,))
            if table == 'datasets':
                con.execute('DELETE FROM files WHERE dataset_id=?', (identifier,))
        # Source files and result history are retained.

    def files(self, dataset_id, kind='image', search='', offset=0, limit=80, model_id=''):
        self.get('datasets', dataset_id)
        with self.connect() as con:
            clauses, args = 'dataset_id=? AND kind=? AND instr(lower(name),lower(?))>0', [dataset_id, kind, search]
            total = con.execute(f'SELECT COUNT(*) FROM files WHERE {clauses}', args).fetchone()[0]
            rows = [dict(r) for r in con.execute(f'SELECT * FROM files WHERE {clauses} ORDER BY path LIMIT ? OFFSET ?',
                                                [*args, limit, offset])]
            for row in rows:
                outcome = con.execute('''SELECT result_id,status FROM outcomes WHERE file_id=? AND model_id=?
                    ORDER BY created_at DESC,rowid DESC LIMIT 1''', (row['id'], model_id)).fetchone()
                row['outcome'] = dict(outcome) if outcome else None
                row['url'] = '/api/media/' + row['id']
                row['thumbnail'] = '/api/thumb/' + row['id'] + '?v=' + str(row['mtime'])
        return {'items': rows, 'total': total, 'offset': offset, 'limit': limit}

    def dataset_files(self, dataset_id, kind):
        with self.connect() as con:
            return [dict(r) for r in con.execute('SELECT * FROM files WHERE dataset_id=? AND kind=? ORDER BY path',
                                                (dataset_id, kind))]

    def save_job(self, job):
        with self.connect() as con:
            con.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                        (job['id'], job['model_id'], job['dataset_id'], job['status'], job['mode'],
                         job['created_at'], now(), json.dumps(job, ensure_ascii=False)))

    def outcome(self, job, file_id, result_id, status):
        with self.connect() as con:
            con.execute('INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?)',
                        (job['id'], file_id, job['model_id'], result_id, status, now()))

    def history(self):
        with self.connect() as con:
            return [json.loads(r['report']) for r in con.execute('SELECT report FROM jobs ORDER BY created_at DESC,rowid DESC LIMIT 100')]

    def job(self, identifier):
        with self.connect() as con:
            row = con.execute('SELECT report FROM jobs WHERE id=?', (identifier,)).fetchone()
        if not row:
            raise ValueError('测试记录不存在。')
        return json.loads(row['report'])
