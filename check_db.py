import psycopg2

try:
    conn = psycopg2.connect(host='localhost', port=5432, dbname='lpr_db', user='admin', password='admin123')
    cur = conn.cursor()

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"[OK] PostgreSQL connected. Tables: {tables}")

    if 'phat_nguoi' in tables:
        cur.execute("SELECT COUNT(*) FROM phat_nguoi")
        print(f"  - phat_nguoi: {cur.fetchone()[0]} rows")
    if 'lich_su_camera' in tables:
        cur.execute("SELECT COUNT(*) FROM lich_su_camera")
        print(f"  - lich_su_camera: {cur.fetchone()[0]} rows")
        # Check columns
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='lich_su_camera'")
        cols = [r[0] for r in cur.fetchall()]
        print(f"  - lich_su_camera columns: {cols}")

    conn.close()
except Exception as e:
    print(f"[FAIL] PostgreSQL: {e}")

# Test MinIO
try:
    from minio import Minio
    mc = Minio("localhost:9000", access_key="admin", secret_key="admin123", secure=False)
    buckets = [b.name for b in mc.list_buckets()]
    print(f"[OK] MinIO connected. Buckets: {buckets}")
except Exception as e:
    print(f"[FAIL] MinIO: {e}")

# Test Qdrant
try:
    from qdrant_client import QdrantClient
    qc = QdrantClient(host="localhost", port=6333)
    cols = [c.name for c in qc.get_collections().collections]
    print(f"[OK] Qdrant connected. Collections: {cols}")
except Exception as e:
    print(f"[FAIL] Qdrant: {e}")

# Test RabbitMQ
try:
    import pika
    creds = pika.PlainCredentials('admin', 'admin123')
    conn = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=creds))
    conn.close()
    print("[OK] RabbitMQ connected.")
except Exception as e:
    print(f"[FAIL] RabbitMQ: {e}")
