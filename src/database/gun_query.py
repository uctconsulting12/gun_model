
import os
import logging
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import json
import logging

load_dotenv()
logger = logging.getLogger("detection")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

try:
    pool = SimpleConnectionPool(
        minconn=1,        
        maxconn=15,       
        **DB_CONFIG
    )
    logger.info("✅ PostgreSQL Connection Pool Created")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")
    raise




logger = logging.getLogger("detection")


def insert_data(d, s3_url):
    """
    Insert gun detection data into gun_detections table
    STRICTLY follows the latest response structure
    """

    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO gun_detections (
                cam_id,
                org_id,
                user_id,
                persons,
                guns,
                gun_holders,
                s3_url,
                status,
                timestamp
            )
            VALUES (
                %s, %s, %s,
                %s::jsonb,
                %s::jsonb,
                %s::jsonb,
                %s, %s,%s
            )
            RETURNING id;
        """

        cursor.execute(
            insert_query,
            (
                d["cam_id"],                       # int
                d["org_id"],                       # int
                d["user_id"],                      # int
                json.dumps(d["persons_present"]),  # jsonb
                json.dumps(d["guns"]),             # jsonb
                json.dumps(d["gun_holders"]),      # jsonb
                s3_url,                            # text
                d["status"] ,
                d["time_stamp"]
            )
        )

        inserted_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"✅ Gun detection inserted | cam_id={d['cam_id']} | id={inserted_id}"
        )
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ DB insert failed: {e}")
        return False

    finally:
        if conn:
            pool.putconn(conn)
