DROP TABLE IF EXISTS lineorder;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS dwdate;

CREATE TABLE part (
    p_partkey      INTEGER PRIMARY KEY,
    p_name         VARCHAR(64),
    p_mfgr         VARCHAR(16),
    p_category     VARCHAR(16),
    p_brand1       VARCHAR(16)
);

CREATE TABLE supplier (
    s_suppkey      INTEGER PRIMARY KEY,
    s_name         VARCHAR(64),
    s_address      VARCHAR(64),
    s_city         VARCHAR(16),
    s_nation       VARCHAR(16),
    s_region       VARCHAR(16)
);

CREATE TABLE customer (
    c_custkey      INTEGER PRIMARY KEY,
    c_name         VARCHAR(64),
    c_address      VARCHAR(64),
    c_city         VARCHAR(16),
    c_nation       VARCHAR(16),
    c_region       VARCHAR(16)
);

CREATE TABLE dwdate (
    d_datekey      INTEGER PRIMARY KEY,
    d_date         VARCHAR(20),
    d_year         INTEGER
);

CREATE TABLE lineorder (
    lo_orderkey      BIGINT,
    lo_linenumber    INTEGER,

    lo_custkey       INTEGER NOT NULL,
    lo_partkey       INTEGER NOT NULL,
    lo_suppkey       INTEGER NOT NULL,
    lo_orderdate     INTEGER NOT NULL,

    lo_revenue       BIGINT NOT NULL,
    lo_supplycost    BIGINT NOT NULL,

    PRIMARY KEY (lo_orderkey, lo_linenumber)
);

CREATE INDEX idx_part_category_key_brand
    ON part (p_category, p_partkey, p_brand1);

CREATE INDEX idx_supplier_nation_key_city
    ON supplier (s_nation, s_suppkey, s_city);

CREATE INDEX idx_customer_region_key
    ON customer (c_region, c_custkey);

CREATE INDEX idx_dwdate_year_key
    ON dwdate (d_year, d_datekey);

CREATE INDEX idx_lineorder_partkey
    ON lineorder (lo_partkey);

CREATE INDEX idx_lineorder_suppkey
    ON lineorder (lo_suppkey);

CREATE INDEX idx_lineorder_custkey
    ON lineorder (lo_custkey);

CREATE INDEX idx_lineorder_orderdate
    ON lineorder (lo_orderdate);

CREATE INDEX idx_part_category
    ON part (p_category);

CREATE INDEX idx_supplier_nation
    ON supplier (s_nation);

CREATE INDEX idx_customer_region
    ON customer (c_region);

CREATE INDEX idx_dwdate_year
    ON dwdate (d_year);


INSERT INTO part
SELECT
    i,
    'Part-' || i,
    'MFGR#' || ((i % 5) + 1),
    CASE
        WHEN i % 10 < 2 THEN 'MFGR#14'
        ELSE 'MFGR#' || ((i % 20) + 10)
    END,
    'Brand#' || (i % 100)
FROM generate_series(1, 20000) AS g(i);

INSERT INTO supplier
SELECT
    i,
    'Supplier-' || i,
    'Address-' || i,
    'City-' || (i % 100),
    CASE
        WHEN i % 10 < 2 THEN 'UNITED STATES'
        ELSE 'CHINA'
    END,
    CASE
        WHEN i % 5 = 0 THEN 'AMERICA'
        ELSE 'ASIA'
    END
FROM generate_series(1, 2000) AS g(i);

INSERT INTO customer
SELECT
    i,
    'Customer-' || i,
    'Address-' || i,
    'City-' || (i % 100),
    CASE
        WHEN i % 4 = 0 THEN 'UNITED STATES'
        ELSE 'CHINA'
    END,
    CASE
        WHEN i % 5 < 2 THEN 'AMERICA'
        ELSE 'ASIA'
    END
FROM generate_series(1, 30000) AS g(i);

INSERT INTO dwdate
SELECT
    i,
    'Date-' || i,
    1992 + (i % 7)
FROM generate_series(1, 2556) AS g(i);

INSERT INTO lineorder
SELECT
    i,
    1,
    1 + (i % 30000),
    1 + (i % 20000),
    1 + (i % 2000),
    1 + (i % 2556),
    1000 + (i % 10000),
    500 + (i % 5000)
FROM generate_series(1, 200000) AS g(i);

ANALYZE part;
ANALYZE supplier;
ANALYZE customer;
ANALYZE dwdate;
ANALYZE lineorder;

EXPLAIN
SELECT
    d.d_year,
    s.s_city,
    p.p_brand1,
    SUM(lo.lo_revenue - lo.lo_supplycost) AS profit
FROM part AS p
JOIN lineorder AS lo
    ON lo.lo_partkey = p.p_partkey
JOIN supplier AS s
    ON lo.lo_suppkey = s.s_suppkey
JOIN customer AS c
    ON lo.lo_custkey = c.c_custkey
JOIN dwdate AS d
    ON lo.lo_orderdate = d.d_datekey
WHERE
    p.p_category = 'MFGR#14'
    AND s.s_nation = 'UNITED STATES'
    AND c.c_region = 'AMERICA'
    AND d.d_year BETWEEN 1997 AND 1998
GROUP BY
    d.d_year,
    s.s_city,
    p.p_brand1
ORDER BY
    d.d_year,
    s.s_city,
    p.p_brand1;
