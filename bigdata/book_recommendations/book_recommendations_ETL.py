import argparse
import os

from pyspark.sql import DataFrame, SparkSession, Window, functions as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Построение витрин данных для рекомендаций книг"
    )
    parser.add_argument("--start_date", required=True, help="Начальная дата периода (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="Конечная дата периода (YYYY-MM-DD)")
    parser.add_argument("--geo_id", required=True, help="Идентификатор геолокации (usage_geo_id)")
    parser.add_argument(
        "--columns", required=True,
        help="Колонки для распределения через запятую, например: usage_platform_ru,genre"
    )
    parser.add_argument("--audition_path", required=True, help="Путь к audition.parquet")
    parser.add_argument("--content_path", required=True, help="Путь к content.parquet")
    parser.add_argument("--output_path", required=True, help="Путь для сохранения витрин (папка results)")

    return parser.parse_args()


def get_spark_session() -> SparkSession:
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")

    spark = (
        SparkSession.builder
        .appName("book_recommendations_ETL")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.executor.instances", "2")
        .config("spark.driver.cores", "4")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.network.timeout", "300")
        .config("spark.hadoop.fs.s3a.endpoint", "https://storage.yandexcloud.net")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    return spark


def load_data(spark: SparkSession, audition_path: str, content_path: str):
    audition_df = spark.read.parquet(audition_path).cache()
    content_df = spark.read.parquet(content_path).cache()

    print("=== audition ===")
    audition_df.printSchema()
    audition_df.show(5, truncate=False)

    print("=== content ===")
    content_df.printSchema()
    content_df.show(5, truncate=False)

    audition_total = audition_df.count()
    content_total = content_df.count()

    print(f"число строк audition: {audition_total}, кол-во колонок: {len(audition_df.columns)}")
    print(f"число строк content: {content_total}, кол-во колонок: {len(content_df.columns)}")

    return audition_df, content_df, audition_total, content_total


def parse_dates(audition_df: DataFrame) -> DataFrame:
    return audition_df.withColumn(
        "msk_business_dt",
        F.to_timestamp(F.col("msk_business_dt_str"), "yyyy-MM-dd")
    )


def analyze_audition_quality(audition_df: DataFrame, total: int) -> None:
    null_exprs = [
        F.sum(F.col(c).isNull().cast("int")).alias(f"null__{c}")
        for c in audition_df.columns
    ]
    anomaly_exprs = [
        F.sum((F.col("hours") < 0).cast("int")).alias("neg_hours"),
        F.sum((F.col("hours_sessions_long") < 0).cast("int")).alias("neg_hours_long"),
        F.sum((F.col("hours") > 24).cast("int")).alias("over_24_hours"),
        F.sum((F.col("hours_sessions_long") > 24).cast("int")).alias("over_24_long"),
        F.sum((F.col("hours") == 0).cast("int")).alias("zero_hours"),
        F.sum((F.col("hours_sessions_long") > F.col("hours")).cast("int")).alias("inconsistent"),
        F.sum(
            (F.col("msk_business_dt").isNull() & F.col("msk_business_dt_str").isNotNull()).cast("int")
        ).alias("date_parse_errors"),
        F.min("msk_business_dt").alias("min_date"),
        F.max("msk_business_dt").alias("max_date"),
        F.sum(
            ((F.col("adult_content_flg") == True) & (F.col("kids_content_flg") == True)).cast("int")
        ).alias("flag_conflict"),
    ]

    row = audition_df.agg(*null_exprs, *anomaly_exprs).collect()[0]

    nulls_found = {
        c: row[f"null__{c}"]
        for c in audition_df.columns
        if row[f"null__{c}"] > 0
    }
    print(f"=== audition: null-проверка (total rows = {total}) ===")
    if nulls_found:
        for c, cnt in nulls_found.items():
            print(f"{c} - nulls: {cnt} ({100 * cnt / total:.2f}%)")
    else:
        print("Пропусков не найдено.")

    print("=== audition: аномалии ===")
    print(f"Отрицательное время: hours = {row['neg_hours']}, hours_sessions_long = {row['neg_hours_long']}")
    print(f"Сессии > 24 часов: hours = {row['over_24_hours']}, hours_sessions_long = {row['over_24_long']}")
    print(f"Нулевые сессии (hours == 0): {row['zero_hours']}")
    print(f"Логические ошибки (hours_sessions_long > hours): {row['inconsistent']}")
    print(f"Ошибки парсинга дат (msk_business_dt_str): {row['date_parse_errors']}")
    print(f"Временной диапазон данных: {row['min_date']} - {row['max_date']}")
    print(f"Конфликт флагов (adult & kids == True): {row['flag_conflict']}")


def analyze_content_quality(content_df: DataFrame, total: int) -> None:
    null_exprs = [
        F.sum(F.col(c).isNull().cast("int")).alias(f"null__{c}")
        for c in content_df.columns
    ]
    anomaly_exprs = [
        F.sum((F.col("main_content_duration_hours") < 0).cast("int")).alias("neg_duration"),
        F.sum((F.col("main_content_duration_hours") == 0).cast("int")).alias("zero_duration"),
        F.sum((F.col("main_content_duration_hours") > 100).cast("int")).alias("extreme_duration"),
        F.sum(F.col("published_topic_title_list").isNull().cast("int")).alias("null_genres"),
    ]

    row = content_df.agg(*null_exprs, *anomaly_exprs).collect()[0]

    nulls_found = {
        c: row[f"null__{c}"]
        for c in content_df.columns
        if row[f"null__{c}"] > 0
    }
    print(f"=== content: null-проверка (total rows = {total}) ===")
    if nulls_found:
        for c, cnt in nulls_found.items():
            print(f"{c} - nulls: {cnt} ({100 * cnt / total:.2f}%)")
    else:
        print("Пропусков не найдено.")

    print("=== content: аномалии ===")
    print(f"Отрицательная длительность (hours < 0): {row['neg_duration']}")
    print(f"Нулевая длительность (hours == 0): {row['zero_duration']}")
    print(f"Экстремальная длительность (hours > 100): {row['extreme_duration']}")
    print(f"Пропуски в жанрах (published_topic_title_list): {row['null_genres']}")


def clean_audition(audition_df: DataFrame) -> DataFrame:
    return (
        audition_df
        .fillna({"app_version": "unknown"})
        .filter((F.col("hours") <= 24) & (F.col("hours_sessions_long") <= 24))
    )


def clean_content(content_df: DataFrame) -> DataFrame:
    return (
        content_df
        .filter(F.col("main_content_duration_hours").isNotNull())
        .fillna({"published_topic_title_list": "unknown"})
    )


def join_audition_content(audition_df: DataFrame, content_df: DataFrame) -> DataFrame:
    joined = audition_df.join(F.broadcast(content_df), on="main_content_id", how="left")

    rows_before = audition_df.count()
    rows_after = joined.count()
    print(f"=== Проверка join на дубли ===")
    print(f"Строк до join: {rows_before}, строк после join: {rows_after}, "
          f"расхождение: {rows_after - rows_before}")
    if rows_after != rows_before:
        print("join размножил строки")

    return joined


def filter_by_period_and_geo(df: DataFrame, start_date: str, end_date: str, geo_id: str) -> DataFrame:
    return df.filter(
        F.col("msk_business_dt").between(start_date, end_date) &
        (F.col("usage_geo_id").cast("string") == str(geo_id))
    )


def split_genres(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "genre_list",
        F.split(F.col("published_topic_title_list"), r",\s*")
    )


def analyze_platform_distribution(df: DataFrame, columns: list):
    for column in columns:
        source_df = df
        if column == "genre":
            source_df = df.withColumn("genre", F.explode(F.col("genre_list")))

        if column not in source_df.columns:
            print(f"Колонка '{column}' не найдена, пропущена.")
            continue

        print(f"=== Распределение уникальных пользователей по колонке: {column} ===")
        (
            source_df
            .groupBy(column)
            .agg(F.countDistinct("puid").alias("unique_users"))
            .orderBy(F.col("unique_users").desc())
        ).show(10, truncate=False)


def analyze_content_and_genre_popularity(df: DataFrame):
    print("=== Популярность типов контента ===")
    (
        df.groupBy("main_content_type")
        .agg(
            F.count("*").alias("sessions_count"),
            F.countDistinct("puid").alias("unique_users"),
            F.sum("hours").alias("total_hours")
        )
        .orderBy(F.col("sessions_count").desc())
    ).show(10, truncate=False)

    print("=== Популярность жанров ===")
    (
        df.withColumn("genre", F.explode(F.col("genre_list")))
        .groupBy("genre")
        .agg(
            F.count("*").alias("sessions_count"),
            F.countDistinct("puid").alias("unique_users"),
            F.sum("hours").alias("total_hours")
        )
        .orderBy(F.col("sessions_count").desc())
    ).show(10, truncate=False)


def analyze_duration_by_dow_geo(df: DataFrame):
    print("=== Длительность сессий по дню недели и геолокации ===")
    (
        df.withColumn("day_of_week", F.date_format(F.col("msk_business_dt"), "EEEE"))
        .groupBy("day_of_week", "usage_geo_id")
        .agg(
            F.avg("hours").alias("avg_hours"),
            F.sum("hours").alias("total_hours"),
            F.count("*").alias("sessions_count")
        )
        .orderBy(F.col("usage_geo_id"), F.col("avg_hours").desc())
    ).show(10, truncate=False)


def build_user_content_mart(filtered_df: DataFrame) -> DataFrame:
    mart = (
        filtered_df
        .groupBy("puid", "main_content_id")
        .agg(
            F.sum("hours").alias("total_hours"),
            F.count("*").alias("sessions_count"),
            F.max("main_content_duration_hours").alias("main_content_duration_hours"),
            F.max("msk_business_dt").alias("last_listening_date"),
            F.min("msk_business_dt").alias("first_listening_date"),
        )
    )

    mart = (
        mart
        .withColumn(
            "days_spent",
            F.datediff(F.col("last_listening_date"), F.col("first_listening_date")) + 1
        )
        .withColumn(
            "finished_percent",
            F.round(F.least(
                F.lit(100.0),
                F.col("total_hours") / F.col("main_content_duration_hours") * 100
            ), 2)
        )
    )

    mart = mart.withColumn(
        "is_completed",
        F.when(F.col("finished_percent") >= 100, 1).otherwise(0)
    ).withColumn(
        "avg_session_hours",
        F.round(F.col("total_hours") / F.col("sessions_count"), 4)
    )

    return mart.select(
        "puid", "main_content_id", "total_hours", "sessions_count",
        "finished_percent", "is_completed", "last_listening_date",
        "days_spent", "avg_session_hours",
    )


def build_user_genre_mart(filtered_df: DataFrame) -> DataFrame:
    df = filtered_df.select(
        "puid", "genre_list", "hours", "main_content_id",
        "main_content_duration_hours", "adult_content_flg", "kids_content_flg"
    )

    per_content = (
        df.withColumn("genre", F.explode(F.col("genre_list")))
        .groupBy("puid", "genre", "main_content_id")
        .agg(
            F.sum("hours").alias("content_total_hours"),
            F.count("*").alias("content_sessions_count"),
            F.max("main_content_duration_hours").alias("max_dur"),
            F.max("adult_content_flg").alias("is_adult"),
            F.max("kids_content_flg").alias("is_kids")
        )
    )

    mart = (
        per_content
        .groupBy("puid", "genre")
        .agg(
            F.sum("content_total_hours").alias("total_hours"),
            F.sum("content_sessions_count").alias("sessions_count"),
            F.sum(F.when(F.col("content_total_hours") >= F.col("max_dur"), 1).otherwise(0)).alias("completed_count"),
            F.sum(F.col("is_adult").cast("int")).alias("adult_count"),
            F.sum(F.col("is_kids").cast("int")).alias("kids_count")
        )
    )

    window_spec = Window.partitionBy("puid").orderBy(F.col("total_hours").desc())
    return mart.withColumn("genre_rank", F.rank().over(window_spec))


def build_user_mart_from_marts(
    filtered_df: DataFrame,
    user_content_mart: DataFrame,
    user_genre_mart: DataFrame,
) -> DataFrame:
    completion_and_content = (
        user_content_mart
        .groupBy("puid")
        .agg(F.round(F.avg("finished_percent"), 2).alias("completion_rate"))
    )

    genre_summary = (
        user_genre_mart
        .groupBy("puid")
        .agg(
            F.max_by("genre", F.when(F.col("genre_rank") == 1, 1).otherwise(0)).alias("favorite_genre"),
            F.count("genre").alias("unique_genres_count"),
        )
    )

    base_user_mart = (
        filtered_df
        .groupBy("puid")
        .agg(
            F.round(F.sum("hours"), 2).alias("total_hours"),
            F.countDistinct("msk_business_dt").alias("active_days"),
            F.countDistinct("main_content_id").alias("unique_content_count"),
            F.max("msk_business_dt").alias("last_date"),
            F.countDistinct(
                F.when(F.col("adult_content_flg"), F.col("main_content_id"))
            ).alias("adult_content_count"),
            F.countDistinct(
                F.when(F.col("kids_content_flg"), F.col("main_content_id"))
            ).alias("kids_content_count"),
        )
    )

    content_type_counts = filtered_df.groupBy("puid", "main_content_type").count()
    fav_content_type = (
        content_type_counts.groupBy("puid")
        .agg(F.max_by("main_content_type", "count").alias("favorite_content_type"))
    )

    platform_counts = filtered_df.groupBy("puid", "usage_platform_ru").count()
    favourite_platform = (
        platform_counts.groupBy("puid")
        .agg(F.max_by("usage_platform_ru", "count").alias("favourite_platform"))
    )

    return (
        base_user_mart
        .join(F.broadcast(completion_and_content), "puid", "left")
        .join(F.broadcast(genre_summary), "puid", "left")
        .join(F.broadcast(fav_content_type), "puid", "left")
        .join(F.broadcast(favourite_platform), "puid", "left")
        .select(
            "puid", "total_hours", "active_days", "unique_content_count",
            "unique_genres_count", "completion_rate", "favorite_genre",
            "favorite_content_type", "favourite_platform",
            "last_date", "adult_content_count", "kids_content_count",
        )
    )


def build_content_mart(filtered_df: DataFrame) -> DataFrame:
    per_user = (
        filtered_df
        .groupBy("main_content_id", "puid")
        .agg(
            F.sum("hours").alias("user_total_hours"),
            F.max("main_content_duration_hours").alias("main_content_duration_hours")
        )
        .withColumn(
            "finished_percent",
            F.least(F.lit(100.0), F.col("user_total_hours") / F.col("main_content_duration_hours") * 100)
        )
    )

    total_users = filtered_df.select("puid").distinct().count()

    platforms = (
        filtered_df.groupBy("main_content_id")
        .agg(F.countDistinct("usage_platform_ru").alias("platforms_count"))
    )

    mart = (
        per_user.groupBy("main_content_id")
        .agg(
            F.countDistinct("puid").alias("unique_users"),
            F.round(F.sum("user_total_hours"), 2).alias("total_hours"),
            F.round(F.avg("finished_percent"), 2).alias("avg_finished_percent"),
        )
    )

    return (
        mart
        .join(F.broadcast(platforms), on="main_content_id", how="left")
        .withColumn("popularity", F.round(F.col("unique_users") / F.lit(total_users), 4))
        .select(
            "main_content_id", "unique_users", "total_hours",
            "avg_finished_percent", "platforms_count", "popularity",
        )
    )


def build_daily_user_mart(filtered_df: DataFrame) -> DataFrame:
    mart = (
        filtered_df
        .groupBy("puid", "msk_business_dt")
        .agg(
            F.first(F.date_format("msk_business_dt", "EEEE")).alias("day_of_week"),
            F.countDistinct("main_content_id").alias("content_count"),
            F.round(F.sum("hours"), 2).alias("hours_spent"),
            F.count("*").alias("sessions_count"),
        )
    )

    return mart.select(
        "puid", F.col("msk_business_dt").alias("session_date"),
        "day_of_week", "content_count", "hours_spent", "sessions_count",
    )


def build_all_marts(joined_df: DataFrame) -> dict:
    user_content_mart = build_user_content_mart(joined_df).cache()
    user_content_mart.count()

    user_genre_mart = build_user_genre_mart(joined_df).cache()
    user_genre_mart.count()

    marts = {
        "user_content_mart": user_content_mart,
        "user_genre_mart": user_genre_mart,
        "user_mart": build_user_mart_from_marts(joined_df, user_content_mart, user_genre_mart),
        "content_mart": build_content_mart(joined_df),
        "daily_user_mart": build_daily_user_mart(joined_df),
    }
    return marts


def save_marts(marts: dict, output_path: str):
    output_path = output_path.rstrip("/")
    for mart_name, mart_df in marts.items():
        try:
            print(f"Начало обработки {mart_name}")
            path = f"{output_path}/{mart_name}.parquet"
            mart_df.write.mode("overwrite").parquet(path)
            print(f"Витрина {mart_name} сохранена: {path}")
        except Exception as e:
            print(f"ошибка в {mart_name}: {str(e)}")
            raise e


def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",")]

    spark = get_spark_session()

    try:
        audition_df, content_df, audition_total, content_total = load_data(
            spark, args.audition_path, args.content_path
        )

        audition_df = parse_dates(audition_df)

        analyze_audition_quality(audition_df, audition_total)
        analyze_content_quality(content_df, content_total)

        audition_clean = clean_audition(audition_df)
        content_clean = clean_content(content_df)

        joined_df = join_audition_content(audition_clean, content_clean)

        filtered_df = filter_by_period_and_geo(
            joined_df, args.start_date, args.end_date, args.geo_id
        )
        filtered_df = split_genres(filtered_df)

        filtered_df = filtered_df.cache()
        filtered_row_count = filtered_df.count()
        print(f"Период: {args.start_date} - {args.end_date}")
        print(f"Геолокация: {args.geo_id}")
        print(f"Строк после фильтрации: {filtered_row_count}")

        audition_df.unpersist()
        content_df.unpersist()

        analyze_platform_distribution(filtered_df, columns)
        analyze_content_and_genre_popularity(filtered_df)
        analyze_duration_by_dow_geo(filtered_df)

        try:
            marts = build_all_marts(filtered_df)
        except Exception as e:
            print(f"Критическая ошибка при построении витрин: {e}")
            raise e

        save_marts(marts, args.output_path)

        print("=== План выполнения user_mart ===")
        marts["user_mart"].explain(mode="extended")

        filtered_df.unpersist()
        marts["user_content_mart"].unpersist()
        marts["user_genre_mart"].unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()