import mysql.connector


# =========================
# DATABASE CONNECTION
# =========================
def connect_db():

    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Pranay@3330",
        database="stock_db"
    )


# =========================
# SAVE PREDICTION
# =========================
def save_prediction(stock, predicted, actual):

    try:

        conn = connect_db()
        cursor = conn.cursor()

        query = """
        INSERT INTO predictions
        (stock, predicted_price, actual_price, created_at)
        VALUES (%s, %s, %s, NOW())
        """

        cursor.execute(
            query,
            (
                str(stock),
                float(predicted),
                float(actual)
            )
        )

        conn.commit()

        print("✅ Prediction Saved")

        cursor.close()
        conn.close()

    except Exception as e:

        print("❌ SAVE ERROR:", e)


# =========================
# GET HISTORY
# =========================
def get_history():

    try:

        conn = connect_db()
        cursor = conn.cursor()

        query = """
        SELECT
        id,
        stock,
        predicted_price,
        actual_price,
        created_at
        FROM predictions
        ORDER BY id DESC
        """

        cursor.execute(query)

        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return data

    except Exception as e:

        print("❌ HISTORY ERROR:", e)

        return []