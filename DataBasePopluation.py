import mysql.connector
from faker import Faker
import random
from decimal import Decimal

fake = Faker()

# =============================
# 1️⃣ Connect to database
# =============================
conn = mysql.connector.connect(
    host="localhost",
    port=5000,
    user="root",
    password="1497",
    database="E_commerce_Platform_Database"
)
cursor = conn.cursor()
print("✅ Connected to database")

# =============================
# 2️⃣ Clean start: truncate tables
# =============================
cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
tables = [
    "Product_Categories", "Customers", "Warehouses", "Suppliers",
    "Shipping_Providers", "Payment_Methods", "Products",
    "Orders", "Order_Details", "Payments", "Reviews",
    "Loyalty_Programs", "Loyalty_Transactions"
]
for t in tables:
    cursor.execute(f"TRUNCATE TABLE {t}")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
conn.commit()
print("✅ Tables truncated")

# =============================
# 3️⃣ Insert Product Categories
# =============================
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Health', 'Toys', 'Sports']
for c in categories:
    cursor.execute("INSERT INTO Product_Categories (category_name) VALUES (%s)", (c,))
print("✅ Inserted categories")

# =============================
# 4️⃣ Insert Customers
# =============================
for _ in range(50):
    cursor.execute("""
        INSERT INTO Customers (first_name, last_name, email, phone, gender, city, country, loyalty_level, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        fake.first_name()[:50],
        fake.last_name()[:50],
        fake.unique.email()[:50],
        fake.msisdn()[:15],
        random.choice(['Male', 'Female', 'Other']),
        fake.city()[:30],
        fake.country()[:50],
        random.choice(['Bronze', 'Silver', 'Gold', 'Platinum']),
        random.choice([True, True, True, False])
    ))
print("✅ Inserted 50 customers")

# =============================
# 5️⃣ Insert Warehouses
# =============================
for _ in range(5):
    cursor.execute("""
        INSERT INTO Warehouses (warehouse_name, location, capacity)
        VALUES (%s, %s, %s)
    """, (fake.company()[:50] + " Storage", fake.city()[:30], random.randint(1000, 10000)))
print("✅ Warehouses inserted")

# =============================
# 6️⃣ Insert Suppliers
# =============================
for _ in range(10):
    cursor.execute("""
        INSERT INTO Suppliers (supplier_name, country, email)
        VALUES (%s, %s, %s)
    """, (fake.company()[:50], fake.country()[:50], fake.company_email()[:50]))
print("✅ Suppliers inserted")

# =============================
# 7️⃣ Insert Shipping Providers
# =============================
for _ in range(5):
    cursor.execute("""
        INSERT INTO Shipping_Providers (provider_name, contact_email, average_delivery_days)
        VALUES (%s, %s, %s)
    """, (fake.company()[:50], fake.company_email()[:50], random.randint(2, 10)))
print("✅ Shipping providers inserted")

# =============================
# 8️⃣ Insert Payment Methods
# =============================
methods = ['Credit Card', 'PayPal', 'Wire Transfer', 'Cash on Delivery']
for m in methods:
    cursor.execute("INSERT INTO Payment_Methods (method_name) VALUES (%s)", (m,))
print("✅ Payment methods inserted")

conn.commit()

# =============================
# 9️⃣ Fetch IDs for relationships
# =============================
cursor.execute("SELECT category_id FROM Product_Categories"); category_ids = [r[0] for r in cursor.fetchall()]
cursor.execute("SELECT customer_id FROM Customers"); customer_ids = [r[0] for r in cursor.fetchall()]
cursor.execute("SELECT provider_id FROM Shipping_Providers"); provider_ids = [r[0] for r in cursor.fetchall()]
cursor.execute("SELECT method_id FROM Payment_Methods"); method_ids = [r[0] for r in cursor.fetchall()]
cursor.execute("SELECT supplier_id FROM Suppliers"); supplier_ids = [r[0] for r in cursor.fetchall()]

# =============================
# 10️⃣ Insert Products
# =============================
for _ in range(100):
    cursor.execute("""
        INSERT INTO Products (product_name, category_id, supplier_id, unit_price, stock_quantity, reorder_point)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        fake.word().capitalize() + " " + random.choice(["Pro", "Max", "Lite", "Plus"]),
        random.choice(category_ids),
        random.choice(supplier_ids),  # ✅ Now valid
        round(random.uniform(5, 500), 2),
        random.randint(10, 500),
        random.randint(5, 30)
    ))

print("✅ Products inserted")
conn.commit()

# =============================
# 11️⃣ Insert Orders + Order Details
# =============================
cursor.execute("SELECT product_id, unit_price FROM Products")
product_prices = {row[0]: row[1] for row in cursor.fetchall()}  # {product_id: Decimal(unit_price)}

for _ in range(200):
    cust = random.choice(customer_ids)
    order_status = random.choice(['Pending','Processing','Shipped','Delivered','Cancelled'])
    shipping_cost = Decimal(str(round(random.uniform(5, 20), 2)))
    total_amount = Decimal('0.00')

    cursor.execute(
        "INSERT INTO Orders (customer_id, order_status, total_amount, shipping_cost) VALUES (%s,%s,%s,%s)",
        (cust, order_status, total_amount, shipping_cost)
    )
    order_id = cursor.lastrowid

    for _ in range(random.randint(1,5)):
        prod = random.choice(list(product_prices.keys()))
        price = Decimal(str(product_prices[prod]))
        qty = Decimal(random.randint(1,3))
        discount = Decimal(str(round(random.uniform(0, 0.3), 2)))

        cursor.execute("""
            INSERT INTO Order_Details (order_id, product_id, quantity, unit_price_at_purchase, discount)
            VALUES (%s,%s,%s,%s,%s)
        """, (order_id, prod, int(qty), price, discount))

        total_amount += qty * price * (Decimal('1') - discount)

    cursor.execute("UPDATE Orders SET total_amount=%s WHERE order_id=%s", (total_amount, order_id))
print("✅ Orders + Details inserted")

# =============================
# 12️⃣ Insert Payments
# =============================
cursor.execute("SELECT order_id, total_amount FROM Orders WHERE order_status!='Cancelled'")
for order_id, amount in cursor.fetchall():
    cursor.execute("""
        INSERT INTO Payments (order_id, method_id, amount)
        VALUES (%s, %s, %s)
    """, (order_id, random.choice(method_ids), Decimal(amount)))
print("✅ Payments inserted")

# =============================
# 13️⃣ Insert Reviews
# =============================
cursor.execute("SELECT product_id FROM Products")
product_ids = [r[0] for r in cursor.fetchall()]

for _ in range(100):
    cursor.execute("""
        INSERT INTO Reviews (product_id, customer_id, rating, review_text)
        VALUES (%s, %s, %s, %s)
    """, (random.choice(product_ids), random.choice(customer_ids), random.randint(1,5), fake.sentence()[:200]))
print("✅ Reviews inserted")

# =============================
# 14️⃣ Loyalty Programs + Transactions
# =============================
cursor.execute("INSERT INTO Loyalty_Programs (program_name, reward_rate) VALUES ('Standard Rewards',0.05),('VIP Rewards',0.10)")
cursor.execute("SELECT program_id FROM Loyalty_Programs"); program_ids = [r[0] for r in cursor.fetchall()]

for _ in range(100):
    cursor.execute("""
        INSERT INTO Loyalty_Transactions (customer_id, program_id, points_earned, txn_date)
        VALUES (%s, %s, %s, %s)
    """, (random.choice(customer_ids), random.choice(program_ids), random.randint(10,500), fake.date_this_year()))

conn.commit()
cursor.close()
conn.close()
print("\n🎉 Database successfully populated!")
