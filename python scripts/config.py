# Configuration for churn prediction
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telecom Customers Churn.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
TEST_SIZE = 0.3
RANDOM_STATE =42

# Business parameters
AVG_CUSTOMER_VALUE = 1500
RETENTION_COST = 100
