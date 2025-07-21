# scripts/export_all_tables_to_csv.py

import sqlite3
import pandas as pd
import os

def export_all_sqlite_tables_to_csv(db_path, output_dir):
    """
    Connects to an SQLite database, retrieves all table names,
    and exports the full content of each table to a separate CSV file.

    Args:
        db_path (str): The file path to the SQLite database.
        output_dir (str): The directory where the CSV files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        print("Please ensure db.sqlite3 is in your 'data/raw/' folder.")
        return

    print(f"--- Exporting all tables from: {db_path} ---")

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database to export.")
            return

        # Iterate through each table and export its data
        for table_name_tuple in tables:
            table_name = table_name_tuple[0] # Extract the table name string from the tuple
            print(f"Exporting table: {table_name}...")

            # Read the entire table into a pandas DataFrame
            # Using SELECT * to get all columns and no WHERE clause for all rows
            query = f"SELECT * FROM `{table_name}`;" # Backticks for table names in case of special characters

            try:
                df = pd.read_sql_query(query, conn)
                
                # Define output CSV path
                output_csv_path = os.path.join(output_dir, f"{table_name}.csv")
                
                # Save DataFrame to CSV
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"  - Saved {df.shape[0]} rows from '{table_name}' to '{output_csv_path}'")
            except pd.io.sql.DatabaseError as e:
                print(f"  Error exporting table '{table_name}': {e}. Skipping this table.")
            except Exception as e:
                print(f"  An unexpected error occurred while exporting '{table_name}': {e}. Skipping.")


    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\n--- All table exports complete. Connection closed. ---")

# --- Main execution ---
if __name__ == "__main__":
    # Define the path to your database file relative to the project root
    # Assuming this script is in `climbing-ai-routes/scripts/`
    db_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw', 'db.sqlite3')
    
    # Define the output directory for the CSVs (e.g., in data/raw/)
    output_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw')

    export_all_sqlite_tables_to_csv(db_file_path, output_directory)
